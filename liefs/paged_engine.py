"""
Stage 4: Paged attention engine.

This engine uses block-based KV-cache storage from paged_attention.py
instead of contiguous tensors. It hooks into the model's forward pass
to intercept and redirect KV-cache storage to/from paged blocks.

APPROACH:
We run the model's forward pass layer by layer and intercept the KV
values at each attention layer. Instead of letting the model store them
in a contiguous DynamicCache, we store them in our PagedKVCache blocks
and retrieve them when the attention layer needs past K/V.

Since we can't easily modify the HF model's internal attention to use
paged blocks natively (that would require custom CUDA kernels like vLLM),
we use a hybrid approach:
1. Run the model forward with use_cache=True
2. After each forward pass, extract the new K/V from the model's cache
3. Store them in our paged blocks
4. Before the next forward pass, reconstruct a contiguous cache from
   our paged blocks to pass to the model

This demonstrates the STORAGE benefit of paging (reduced fragmentation,
efficient memory accounting) without the COMPUTE benefit (which requires
custom attention kernels that read directly from block tables).

SIMPLIFICATION vs PRODUCTION:
    - vLLM's PagedAttention CUDA kernel reads K/V directly from block tables
      during attention computation. No gather/scatter overhead.
    - We gather from blocks → contiguous tensor → pass to model. This adds
      overhead but demonstrates the memory management concept.
    - In production, the block allocator is tightly integrated with the
      scheduler for memory-aware scheduling decisions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liefs.paged_attention import BlockAllocator, PagedKVCache
from liefs.utils import (
    GenerationMetrics,
    cuda_timer,
    get_peak_vram_mb,
    reset_vram_stats,
)


class PagedEngine:
    """Generation engine with paged KV-cache storage.

    Uses fixed-size blocks for KV-cache instead of contiguous tensors,
    demonstrating the OS virtual memory paging concept.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        block_size: int = 16,
        max_num_blocks: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Model architecture parameters (Qwen2.5-0.5B)
        config = model.config
        self.num_layers = config.num_hidden_layers      # 24
        self.num_kv_heads = config.num_key_value_heads  # 2
        self.head_dim = config.hidden_size // config.num_attention_heads  # 64

        # Create block allocator
        self.allocator = BlockAllocator(
            num_blocks=max_num_blocks,
            block_size=block_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        # EOS tokens
        self.eos_token_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(tokenizer.eos_token_id)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
            self.eos_token_ids.add(im_end_id)

    def _extract_kv_from_cache(self, cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Extract per-layer (K, V) tensors from the model's cache object."""
        kv_pairs = []
        # DynamicCache in modern transformers
        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            for layer_idx in range(len(cache.key_cache)):
                kv_pairs.append((cache.key_cache[layer_idx], cache.value_cache[layer_idx]))
        else:
            # Older tuple format
            for layer_cache in cache:
                kv_pairs.append((layer_cache[0], layer_cache[1]))
        return kv_pairs

    def _build_cache_from_paged(self, paged_cache: PagedKVCache):
        """Reconstruct a model-compatible cache from paged blocks.

        Gathers K/V from all blocks into contiguous tensors that the
        HF model can consume as past_key_values.
        """
        from transformers import DynamicCache

        cache = DynamicCache()

        for layer_idx in range(self.num_layers):
            key, value = paged_cache.get_kv(layer_idx)
            if key is not None:
                cache.update(key, value, layer_idx)

        return cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple[list[int], GenerationMetrics]:
        """Generate tokens using paged KV-cache storage.

        The flow:
        1. Prefill: run model forward, extract K/V, store in paged blocks
        2. Decode loop:
           a. Reconstruct contiguous cache from paged blocks
           b. Forward 1 token with reconstructed cache
           c. Extract new K/V, store in paged blocks
           d. Repeat
        """
        reset_vram_stats()
        generated_ids: list[int] = []
        token_times_ms: list[float] = []

        # Create paged cache for this sequence
        paged_cache = PagedKVCache(self.allocator)

        try:
            # ── Prefill ──
            with cuda_timer() as prefill_elapsed:
                outputs = self.model(input_ids=input_ids, use_cache=True)
                logits = outputs.logits[:, -1, :]

                # Extract K/V from model's cache and store in paged blocks
                kv_pairs = self._extract_kv_from_cache(outputs.past_key_values)
                for layer_idx, (k, v) in enumerate(kv_pairs):
                    paged_cache.append_kv(layer_idx, k, v)

            token_times_ms.append(prefill_elapsed())

            # ── Decode loop ──
            for step in range(max_new_tokens):
                next_token_id = logits.argmax(dim=-1).item()

                if next_token_id in self.eos_token_ids:
                    break

                generated_ids.append(next_token_id)
                if len(generated_ids) >= max_new_tokens:
                    break

                next_token_tensor = torch.tensor(
                    [[next_token_id]], device=input_ids.device, dtype=input_ids.dtype
                )

                with cuda_timer() as decode_elapsed:
                    # Reconstruct contiguous cache from paged blocks
                    reconstructed_cache = self._build_cache_from_paged(paged_cache)

                    # Forward single token with reconstructed cache
                    outputs = self.model(
                        input_ids=next_token_tensor,
                        past_key_values=reconstructed_cache,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]

                    # Extract the NEW K/V (just the new token) and store in pages
                    new_kv_pairs = self._extract_kv_from_cache(outputs.past_key_values)
                    for layer_idx, (k, v) in enumerate(new_kv_pairs):
                        # The model returns the FULL cache including old entries.
                        # We only want the LAST token's K/V (the new one).
                        new_k = k[:, :, -1:, :]
                        new_v = v[:, :, -1:, :]
                        paged_cache.append_kv(layer_idx, new_k, new_v)

                token_times_ms.append(decode_elapsed())

        finally:
            # Always free blocks back to the pool
            paged_cache.free_all()

        metrics = self._compute_metrics(token_times_ms, len(generated_ids))
        return generated_ids, metrics

    def _compute_metrics(
        self, token_times_ms: list[float], num_generated: int
    ) -> GenerationMetrics:
        if not token_times_ms:
            return GenerationMetrics()

        total_time = sum(token_times_ms)
        ttft = token_times_ms[0]
        tpot = sum(token_times_ms[1:]) / (len(token_times_ms) - 1) if len(token_times_ms) > 1 else 0.0
        tokens_per_sec = (num_generated / total_time * 1000.0) if total_time > 0 else 0.0

        return GenerationMetrics(
            ttft_ms=ttft,
            tpot_ms=tpot,
            tokens_per_sec=tokens_per_sec,
            total_tokens_generated=num_generated,
            total_time_ms=total_time,
            peak_vram_mb=get_peak_vram_mb(),
        )

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        system_message: str = "You are a helpful assistant.",
    ) -> tuple[str, GenerationMetrics]:
        from liefs.model_loader import format_chat_prompt
        input_ids = format_chat_prompt(self.tokenizer, prompt, system_message)
        generated_ids, metrics = self.generate(input_ids, max_new_tokens)
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True), metrics
