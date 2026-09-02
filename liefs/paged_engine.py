"""
Stage 4: Paged attention engine.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liefs.paged_attention import BlockAllocator, PagedKVCache
from liefs.utils import (
    GenerationMetrics,
    compute_generation_metrics,
    cuda_timer,
    get_eos_token_ids,
    get_peak_vram_mb,
    reset_vram_stats,
)


class PagedEngine:
    """Generation engine with paged KV-cache storage."""

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

        config = model.config
        self.num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 24))
        num_attn_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 16))
        self.num_kv_heads = getattr(config, 'num_key_value_heads', num_attn_heads)
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', 1024))
        self.head_dim = getattr(config, 'head_dim', hidden_size // num_attn_heads if num_attn_heads > 0 else 64)

        self.allocator = BlockAllocator(
            num_blocks=max_num_blocks,
            block_size=block_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        self.eos_token_ids = get_eos_token_ids(tokenizer)

    def _extract_kv_from_cache(self, cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Extract per-layer (K, V) tensors from the model's cache object."""
        kv_pairs = []
        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            for layer_idx in range(len(cache.key_cache)):
                kv_pairs.append((cache.key_cache[layer_idx], cache.value_cache[layer_idx]))
        else:
            for layer_cache in cache:
                kv_pairs.append((layer_cache[0], layer_cache[1]))
        return kv_pairs

    def _build_cache_from_paged(self, paged_cache: PagedKVCache):
        """Reconstruct a model-compatible cache from paged blocks."""
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
        """Generate tokens using paged KV-cache storage."""
        reset_vram_stats()
        generated_ids: list[int] = []
        token_times_ms: list[float] = []

        paged_cache = PagedKVCache(self.allocator)

        try:
            with cuda_timer() as prefill_elapsed:
                outputs = self.model(input_ids=input_ids, use_cache=True)
                logits = outputs.logits[:, -1, :]

                kv_pairs = self._extract_kv_from_cache(outputs.past_key_values)
                for layer_idx, (k, v) in enumerate(kv_pairs):
                    paged_cache.append_kv(layer_idx, k, v)

            token_times_ms.append(prefill_elapsed())

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
                    reconstructed_cache = self._build_cache_from_paged(paged_cache)

                    outputs = self.model(
                        input_ids=next_token_tensor,
                        past_key_values=reconstructed_cache,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]

                    new_kv_pairs = self._extract_kv_from_cache(outputs.past_key_values)
                    for layer_idx, (k, v) in enumerate(new_kv_pairs):
                        new_k = k[:, :, -1:, :]
                        new_v = v[:, :, -1:, :]
                        paged_cache.append_kv(layer_idx, new_k, new_v)

                token_times_ms.append(decode_elapsed())

        finally:
            paged_cache.free_all()

        metrics = self._compute_metrics(token_times_ms, len(generated_ids))
        return generated_ids, metrics

    def _compute_metrics(
        self, token_times_ms: list[float], num_generated: int
    ) -> GenerationMetrics:
        return compute_generation_metrics(token_times_ms, num_generated)

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
