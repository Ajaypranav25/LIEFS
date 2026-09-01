"""
Stage 2: KV-Cache inference engine.

THE KEY OPTIMIZATION — WHY KV-CACHE MATTERS:
=============================================

In a transformer's self-attention, each layer computes:
    Q = X @ W_q    (queries)
    K = X @ W_k    (keys)
    V = X @ W_v    (values)
    Attn = softmax(Q @ K^T / sqrt(d)) @ V

During autoregressive generation, at step t we have tokens [x_1, ..., x_t].
The NAIVE approach (Stage 1) recomputes Q, K, V for ALL t tokens, even though
the K and V for tokens x_1..x_{t-1} haven't changed — they only depend on the
input token, not on future tokens (thanks to causal masking).

THE INSIGHT: Cache the K and V tensors from previous steps. At step t, only
compute Q, K, V for the NEW token x_t, then concatenate the new K, V with
the cached K, V from steps 1..t-1.

COMPLEXITY ANALYSIS:
    Naive (Stage 1):
        Per step t: O(t * d_model * d_hidden) for linear projections
                    O(t^2 * d_head) for attention
        Total over N steps: O(N^2 * d + N^3) — quadratic/cubic!

    KV-Cache (Stage 2):
        Prefill (step 0): O(P * d_model * d_hidden) + O(P^2 * d_head)
            Process all P prompt tokens at once.
        Per decode step t: O(d_model * d_hidden) for projections (1 token)
                           O(t * d_head) for attention (1 query vs t keys)
        Total over N steps: O(N * d + N^2) — linear/quadratic!

    The savings come from:
    1. Linear layers: O(N * d) instead of O(N^2 * d)  [huge win]
    2. Attention: O(N^2) instead of O(N^3)             [big win at long seq]

TWO-PHASE GENERATION:
    The KV-cache splits generation into two distinct phases:

    1. PREFILL: Process the entire prompt in one forward pass.
       - All prompt tokens processed in parallel (GPU-efficient)
       - Produces KV-cache for all prompt positions
       - Returns logits for the first generated token
       - This is compute-bound (large batch of tokens)

    2. DECODE: Generate tokens one at a time.
       - Each step processes only 1 token through the model
       - Reads from the (growing) KV-cache
       - This is memory-bandwidth-bound (reading large cache, small compute)

    This prefill/decode split is fundamental to LLM serving and explains why
    TTFT and TPOT behave differently under load.

IMPLEMENTATION APPROACH:
    We use the HuggingFace model's forward() method with use_cache=True and
    pass past_key_values explicitly. We DON'T call model.generate() — we
    manage the cache lifecycle ourselves.

    In newer transformers, past_key_values uses a DynamicCache object. We let
    the model create and update this cache, but WE control when to create it,
    when to pass it, and when to discard it.

SIMPLIFICATION vs PRODUCTION:
    - We rely on HF's internal attention to concatenate new K/V with cached K/V.
      Production systems (vLLM) implement custom CUDA kernels for this.
    - Our cache is a single contiguous tensor per layer. vLLM uses paged blocks
      (we'll implement this in Stage 4).
    - We don't handle cache eviction, prefix sharing, or speculative decoding.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liefs.utils import (
    GenerationMetrics,
    cuda_timer,
    get_peak_vram_mb,
    reset_vram_stats,
)


class KVCacheEngine:
    """Autoregressive generation with KV-cache (prefill + decode).

    This is Stage 2 — each decode step only processes 1 new token through
    the model, using cached key/value tensors from all previous steps.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.eos_token_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(tokenizer.eos_token_id)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
            self.eos_token_ids.add(im_end_id)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, object]:
        """Prefill phase: process the entire prompt in one forward pass.

        This is the first phase of KV-cache generation. We process all prompt
        tokens at once (GPU-efficient parallel processing) and get back:
        1. The logits for predicting the first output token
        2. The KV-cache containing K/V tensors for all prompt positions

        Args:
            input_ids: Full prompt token IDs, shape (1, prompt_len).

        Returns:
            (last_token_logits, kv_cache) where:
            - last_token_logits: shape (1, vocab_size)
            - kv_cache: the model's cache object (DynamicCache in modern
              transformers, tuple of (K, V) tensors per layer in older versions)
        """
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,  # Tell the model to return KV-cache
        )

        # outputs.past_key_values contains the cached K/V tensors.
        # For Qwen2.5-0.5B with 24 layers and 2 KV heads:
        #   Each layer's K: shape (batch=1, num_kv_heads=2, seq_len, head_dim=64)
        #   Each layer's V: shape (batch=1, num_kv_heads=2, seq_len, head_dim=64)
        #   Total cache size: 2 * 24 * 2 * seq_len * 64 * 2 bytes = 12 KB/token
        return outputs.logits[:, -1, :], outputs.past_key_values

    @torch.no_grad()
    def decode_step(
        self, token_id: torch.Tensor, kv_cache: object
    ) -> tuple[torch.Tensor, object]:
        """Decode phase: process a single new token using the KV-cache.

        This is where the speedup comes from. Instead of processing ALL tokens
        through the model (like Stage 1), we only process the ONE new token.
        The model uses the cached K/V tensors for attention against all
        previous positions.

        Args:
            token_id: Single token ID, shape (1, 1).
            kv_cache: Cache from prefill or previous decode step.

        Returns:
            (logits, updated_kv_cache) where logits has shape (1, vocab_size).
        """
        outputs = self.model(
            input_ids=token_id,
            past_key_values=kv_cache,  # Pass the cache — model reads K/V from it
            use_cache=True,            # Continue caching (append new K/V)
        )

        return outputs.logits[:, -1, :], outputs.past_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple[list[int], GenerationMetrics]:
        """Generate tokens with KV-cache (prefill + decode loop).

        Phase 1 — Prefill:
            Process ALL prompt tokens in one forward pass → KV-cache + first logits

        Phase 2 — Decode:
            For each new token:
                1. Argmax the logits → next token
                2. Forward ONLY that one token with the KV-cache
                3. Get updated KV-cache + next logits
                4. Repeat until EOS or max tokens

        Args:
            input_ids: Tokenized prompt, shape (1, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Tuple of (generated_token_ids, metrics).
        """
        reset_vram_stats()
        generated_ids: list[int] = []
        token_times_ms: list[float] = []

        # ── Phase 1: Prefill ──────────────────────────────────────────────
        # Process the entire prompt at once. This is a single forward pass
        # over all prompt tokens — expensive but done only once.
        with cuda_timer() as prefill_elapsed:
            logits, kv_cache = self.prefill(input_ids)

        # The prefill time IS the TTFT (time-to-first-token) since it produces
        # the logits we need to select the first output token.
        token_times_ms.append(prefill_elapsed())

        # ── Phase 2: Decode ───────────────────────────────────────────────
        for step in range(max_new_tokens):
            # Select next token (greedy)
            next_token_id = logits.argmax(dim=-1).item()

            # Check EOS
            if next_token_id in self.eos_token_ids:
                break

            generated_ids.append(next_token_id)
            if len(generated_ids) >= max_new_tokens:
                break

            # Prepare the single new token for the next forward pass
            next_token_tensor = torch.tensor(
                [[next_token_id]], device=input_ids.device, dtype=input_ids.dtype
            )

            # Forward ONLY the new token, using cached K/V for all previous tokens.
            # This is the key optimization: 1 token through the model instead of
            # (prompt_len + step) tokens.
            with cuda_timer() as decode_elapsed:
                logits, kv_cache = self.decode_step(next_token_tensor, kv_cache)

            token_times_ms.append(decode_elapsed())

        # ── Compute metrics ───────────────────────────────────────────────
        metrics = self._compute_metrics(token_times_ms, len(generated_ids))
        return generated_ids, metrics

    def _compute_metrics(
        self, token_times_ms: list[float], num_generated: int
    ) -> GenerationMetrics:
        """Compute generation metrics from per-token timing data."""
        if not token_times_ms:
            return GenerationMetrics()

        total_time = sum(token_times_ms)
        ttft = token_times_ms[0]  # Prefill time = TTFT

        # TPOT: average decode time (excludes prefill)
        if len(token_times_ms) > 1:
            tpot = sum(token_times_ms[1:]) / (len(token_times_ms) - 1)
        else:
            tpot = 0.0

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
        """High-level API: string prompt → generated text + metrics."""
        from liefs.model_loader import format_chat_prompt

        input_ids = format_chat_prompt(self.tokenizer, prompt, system_message)
        generated_ids, metrics = self.generate(input_ids, max_new_tokens)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, metrics
