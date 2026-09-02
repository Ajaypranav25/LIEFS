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
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liefs.utils import (
    GenerationMetrics,
    compute_generation_metrics,
    cuda_timer,
    get_eos_token_ids,
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
        self.eos_token_ids = get_eos_token_ids(tokenizer)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, object]:
        """Prefill phase: process the entire prompt in one forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
        )
        return outputs.logits[:, -1, :], outputs.past_key_values

    @torch.no_grad()
    def decode_step(
        self, token_id: torch.Tensor, kv_cache: object
    ) -> tuple[torch.Tensor, object]:
        """Decode phase: process a single new token using the KV-cache."""
        outputs = self.model(
            input_ids=token_id,
            past_key_values=kv_cache,
            use_cache=True,
        )
        return outputs.logits[:, -1, :], outputs.past_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple[list[int], GenerationMetrics]:
        """Generate tokens with KV-cache (prefill + decode loop)."""
        reset_vram_stats()
        generated_ids: list[int] = []
        token_times_ms: list[float] = []

        # ── Phase 1: Prefill ──────────────────────────────────────────────
        with cuda_timer() as prefill_elapsed:
            logits, kv_cache = self.prefill(input_ids)

        token_times_ms.append(prefill_elapsed())

        # ── Phase 2: Decode ───────────────────────────────────────────────
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
                logits, kv_cache = self.decode_step(next_token_tensor, kv_cache)

            token_times_ms.append(decode_elapsed())

        metrics = self._compute_metrics(token_times_ms, len(generated_ids))
        return generated_ids, metrics

    def _compute_metrics(
        self, token_times_ms: list[float], num_generated: int
    ) -> GenerationMetrics:
        """Compute generation metrics from per-token timing data."""
        return compute_generation_metrics(token_times_ms, num_generated)

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
