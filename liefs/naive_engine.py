"""
Stage 1: Naive autoregressive generation engine.

This engine performs FULL RECOMPUTATION at every generation step — it feeds
the entire token sequence (prompt + all generated tokens so far) through every
layer of the model to produce the next token. This is intentionally wasteful.

WHY THIS IS SLOW (the key insight for interviews):
------------------------------------------------------
Consider generating N tokens for a prompt of length P:

At step 1: forward pass processes P+1 tokens
At step 2: forward pass processes P+2 tokens
...
At step N: forward pass processes P+N tokens

Total tokens processed across all steps: sum(P+i for i=1..N) = N*P + N*(N+1)/2

The attention mechanism in each layer computes Q @ K^T, which is O(seq_len^2)
per layer. So total compute scales as O(sum((P+i)^2 for i=1..N)).

With KV-caching (Stage 2), at step i we only compute Q for the NEW token and
use cached K,V from prior steps. This makes each step O(P+i) for attention
(just the dot product of 1 query against P+i cached keys), and total compute
becomes O(N*(P+N)) — removing the quadratic-in-step-count overhead.

Even ignoring attention, the linear projections (Q/K/V/O, MLP) are recomputed
for ALL tokens at every step, which is O(N^2 * d_model * d_hidden) total
instead of O(N * d_model * d_hidden) with caching.

SIMPLIFICATION vs PRODUCTION:
- We call model.model.forward() from HuggingFace, passing input_ids directly.
- Production systems (vLLM, TGI) reimplement the entire forward pass for
  memory layout control, custom CUDA kernels, and paged attention.
- We use greedy decoding (argmax) only. No temperature, top-k, top-p, or
  repetition penalty. Greedy is deterministic, which makes testing easier.
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


class NaiveEngine:
    """Naive autoregressive generation with full recomputation per step.

    This is our Stage 1 baseline — intentionally slow, maximally simple.

    Attributes:
        model: The HuggingFace causal LM model (e.g., Qwen2ForCausalLM).
        tokenizer: The corresponding tokenizer.
        eos_token_ids: Set of token IDs that signal generation should stop.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = get_eos_token_ids(tokenizer)

    @torch.no_grad()
    def forward_pass(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the full model forward pass and return logits for the last token.

        Args:
            input_ids: Token IDs, shape (1, seq_len). The FULL sequence
                including prompt and all generated tokens so far.

        Returns:
            Logits for the last position only, shape (1, vocab_size).
        """
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,  # No KV-cache — full recomputation!
        )
        return outputs.logits[:, -1, :]  # (1, vocab_size)

    def greedy_decode(self, logits: torch.Tensor) -> int:
        """Select the highest-probability token (greedy / argmax decoding)."""
        return logits.argmax(dim=-1).item()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple[list[int], GenerationMetrics]:
        """Generate tokens autoregressively with full recomputation."""
        reset_vram_stats()

        generated_ids: list[int] = []
        current_ids = input_ids.clone()
        token_times_ms: list[float] = []

        for step in range(max_new_tokens):
            with cuda_timer() as elapsed:
                logits = self.forward_pass(current_ids)
                next_token_id = self.greedy_decode(logits)

            token_times_ms.append(elapsed())

            if next_token_id in self.eos_token_ids:
                break

            generated_ids.append(next_token_id)
            next_token_tensor = torch.tensor(
                [[next_token_id]], device=current_ids.device, dtype=current_ids.dtype
            )
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

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
        """High-level API: takes a string prompt, returns generated text + metrics."""
        from liefs.model_loader import format_chat_prompt

        input_ids = format_chat_prompt(self.tokenizer, prompt, system_message)
        generated_ids, metrics = self.generate(input_ids, max_new_tokens)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, metrics
