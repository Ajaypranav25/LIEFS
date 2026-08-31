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
  We don't reimplement the transformer layers from scratch — that comes in
  Stage 2 when we need to control KV-cache storage.
- Production systems (vLLM, TGI) reimplement the entire forward pass for
  memory layout control, custom CUDA kernels, and paged attention.
- We use greedy decoding (argmax) only. No temperature, top-k, top-p, or
  repetition penalty. Greedy is deterministic, which makes testing easier.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liefs.utils import (
    GenerationMetrics,
    cuda_timer,
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

        # Qwen2.5 has multiple stop tokens:
        #   <|im_end|> (151645) — end of a chat turn
        #   <|endoftext|> (151643) — end of text
        # We stop on either one.
        self.eos_token_ids: set[int] = set()

        if tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(tokenizer.eos_token_id)

        # Also grab the <|im_end|> token which is the chat turn terminator
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
            self.eos_token_ids.add(im_end_id)

    @torch.no_grad()
    def forward_pass(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the full model forward pass and return logits for the last token.

        Args:
            input_ids: Token IDs, shape (1, seq_len). The FULL sequence
                including prompt and all generated tokens so far.

        Returns:
            Logits for the last position only, shape (1, vocab_size).

        Note:
            This is where the waste happens — we process ALL tokens through
            ALL layers every time, even though only the last token's logits
            are new. The prefix tokens produce the same hidden states as
            in the previous step, but we recompute them anyway.
        """
        # model.model is the Qwen2Model (transformer body without lm_head).
        # We call the full Qwen2ForCausalLM forward which runs:
        #   1. embed_tokens(input_ids) → (1, seq_len, 896)
        #   2. 24x Qwen2DecoderLayer (RMSNorm → GQA Attention → RMSNorm → SwiGLU MLP)
        #   3. final RMSNorm
        #   4. lm_head linear projection → (1, seq_len, 151936)
        #
        # We pass use_cache=False to explicitly disable KV-caching.
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,  # No KV-cache — full recomputation!
        )

        # outputs.logits has shape (1, seq_len, vocab_size).
        # We only need the logits at the LAST position — that's the prediction
        # for the next token. All other positions' logits are wasted compute.
        return outputs.logits[:, -1, :]  # (1, vocab_size)

    def greedy_decode(self, logits: torch.Tensor) -> int:
        """Select the highest-probability token (greedy / argmax decoding).

        Args:
            logits: Logits for a single position, shape (1, vocab_size).

        Returns:
            The token ID with the highest logit value.

        Note:
            Greedy decoding is deterministic — same input always produces
            same output. This is crucial for reproducible benchmarks and
            correctness testing. We'll add sampling (temperature, top-k,
            top-p) in later stages.
        """
        return logits.argmax(dim=-1).item()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple[list[int], GenerationMetrics]:
        """Generate tokens autoregressively with full recomputation.

        The core loop:
            1. Run forward pass on the FULL sequence (prompt + generated so far)
            2. Take argmax of last-position logits → next token
            3. Append next token to sequence
            4. Repeat until EOS or max_new_tokens

        Args:
            input_ids: Tokenized prompt, shape (1, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Tuple of (generated_token_ids, metrics).
            generated_token_ids contains ONLY the newly generated tokens
            (not the prompt).
        """
        reset_vram_stats()

        generated_ids: list[int] = []
        current_ids = input_ids.clone()  # We'll append to this each step
        prompt_len = input_ids.shape[1]

        # Track per-token times for TTFT and TPOT calculation
        token_times_ms: list[float] = []

        for step in range(max_new_tokens):
            # Time each individual token generation step
            with cuda_timer() as elapsed:
                # FULL recomputation: feed ALL tokens through ALL layers
                logits = self.forward_pass(current_ids)
                next_token_id = self.greedy_decode(logits)

            token_times_ms.append(elapsed())

            # Check for end-of-sequence
            if next_token_id in self.eos_token_ids:
                break

            # Append the new token and continue
            generated_ids.append(next_token_id)
            next_token_tensor = torch.tensor(
                [[next_token_id]], device=current_ids.device, dtype=current_ids.dtype
            )
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

        # Compute metrics
        metrics = self._compute_metrics(token_times_ms, len(generated_ids))
        return generated_ids, metrics

    def _compute_metrics(
        self, token_times_ms: list[float], num_generated: int
    ) -> GenerationMetrics:
        """Compute generation metrics from per-token timing data.

        Args:
            token_times_ms: Wall time in ms for each generation step.
            num_generated: Number of tokens actually generated (excluding EOS).
        """
        if not token_times_ms:
            return GenerationMetrics()

        total_time = sum(token_times_ms)
        ttft = token_times_ms[0]

        # TPOT excludes the first token (which includes prefill cost)
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
        """High-level API: takes a string prompt, returns generated text + metrics.

        This is a convenience wrapper around generate() that handles
        tokenization and decoding.

        Args:
            prompt: User message string.
            max_new_tokens: Maximum tokens to generate.
            system_message: System prompt for the chat template.

        Returns:
            Tuple of (generated_text, metrics).
        """
        from liefs.model_loader import format_chat_prompt

        input_ids = format_chat_prompt(self.tokenizer, prompt, system_message)
        generated_ids, metrics = self.generate(input_ids, max_new_tokens)

        # Decode only the generated tokens (not the prompt)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, metrics
