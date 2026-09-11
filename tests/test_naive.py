"""
Correctness tests for Stage 1: Naive generation engine.

These tests verify that our hand-rolled generation loop produces correct,
deterministic output. They do NOT test performance — that's the benchmark
script's job.

Usage:
    python -m pytest tests/test_naive.py -v
"""

import pytest
import torch

from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.naive_engine import NaiveEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────
# We use module-scoped fixtures so the model is loaded once for all tests,
# not once per test (loading takes ~5s and 1GB VRAM).

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model + tokenizer once for the entire test module."""
    model, tokenizer = load_model_and_tokenizer()
    return model, tokenizer


@pytest.fixture(scope="module")
def engine(model_and_tokenizer):
    """Create a NaiveEngine from the loaded model."""
    model, tokenizer = model_and_tokenizer
    return NaiveEngine(model, tokenizer)


@pytest.fixture(scope="module")
def tokenizer(model_and_tokenizer):
    """Convenience fixture for just the tokenizer."""
    _, tokenizer = model_and_tokenizer
    return tokenizer


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Greedy decoding must be deterministic — same input → same output."""

    def test_same_prompt_same_output(self, engine, tokenizer):
        """Running the same prompt twice must produce identical token sequences."""
        input_ids = format_chat_prompt(tokenizer, "What is the capital of France?")

        ids_1, _ = engine.generate(input_ids, max_new_tokens=32)
        ids_2, _ = engine.generate(input_ids, max_new_tokens=32)

        assert ids_1 == ids_2, (
            f"Non-deterministic output!\n"
            f"  Run 1: {ids_1[:10]}...\n"
            f"  Run 2: {ids_2[:10]}..."
        )


class TestEOS:
    """Generation must stop at end-of-sequence tokens."""

    def test_stops_at_eos(self, engine, tokenizer):
        """Short answers should stop before max_new_tokens."""
        input_ids = format_chat_prompt(tokenizer, "What is 2+2?")

        generated_ids, metrics = engine.generate(input_ids, max_new_tokens=256)

        # A simple math answer should NOT use all 256 tokens
        assert metrics.total_tokens_generated < 256, (
            f"Expected generation to stop early (got {metrics.total_tokens_generated} tokens)"
        )
        assert metrics.total_tokens_generated > 0, "Generated zero tokens"

    def test_eos_not_in_output(self, engine, tokenizer):
        """EOS tokens should not appear in the returned generated_ids."""
        input_ids = format_chat_prompt(tokenizer, "Say hello.")

        generated_ids, _ = engine.generate(input_ids, max_new_tokens=64)

        for eos_id in engine.eos_token_ids:
            assert eos_id not in generated_ids, (
                f"EOS token {eos_id} found in generated output"
            )


class TestCoherence:
    """Basic sanity checks that the model produces sensible output."""

    def test_math_answer(self, engine, tokenizer):
        """The model should answer basic math correctly."""
        input_ids = format_chat_prompt(tokenizer, "What is 2+2? Answer with just the number.")

        generated_ids, _ = engine.generate(input_ids, max_new_tokens=32)
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        assert "4" in text, f"Expected '4' in response, got: {text!r}"

    def test_nonempty_output(self, engine, tokenizer):
        """Generation should produce at least some text."""
        input_ids = format_chat_prompt(tokenizer, "Tell me a fact about the moon.")

        generated_ids, _ = engine.generate(input_ids, max_new_tokens=64)
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        assert len(text.strip()) > 0, "Generated empty text"


class TestMaxTokens:
    """Generation must respect the max_new_tokens limit."""

    def test_respects_max_new_tokens(self, engine, tokenizer):
        """Should not generate more tokens than max_new_tokens."""
        max_tokens = 10
        input_ids = format_chat_prompt(
            tokenizer, "Write a very long essay about the history of computing."
        )

        generated_ids, metrics = engine.generate(input_ids, max_new_tokens=max_tokens)

        assert len(generated_ids) <= max_tokens, (
            f"Generated {len(generated_ids)} tokens, max was {max_tokens}"
        )

    def test_single_token(self, engine, tokenizer):
        """Should work with max_new_tokens=1."""
        input_ids = format_chat_prompt(tokenizer, "Hi")

        generated_ids, metrics = engine.generate(input_ids, max_new_tokens=1)

        assert len(generated_ids) <= 1


class TestMetrics:
    """Verify that metrics are populated correctly."""

    def test_metrics_populated(self, engine, tokenizer):
        """All metric fields should have sensible values after generation."""
        input_ids = format_chat_prompt(tokenizer, "What is Python?")

        _, metrics = engine.generate(input_ids, max_new_tokens=32)

        assert metrics.total_tokens_generated > 0
        assert metrics.total_time_ms > 0
        assert metrics.ttft_ms > 0
        assert metrics.tokens_per_sec > 0

        import torch
        if torch.cuda.is_available():
            assert metrics.peak_vram_mb > 0
        else:
            assert metrics.peak_vram_mb >= 0

    def test_ttft_less_than_total(self, engine, tokenizer):
        """TTFT should be less than total generation time."""
        input_ids = format_chat_prompt(tokenizer, "Explain gravity in one sentence.")

        _, metrics = engine.generate(input_ids, max_new_tokens=64)

        if metrics.total_tokens_generated > 1:
            assert metrics.ttft_ms < metrics.total_time_ms
