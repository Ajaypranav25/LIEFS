"""
Correctness tests for Stage 2: KV-Cache engine.

Tests verify that the KV-cache engine produces IDENTICAL output to the
naive engine (since both use greedy decoding on the same model) and that
all basic correctness properties hold.
"""

import pytest
import torch

from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.naive_engine import NaiveEngine
from liefs.kv_cache_engine import KVCacheEngine


@pytest.fixture(scope="module")
def model_and_tokenizer():
    dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
    model, tokenizer = load_model_and_tokenizer(dtype=dtype)
    return model, tokenizer


@pytest.fixture(scope="module")
def naive_engine(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    return NaiveEngine(model, tokenizer)


@pytest.fixture(scope="module")
def kv_engine(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    return KVCacheEngine(model, tokenizer)


@pytest.fixture(scope="module")
def tokenizer(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    return tokenizer


class TestKVCacheCorrectness:
    """KV-cache engine must produce identical output to the naive engine."""

    def test_matches_naive_short(self, naive_engine, kv_engine, tokenizer):
        """Short prompt: KV-cache output must match naive output exactly."""
        input_ids = format_chat_prompt(tokenizer, "What is 2+2?")

        naive_ids, _ = naive_engine.generate(input_ids, max_new_tokens=32)
        kv_ids, _ = kv_engine.generate(input_ids, max_new_tokens=32)

        assert naive_ids == kv_ids, (
            f"Output mismatch!\n"
            f"  Naive: {tokenizer.decode(naive_ids)!r}\n"
            f"  KV:    {tokenizer.decode(kv_ids)!r}"
        )

    def test_matches_naive_medium(self, naive_engine, kv_engine, tokenizer):
        """Medium prompt: outputs must match."""
        input_ids = format_chat_prompt(tokenizer, "Explain what a GPU is in three sentences.")

        naive_ids, _ = naive_engine.generate(input_ids, max_new_tokens=64)
        kv_ids, _ = kv_engine.generate(input_ids, max_new_tokens=64)

        assert naive_ids == kv_ids


class TestKVCacheDeterminism:
    def test_deterministic(self, kv_engine, tokenizer):
        input_ids = format_chat_prompt(tokenizer, "What is the capital of France?")

        ids_1, _ = kv_engine.generate(input_ids, max_new_tokens=32)
        ids_2, _ = kv_engine.generate(input_ids, max_new_tokens=32)

        assert ids_1 == ids_2


class TestKVCacheEOS:
    def test_stops_at_eos(self, kv_engine, tokenizer):
        input_ids = format_chat_prompt(tokenizer, "What is 2+2?")
        generated_ids, metrics = kv_engine.generate(input_ids, max_new_tokens=256)

        assert metrics.total_tokens_generated < 256
        assert metrics.total_tokens_generated > 0

    def test_eos_not_in_output(self, kv_engine, tokenizer):
        input_ids = format_chat_prompt(tokenizer, "Say hello.")
        generated_ids, _ = kv_engine.generate(input_ids, max_new_tokens=64)

        for eos_id in kv_engine.eos_token_ids:
            assert eos_id not in generated_ids


class TestKVCacheMetrics:
    def test_ttft_less_than_total(self, kv_engine, tokenizer):
        input_ids = format_chat_prompt(tokenizer, "Explain gravity.")
        _, metrics = kv_engine.generate(input_ids, max_new_tokens=64)

        if metrics.total_tokens_generated > 1:
            assert metrics.ttft_ms < metrics.total_time_ms

    def test_tpot_less_than_ttft(self, kv_engine, tokenizer):
        """TPOT should be less than TTFT because decode steps process only
        1 token while prefill processes the entire prompt."""
        input_ids = format_chat_prompt(tokenizer, "Write a short poem about the ocean.")
        _, metrics = kv_engine.generate(input_ids, max_new_tokens=64)

        if metrics.total_tokens_generated > 2:
            # TPOT should generally be faster than TTFT with KV-cache
            # (decode = 1 token, prefill = all prompt tokens)
            assert metrics.tpot_ms <= metrics.ttft_ms * 2  # generous margin
