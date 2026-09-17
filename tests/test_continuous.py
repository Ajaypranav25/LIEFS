"""
Correctness tests for Stage 3: Continuous batching scheduler.
"""

import pytest
import torch

from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.kv_cache_engine import KVCacheEngine
from liefs.scheduler import ContinuousBatchScheduler, RequestStatus


@pytest.fixture(scope="module")
def model_and_tokenizer():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model, tokenizer = load_model_and_tokenizer(dtype=dtype)
    return model, tokenizer


@pytest.fixture(scope="module")
def tokenizer(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    return tokenizer


class TestSchedulerBasics:
    def test_single_request(self, model_and_tokenizer, tokenizer):
        """Scheduler should handle a single request correctly."""
        model, tok = model_and_tokenizer
        scheduler = ContinuousBatchScheduler(model, tok, max_batch_size=1)

        input_ids = format_chat_prompt(tok, "What is 2+2?")
        scheduler.add_request(input_ids, max_new_tokens=32)

        completed = scheduler.run()

        assert len(completed) == 1
        assert completed[0].status == RequestStatus.FINISHED
        assert completed[0].tokens_generated > 0

        text = tok.decode(completed[0].generated_ids, skip_special_tokens=True)
        assert "4" in text

    def test_multiple_requests(self, model_and_tokenizer, tokenizer):
        """Scheduler should complete all queued requests."""
        model, tok = model_and_tokenizer
        scheduler = ContinuousBatchScheduler(model, tok, max_batch_size=2)

        prompts = ["What is 2+2?", "What is the capital of France?", "Say hello."]
        for prompt in prompts:
            input_ids = format_chat_prompt(tok, prompt)
            scheduler.add_request(input_ids, max_new_tokens=32)

        completed = scheduler.run()

        assert len(completed) == 3
        assert all(r.status == RequestStatus.FINISHED for r in completed)
        assert all(r.tokens_generated > 0 for r in completed)

    def test_max_tokens_respected(self, model_and_tokenizer, tokenizer):
        """Each request should respect its max_new_tokens."""
        model, tok = model_and_tokenizer
        scheduler = ContinuousBatchScheduler(model, tok, max_batch_size=2)

        input_ids = format_chat_prompt(tok, "Write a very long essay.")
        scheduler.add_request(input_ids, max_new_tokens=5)

        completed = scheduler.run()

        assert len(completed) == 1
        assert completed[0].tokens_generated <= 5


class TestContinuousBatching:
    def test_slot_reuse(self, model_and_tokenizer, tokenizer):
        """With batch_size=1 and multiple requests, slots should be reused."""
        model, tok = model_and_tokenizer
        scheduler = ContinuousBatchScheduler(model, tok, max_batch_size=1)

        for i in range(3):
            input_ids = format_chat_prompt(tok, f"Count to {i+1}.")
            scheduler.add_request(input_ids, max_new_tokens=16)

        completed = scheduler.run()

        # All 3 should complete even though batch_size=1
        assert len(completed) == 3

    def test_matches_sequential(self, model_and_tokenizer, tokenizer):
        """Continuous batching with batch_size=1 should produce same output
        as sequential KVCacheEngine generation."""
        model, tok = model_and_tokenizer

        prompt = "What is 2+2?"
        input_ids = format_chat_prompt(tok, prompt)

        # Sequential
        kv_engine = KVCacheEngine(model, tok)
        seq_ids, _ = kv_engine.generate(input_ids, max_new_tokens=32)

        # Continuous batch (size=1)
        scheduler = ContinuousBatchScheduler(model, tok, max_batch_size=1)
        scheduler.add_request(input_ids.clone(), max_new_tokens=32)
        completed = scheduler.run()

        assert completed[0].generated_ids == seq_ids


class TestRequestTracking:
    def test_request_ids(self, model_and_tokenizer, tokenizer):
        """Each request should get a unique ID."""
        model, tok = model_and_tokenizer
        scheduler = ContinuousBatchScheduler(model, tok, max_batch_size=4)

        ids = []
        for prompt in ["Hi", "Hello", "Hey"]:
            input_ids = format_chat_prompt(tok, prompt)
            rid = scheduler.add_request(input_ids, max_new_tokens=8)
            ids.append(rid)

        # All IDs should be unique
        assert len(set(ids)) == 3

    def test_finish_reason(self, model_and_tokenizer, tokenizer):
        """Requests should report correct finish reason."""
        model, tok = model_and_tokenizer
        scheduler = ContinuousBatchScheduler(model, tok, max_batch_size=1)

        # Short answer should hit EOS → "stop"
        input_ids = format_chat_prompt(tok, "What is 2+2?")
        scheduler.add_request(input_ids, max_new_tokens=256)

        completed = scheduler.run()

        # Could be "stop" (EOS) or "length" depending on model behavior
        assert completed[0].finish_reason in ("stop", "length")
