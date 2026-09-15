"""
Correctness tests for Stage 4: Paged attention.
"""

import pytest
import torch

from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.kv_cache_engine import KVCacheEngine
from liefs.paged_engine import PagedEngine
from liefs.paged_attention import BlockAllocator, PagedKVCache


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer()
    return model, tokenizer


@pytest.fixture(scope="module")
def tokenizer(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    return tokenizer


@pytest.fixture(scope="module")
def kv_engine(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    return KVCacheEngine(model, tokenizer)


@pytest.fixture(scope="module")
def paged_engine(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    engine = PagedEngine(model, tokenizer, block_size=16, max_num_blocks=256)

    if not torch.cuda.is_available():
        from liefs.paged_attention import BlockAllocator
        engine.allocator = BlockAllocator(
            num_blocks=256,
            block_size=16,
            num_layers=engine.num_layers,
            num_kv_heads=engine.num_kv_heads,
            head_dim=engine.head_dim,
            device="cpu"
        )

    return engine


class TestPagedCorrectness:
    """Paged engine output must match contiguous KV-cache engine."""

    def test_matches_kv_cache_short(self, kv_engine, paged_engine, tokenizer):
        input_ids = format_chat_prompt(tokenizer, "What is 2+2?")

        kv_ids, _ = kv_engine.generate(input_ids, max_new_tokens=32)
        paged_ids, _ = paged_engine.generate(input_ids, max_new_tokens=32)

        assert kv_ids == paged_ids, (
            f"Output mismatch!\n"
            f"  KV:    {tokenizer.decode(kv_ids)!r}\n"
            f"  Paged: {tokenizer.decode(paged_ids)!r}"
        )

    def test_matches_kv_cache_medium(self, kv_engine, paged_engine, tokenizer):
        input_ids = format_chat_prompt(tokenizer, "Explain what a GPU is.")

        kv_ids, _ = kv_engine.generate(input_ids, max_new_tokens=64)
        paged_ids, _ = paged_engine.generate(input_ids, max_new_tokens=64)

        assert kv_ids == paged_ids


class TestBlockAllocator:
    def test_allocate_and_free(self):
        allocator = BlockAllocator(
            num_blocks=4, block_size=8,
            num_layers=2, num_kv_heads=2, head_dim=64,
            device="cpu",
        )
        assert allocator.num_free_blocks == 4

        b1 = allocator.allocate()
        assert allocator.num_free_blocks == 3

        b2 = allocator.allocate()
        assert allocator.num_free_blocks == 2

        allocator.free(b1)
        assert allocator.num_free_blocks == 3

    def test_out_of_blocks(self):
        allocator = BlockAllocator(
            num_blocks=2, block_size=4,
            num_layers=1, num_kv_heads=1, head_dim=32,
            device="cpu",
        )
        allocator.allocate()
        allocator.allocate()
        assert allocator.allocate() is None  # No more blocks


class TestPagedKVCache:
    def test_blocks_freed_after_generation(self, paged_engine, tokenizer):
        """All blocks should be returned to the pool after generation."""
        initial_free = paged_engine.allocator.num_free_blocks

        input_ids = format_chat_prompt(tokenizer, "Hello")
        paged_engine.generate(input_ids, max_new_tokens=16)

        assert paged_engine.allocator.num_free_blocks == initial_free

    def test_deterministic(self, paged_engine, tokenizer):
        input_ids = format_chat_prompt(tokenizer, "What is the speed of light?")

        ids_1, _ = paged_engine.generate(input_ids, max_new_tokens=32)
        ids_2, _ = paged_engine.generate(input_ids, max_new_tokens=32)

        assert ids_1 == ids_2
