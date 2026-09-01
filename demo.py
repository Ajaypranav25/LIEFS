"""
Interactive Demo Script for LIEFS (LLM Inference Engine From Scratch).

Runs an end-to-end demonstration comparing all implemented engines:
1. Naive Engine (Stage 1)
2. KV-Cache Engine (Stage 2)
3. Paged Engine (Stage 4)
4. Continuous Batching Scheduler (Stage 3)
5. INT8 Quantized Engine (Stage 5)

Usage:
    python demo.py
"""

import time
import torch

from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.naive_engine import NaiveEngine
from liefs.kv_cache_engine import KVCacheEngine
from liefs.paged_engine import PagedEngine
from liefs.quantized_engine import create_quantized_engine
from liefs.scheduler import ContinuousBatchScheduler
from liefs.utils import print_metrics


def run_demo():
    print("=" * 80)
    print("LIEFS: LLM Inference Engine From Scratch - Interactive Demo")
    print("=" * 80)

    # 1. Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer()

    test_prompt = "Explain in two sentences why GPUs are fast for neural networks."
    max_tokens = 64

    print(f"Prompt: {test_prompt!r}")
    print(f"Max new tokens: {max_tokens}\n")

    input_ids = format_chat_prompt(tokenizer, test_prompt)

    # ── Demo 1: Naive Engine ───────────────────────────────────────────
    print("-" * 80)
    print("[Stage 1] Naive Engine (Full Recomputation every step, No KV-Cache)")
    print("-" * 80)
    naive_engine = NaiveEngine(model, tokenizer)
    torch.cuda.empty_cache()
    gen_ids, metrics = naive_engine.generate(input_ids, max_new_tokens=max_tokens)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"\nGenerated text:\n{text.strip()}\n")
    print_metrics(metrics, "Naive Engine")

    # ── Demo 2: KV-Cache Engine ─────────────────────────────────────────
    print("-" * 80)
    print("[Stage 2] KV-Cache Engine (Prefill + Decode with Cached Projections)")
    print("-" * 80)
    kv_engine = KVCacheEngine(model, tokenizer)
    torch.cuda.empty_cache()
    gen_ids, metrics = kv_engine.generate(input_ids, max_new_tokens=max_tokens)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"\nGenerated text:\n{text.strip()}\n")
    print_metrics(metrics, "KV-Cache Engine")

    # ── Demo 3: Paged Attention Engine ───────────────────────────────────
    print("-" * 80)
    print("[Stage 4] Paged Attention Engine (16-token BlockAllocator Pool)")
    print("-" * 80)
    paged_engine = PagedEngine(model, tokenizer, block_size=16, max_num_blocks=256)
    torch.cuda.empty_cache()
    gen_ids, metrics = paged_engine.generate(input_ids, max_new_tokens=max_tokens)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"\nGenerated text:\n{text.strip()}\n")
    print_metrics(metrics, "Paged Engine")

    # ── Demo 4: Continuous Batching Scheduler ────────────────────────────
    print("-" * 80)
    print("[Stage 3] Continuous Batching (3 Concurrent Requests)")
    print("-" * 80)
    batch_prompts = [
        "What is 2+2?",
        "Name 3 primary colors.",
        "What is the capital of Japan?",
    ]
    scheduler = ContinuousBatchScheduler(model, tokenizer, max_batch_size=2)
    for p in batch_prompts:
        p_ids = format_chat_prompt(tokenizer, p)
        scheduler.add_request(p_ids, max_new_tokens=32)

    torch.cuda.empty_cache()
    start_time = time.perf_counter()
    completed = scheduler.run()
    total_time = time.perf_counter() - start_time

    for req in completed:
        ans = tokenizer.decode(req.generated_ids, skip_special_tokens=True)
        print(f"  * Request [{req.request_id}] -> {ans.strip()[:60]}... ({req.tokens_generated} tokens)")

    print(f"\n  Batch throughput: {sum(r.tokens_generated for r in completed) / total_time:.2f} tok/s in {total_time:.2f}s\n")

    # ── Demo 5: INT8 Quantized Engine ────────────────────────────────────
    print("-" * 80)
    print("[Stage 5] INT8 Quantized Engine (Per-Channel Symmetric Quantization)")
    print("-" * 80)
    vram_before = torch.cuda.memory_allocated() / (1024 * 1024)
    q_engine = create_quantized_engine(model, tokenizer)
    vram_after = torch.cuda.memory_allocated() / (1024 * 1024)

    print(f"  VRAM before quantization: {vram_before:.1f} MB")
    print(f"  VRAM after quantization:  {vram_after:.1f} MB (Saved {vram_before - vram_after:.1f} MB)")

    torch.cuda.empty_cache()
    gen_ids, metrics = q_engine.generate(input_ids, max_new_tokens=max_tokens)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"\nGenerated text:\n{text.strip()}\n")
    print_metrics(metrics, "INT8 Quantized Engine")

    print("=" * 80)
    print("All LIEFS Inference Engine stages successfully demonstrated!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
