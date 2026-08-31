"""
Stage 3 benchmark: Continuous batching throughput vs static batching.

Compares throughput when processing multiple concurrent requests using:
1. Sequential processing (one at a time)
2. Continuous batching (fill slots as sequences finish)

Usage:
    python -m benchmarks.bench_continuous
"""

import time
import statistics

import torch

from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.scheduler import ContinuousBatchScheduler
from liefs.kv_cache_engine import KVCacheEngine
from liefs.utils import cuda_timer, reset_vram_stats, get_peak_vram_mb
from benchmarks.prompts import BENCHMARK_PROMPTS


def generate_workload(tokenizer, n_requests: int = 8) -> list[tuple[torch.Tensor, int]]:
    """Create a workload of n_requests by cycling through benchmark prompts."""
    workload = []
    for i in range(n_requests):
        name, user_msg, max_tokens = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        input_ids = format_chat_prompt(tokenizer, user_msg)
        workload.append((input_ids, max_tokens))
    return workload


def benchmark_sequential(
    model, tokenizer, workload: list[tuple[torch.Tensor, int]]
) -> dict:
    """Process requests one at a time (static batch_size=1)."""
    engine = KVCacheEngine(model, tokenizer)
    reset_vram_stats()

    total_tokens = 0
    torch.cuda.synchronize()
    start = time.perf_counter()

    for input_ids, max_tokens in workload:
        gen_ids, _ = engine.generate(input_ids, max_new_tokens=max_tokens)
        total_tokens += len(gen_ids)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    return {
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "throughput_tok_s": total_tokens / total_time,
        "peak_vram_mb": get_peak_vram_mb(),
    }


def benchmark_continuous(
    model, tokenizer, workload: list[tuple[torch.Tensor, int]], batch_size: int
) -> dict:
    """Process requests with continuous batching."""
    scheduler = ContinuousBatchScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=batch_size,
    )

    for input_ids, max_tokens in workload:
        scheduler.add_request(input_ids, max_new_tokens=max_tokens)

    reset_vram_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    completed = scheduler.run()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    total_tokens = sum(r.tokens_generated for r in completed)

    return {
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "throughput_tok_s": total_tokens / total_time,
        "peak_vram_mb": get_peak_vram_mb(),
        "num_completed": len(completed),
    }


def main():
    print("LIEFS — Stage 3: Continuous Batching Benchmark")
    print("-" * 50)

    model, tokenizer = load_model_and_tokenizer()

    n_requests = 8
    workload = generate_workload(tokenizer, n_requests)
    print(f"\nWorkload: {n_requests} requests")

    # Sequential baseline
    print("\n>>> Sequential (batch_size=1)...")
    seq_result = benchmark_sequential(model, tokenizer, workload)
    print(f"  Total tokens: {seq_result['total_tokens']}")
    print(f"  Total time: {seq_result['total_time_s']:.2f}s")
    print(f"  Throughput: {seq_result['throughput_tok_s']:.2f} tok/s")
    print(f"  Peak VRAM: {seq_result['peak_vram_mb']:.1f} MB")

    # Continuous batching at various batch sizes
    print("\n" + "=" * 70)
    print(f"{'Batch Size':>12} {'Tokens':>8} {'Time (s)':>10} {'Tok/s':>10} {'Speedup':>10} {'VRAM (MB)':>10}")
    print("-" * 70)
    print(f"{'sequential':>12} {seq_result['total_tokens']:>8} {seq_result['total_time_s']:>10.2f} {seq_result['throughput_tok_s']:>10.2f} {'1.0x':>10} {seq_result['peak_vram_mb']:>10.1f}")

    for batch_size in [2, 4, 8]:
        result = benchmark_continuous(model, tokenizer, workload, batch_size)
        speedup = result["throughput_tok_s"] / seq_result["throughput_tok_s"]
        print(f"{batch_size:>12} {result['total_tokens']:>8} {result['total_time_s']:>10.2f} {result['throughput_tok_s']:>10.2f} {speedup:>9.1f}x {result['peak_vram_mb']:>10.1f}")

    print()


if __name__ == "__main__":
    main()
