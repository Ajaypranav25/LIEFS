"""
Stage 2 benchmark: KV-Cache engine vs Naive baseline.

Compares the KV-cache engine against the Stage 1 naive engine to measure
the speedup from eliminating redundant recomputation.

Usage:
    python -m benchmarks.bench_kv_cache
"""

import statistics

import torch

from benchmarks.prompts import BENCHMARK_PROMPTS
from liefs.kv_cache_engine import KVCacheEngine
from liefs.model_loader import format_chat_prompt, load_model_and_tokenizer
from liefs.naive_engine import NaiveEngine
from liefs.utils import GenerationMetrics

NUM_RUNS = 3


def benchmark_engine(engine, engine_name: str) -> dict[str, list[GenerationMetrics]]:
    """Run benchmark for any engine that has a .generate() method."""
    results: dict[str, list[GenerationMetrics]] = {}

    for name, user_msg, max_tokens in BENCHMARK_PROMPTS:
        print(
            f"\n--- [{engine_name}] Prompt: {name!r} (max_new_tokens={max_tokens}) ---"
        )
        input_ids = format_chat_prompt(engine.tokenizer, user_msg)
        print(f"  Prompt length: {input_ids.shape[1]} tokens")

        # Warm-up
        engine.generate(input_ids, max_new_tokens=max_tokens)

        run_metrics: list[GenerationMetrics] = []
        for run_idx in range(NUM_RUNS):
            torch.cuda.empty_cache()
            gen_ids, metrics = engine.generate(input_ids, max_new_tokens=max_tokens)

            if run_idx == 0:
                text = engine.tokenizer.decode(gen_ids, skip_special_tokens=True)
                preview = text[:150] + "..." if len(text) > 150 else text
                print(f"  Output: {preview!r}")

            run_metrics.append(metrics)

        results[name] = run_metrics

    return results


def print_comparison(
    naive_results: dict[str, list[GenerationMetrics]],
    kv_results: dict[str, list[GenerationMetrics]],
):
    """Print side-by-side comparison."""
    print("\n" + "=" * 90)
    print("STAGE 2 BENCHMARK — KV-CACHE vs NAIVE BASELINE")
    print("=" * 90)

    for name, kv_list in kv_results.items():
        naive_list = naive_results.get(name, [])

        kv_tps = statistics.mean([m.tokens_per_sec for m in kv_list])
        naive_tps = statistics.mean([m.tokens_per_sec for m in naive_list])
        speedup = kv_tps / naive_tps if naive_tps > 0 else float("inf")

        kv_ttft = statistics.mean([m.ttft_ms for m in kv_list])
        naive_ttft = statistics.mean([m.ttft_ms for m in naive_list])

        kv_tpot = statistics.mean([m.tpot_ms for m in kv_list])
        naive_tpot = statistics.mean([m.tpot_ms for m in naive_list])

        kv_vram = statistics.mean([m.peak_vram_mb for m in kv_list])
        naive_vram = statistics.mean([m.peak_vram_mb for m in naive_list])

        n_toks = kv_list[0].total_tokens_generated

        print(f"\n  Prompt: {name!r} ({n_toks} tokens)")
        print(f"    {'Metric':<20} {'Naive':>12} {'KV-Cache':>12} {'Speedup':>10}")
        print(f"    {'-' * 54}")
        print(
            f"    {'Throughput (tok/s)':<20} {naive_tps:>12.2f} {kv_tps:>12.2f} {speedup:>9.1f}x"
        )
        print(
            f"    {'TTFT (ms)':<20} {naive_ttft:>12.2f} {kv_ttft:>12.2f} {naive_ttft / kv_ttft if kv_ttft > 0 else 0:>9.1f}x"
        )
        print(
            f"    {'TPOT (ms)':<20} {naive_tpot:>12.2f} {kv_tpot:>12.2f} {naive_tpot / kv_tpot if kv_tpot > 0 else 0:>9.1f}x"
        )
        print(f"    {'Peak VRAM (MB)':<20} {naive_vram:>12.2f} {kv_vram:>12.2f}")


def main():
    print("LIEFS — Stage 2: KV-Cache Benchmark")
    print("-" * 50)

    model, tokenizer = load_model_and_tokenizer()

    naive_engine = NaiveEngine(model, tokenizer)
    kv_engine = KVCacheEngine(model, tokenizer)

    print("\n>>> Running naive engine benchmark...")
    naive_results = benchmark_engine(naive_engine, "Naive")

    print("\n>>> Running KV-cache engine benchmark...")
    kv_results = benchmark_engine(kv_engine, "KV-Cache")

    print_comparison(naive_results, kv_results)


if __name__ == "__main__":
    main()
