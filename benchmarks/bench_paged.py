"""
Stage 4 benchmark: Paged attention vs contiguous KV-cache.

Measures memory utilization and fragmentation characteristics.

Usage:
    python -m benchmarks.bench_paged
"""

import statistics
import torch

from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.kv_cache_engine import KVCacheEngine
from liefs.paged_engine import PagedEngine
from liefs.utils import GenerationMetrics, reset_vram_stats, get_peak_vram_mb
from benchmarks.prompts import BENCHMARK_PROMPTS


NUM_RUNS = 3


def benchmark_engine(engine, engine_name: str) -> dict[str, list[GenerationMetrics]]:
    results = {}
    for name, user_msg, max_tokens in BENCHMARK_PROMPTS:
        print(f"\n--- [{engine_name}] Prompt: {name!r} (max_new_tokens={max_tokens}) ---")
        input_ids = format_chat_prompt(engine.tokenizer, user_msg)

        # Warm-up
        engine.generate(input_ids, max_new_tokens=max_tokens)

        run_metrics = []
        for run_idx in range(NUM_RUNS):
            torch.cuda.empty_cache()
            gen_ids, metrics = engine.generate(input_ids, max_new_tokens=max_tokens)
            if run_idx == 0:
                text = engine.tokenizer.decode(gen_ids, skip_special_tokens=True)
                print(f"  Output: {text[:120]!r}...")
            run_metrics.append(metrics)
        results[name] = run_metrics
    return results


def main():
    print("LIEFS — Stage 4: Paged Attention Benchmark")
    print("-" * 50)

    model, tokenizer = load_model_and_tokenizer()

    kv_engine = KVCacheEngine(model, tokenizer)
    paged_engine = PagedEngine(model, tokenizer, block_size=16, max_num_blocks=256)

    print(f"\nBlock allocator: {paged_engine.allocator.num_blocks} blocks x "
          f"{paged_engine.block_size} tokens = "
          f"{paged_engine.allocator.num_blocks * paged_engine.block_size} max tokens")
    print(f"Block pool memory: {paged_engine.allocator.total_memory_mb:.1f} MB")

    print("\n>>> KV-Cache engine benchmark...")
    kv_results = benchmark_engine(kv_engine, "KV-Cache")

    print("\n>>> Paged engine benchmark...")
    paged_results = benchmark_engine(paged_engine, "Paged")

    # Comparison
    print("\n" + "=" * 80)
    print("STAGE 4 BENCHMARK — PAGED vs CONTIGUOUS KV-CACHE")
    print("=" * 80)

    for name in kv_results:
        kv_list = kv_results[name]
        paged_list = paged_results[name]

        kv_tps = statistics.mean([m.tokens_per_sec for m in kv_list])
        paged_tps = statistics.mean([m.tokens_per_sec for m in paged_list])

        kv_vram = statistics.mean([m.peak_vram_mb for m in kv_list])
        paged_vram = statistics.mean([m.peak_vram_mb for m in paged_list])

        kv_tpot = statistics.mean([m.tpot_ms for m in kv_list])
        paged_tpot = statistics.mean([m.tpot_ms for m in paged_list])

        print(f"\n  Prompt: {name!r}")
        print(f"    {'Metric':<20} {'Contiguous':>12} {'Paged':>12}")
        print(f"    {'-'*44}")
        print(f"    {'Throughput (tok/s)':<20} {kv_tps:>12.2f} {paged_tps:>12.2f}")
        print(f"    {'TPOT (ms)':<20} {kv_tpot:>12.2f} {paged_tpot:>12.2f}")
        print(f"    {'Peak VRAM (MB)':<20} {kv_vram:>12.2f} {paged_vram:>12.2f}")

    # Memory utilization report
    print(f"\n  Block allocator utilization after benchmark:")
    print(f"    Free blocks: {paged_engine.allocator.num_free_blocks}/{paged_engine.allocator.num_blocks}")
    print(f"    Utilization: {paged_engine.allocator.utilization:.1%}")


if __name__ == "__main__":
    main()
