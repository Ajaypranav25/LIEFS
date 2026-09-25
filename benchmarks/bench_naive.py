"""
Stage 1 benchmark: Naive baseline (full recomputation, no KV-cache).

Measures TTFT, TPOT, tokens/sec, and peak VRAM across the standard prompt
set. Also optionally compares against HuggingFace model.generate() to show
what we're leaving on the table before our own optimizations kick in.

Usage:
    python -m benchmarks.bench_naive
"""

import statistics

import torch

from benchmarks.prompts import BENCHMARK_PROMPTS
from liefs.model_loader import format_chat_prompt, load_model_and_tokenizer
from liefs.naive_engine import NaiveEngine
from liefs.utils import (
    GenerationMetrics,
    cuda_timer,
    get_peak_vram_mb,
    reset_vram_stats,
)

NUM_RUNS = 3  # Runs per prompt for mean ± std


def benchmark_naive_engine(engine: NaiveEngine) -> dict[str, list[GenerationMetrics]]:
    """Run the naive engine benchmark across all prompts.

    Protocol:
        1. One warm-up run per prompt (discarded) — populates CUDA caches,
           triggers JIT kernel compilation for these shapes.
        2. NUM_RUNS timed runs per prompt.

    Returns:
        Dict mapping prompt name → list of GenerationMetrics.
    """
    results: dict[str, list[GenerationMetrics]] = {}

    for name, user_msg, max_tokens in BENCHMARK_PROMPTS:
        print(f"\n--- Prompt: {name!r} (max_new_tokens={max_tokens}) ---")
        input_ids = format_chat_prompt(engine.tokenizer, user_msg)
        print(f"  Prompt length: {input_ids.shape[1]} tokens")

        # Warm-up run (not measured)
        print("  Warm-up run...")
        engine.generate(input_ids, max_new_tokens=max_tokens)

        # Timed runs
        run_metrics: list[GenerationMetrics] = []
        for run_idx in range(NUM_RUNS):
            # Clear CUDA cache between runs for clean VRAM measurement
            torch.cuda.empty_cache()

            text_ids, metrics = engine.generate(input_ids, max_new_tokens=max_tokens)
            run_metrics.append(metrics)

            # Print generated text on first run only (for sanity checking)
            if run_idx == 0:
                text = engine.tokenizer.decode(text_ids, skip_special_tokens=True)
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"  Output preview: {preview!r}")

        results[name] = run_metrics

    return results


def benchmark_hf_generate(model, tokenizer) -> dict[str, list[GenerationMetrics]]:
    """Run HuggingFace model.generate() for comparison.

    This uses HF's optimized generation pipeline (with KV-cache enabled
    by default) to show the gap between our naive approach and a standard
    library implementation.
    """
    results: dict[str, list[GenerationMetrics]] = {}

    for name, user_msg, max_tokens in BENCHMARK_PROMPTS:
        print(f"\n--- [HF generate] Prompt: {name!r} (max_new_tokens={max_tokens}) ---")
        input_ids = format_chat_prompt(tokenizer, user_msg)
        prompt_len = input_ids.shape[1]

        # Warm-up
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)

        run_metrics: list[GenerationMetrics] = []
        for run_idx in range(NUM_RUNS):
            torch.cuda.empty_cache()
            reset_vram_stats()

            with cuda_timer() as elapsed, torch.no_grad():
                output_ids = model.generate(
                    input_ids, max_new_tokens=max_tokens, do_sample=False
                )

            total_time = elapsed()
            num_generated = output_ids.shape[1] - prompt_len

            # HF generate doesn't give us per-token timing, so we can only
            # measure aggregate metrics
            metrics = GenerationMetrics(
                ttft_ms=0.0,  # Can't measure with HF generate
                tpot_ms=(total_time / num_generated) if num_generated > 0 else 0.0,
                tokens_per_sec=(num_generated / total_time * 1000)
                if total_time > 0
                else 0.0,
                total_tokens_generated=num_generated,
                total_time_ms=total_time,
                peak_vram_mb=get_peak_vram_mb(),
            )
            run_metrics.append(metrics)

        results[name] = run_metrics

    return results


def print_summary(
    naive_results: dict[str, list[GenerationMetrics]],
    hf_results: dict[str, list[GenerationMetrics]] | None = None,
):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("STAGE 1 BENCHMARK RESULTS — NAIVE BASELINE")
    print("=" * 80)

    for name, metrics_list in naive_results.items():
        # Compute mean ± std for each metric
        ttfts = [m.ttft_ms for m in metrics_list]
        tpots = [m.tpot_ms for m in metrics_list]
        tps = [m.tokens_per_sec for m in metrics_list]
        vrams = [m.peak_vram_mb for m in metrics_list]
        n_toks = [m.total_tokens_generated for m in metrics_list]

        print(f"\n  Prompt: {name!r}")
        print(f"    Tokens generated : {n_toks[0]}")
        print(
            f"    TTFT             : {statistics.mean(ttfts):>8.2f} ± {statistics.stdev(ttfts):>6.2f} ms"
            if len(ttfts) > 1
            else f"    TTFT             : {ttfts[0]:>8.2f} ms"
        )
        print(
            f"    TPOT             : {statistics.mean(tpots):>8.2f} ± {statistics.stdev(tpots):>6.2f} ms"
            if len(tpots) > 1
            else f"    TPOT             : {tpots[0]:>8.2f} ms"
        )
        print(
            f"    Throughput       : {statistics.mean(tps):>8.2f} ± {statistics.stdev(tps):>6.2f} tok/s"
            if len(tps) > 1
            else f"    Throughput       : {tps[0]:>8.2f} tok/s"
        )
        print(f"    Peak VRAM        : {statistics.mean(vrams):>8.2f} MB")

        # HF comparison
        if hf_results and name in hf_results:
            hf_list = hf_results[name]
            hf_tps = [m.tokens_per_sec for m in hf_list]
            hf_vrams = [m.peak_vram_mb for m in hf_list]
            hf_mean_tps = statistics.mean(hf_tps)
            naive_mean_tps = statistics.mean(tps)
            speedup = (
                hf_mean_tps / naive_mean_tps if naive_mean_tps > 0 else float("inf")
            )

            print("    --- HF generate ---")
            print(
                f"    HF Throughput    : {hf_mean_tps:>8.2f} tok/s  ({speedup:.1f}x faster)"
            )
            print(f"    HF Peak VRAM     : {statistics.mean(hf_vrams):>8.2f} MB")


def main():
    print("LIEFS — Stage 1: Naive Baseline Benchmark")
    print("-" * 50)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Create naive engine
    engine = NaiveEngine(model, tokenizer)

    # Run naive benchmark
    print("\n>>> Running naive engine benchmark...")
    naive_results = benchmark_naive_engine(engine)

    # Run HF comparison
    print("\n>>> Running HuggingFace generate() comparison...")
    hf_results = benchmark_hf_generate(model, tokenizer)

    # Print summary
    print_summary(naive_results, hf_results)


if __name__ == "__main__":
    main()
