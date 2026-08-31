"""
Stage 5 benchmark: INT8 Quantized model vs FP16 baseline.

Measures memory footprint reduction, throughput, and quality retention.

Usage:
    python -m benchmarks.bench_quantized
"""

import statistics
import torch

from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.kv_cache_engine import KVCacheEngine
from liefs.quantization import quantize_model
from liefs.utils import GenerationMetrics
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
                preview = text[:150] + "..." if len(text) > 150 else text
                print(f"  Output: {preview!r}")

            run_metrics.append(metrics)
        results[name] = run_metrics
    return results


def main():
    print("LIEFS — Stage 5: INT8 Quantization Benchmark")
    print("-" * 50)

    # 1. Load FP16 model
    print("Loading FP16 model...")
    model, tokenizer = load_model_and_tokenizer()
    torch.cuda.synchronize()
    vram_fp16 = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"  FP16 Model VRAM: {vram_fp16:.1f} MB")

    fp16_engine = KVCacheEngine(model, tokenizer)

    print("\n>>> Running FP16 KV-Cache benchmark...")
    fp16_results = benchmark_engine(fp16_engine, "FP16")

    # 2. Quantize model in-place to INT8
    print("\n>>> Quantizing model to INT8 (per-channel symmetric)...")
    quantize_model(model)
    torch.cuda.synchronize()
    vram_int8 = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"  INT8 Model VRAM: {vram_int8:.1f} MB")
    print(f"  Weight Memory Saved: {vram_fp16 - vram_int8:.1f} MB ({(vram_fp16 - vram_int8) / vram_fp16 * 100:.1f}% reduction)")

    int8_engine = KVCacheEngine(model, tokenizer)

    print("\n>>> Running INT8 Quantized benchmark...")
    int8_results = benchmark_engine(int8_engine, "INT8")

    # Summary
    print("\n" + "=" * 90)
    print("STAGE 5 BENCHMARK — FP16 vs INT8 QUANTIZATION")
    print("=" * 90)

    for name in int8_results:
        int8_list = int8_results[name]
        fp16_list = fp16_results[name]

        fp16_tps = statistics.mean([m.tokens_per_sec for m in fp16_list])
        int8_tps = statistics.mean([m.tokens_per_sec for m in int8_list])

        fp16_ttft = statistics.mean([m.ttft_ms for m in fp16_list])
        int8_ttft = statistics.mean([m.ttft_ms for m in int8_list])

        fp16_tpot = statistics.mean([m.tpot_ms for m in fp16_list])
        int8_tpot = statistics.mean([m.tpot_ms for m in int8_list])

        fp16_vram = statistics.mean([m.peak_vram_mb for m in fp16_list])
        int8_vram = statistics.mean([m.peak_vram_mb for m in int8_list])

        print(f"\n  Prompt: {name!r}")
        print(f"    {'Metric':<22} {'FP16':>12} {'INT8':>12} {'Change':>12}")
        print(f"    {'-'*60}")
        print(f"    {'Throughput (tok/s)':<22} {fp16_tps:>12.2f} {int8_tps:>12.2f} {int8_tps / fp16_tps if fp16_tps > 0 else 0:>11.2f}x")
        print(f"    {'TTFT (ms)':<22} {fp16_ttft:>12.2f} {int8_ttft:>12.2f} {int8_ttft / fp16_ttft if fp16_ttft > 0 else 0:>11.2f}x")
        print(f"    {'TPOT (ms)':<22} {fp16_tpot:>12.2f} {int8_tpot:>12.2f} {int8_tpot / fp16_tpot if fp16_tpot > 0 else 0:>11.2f}x")
        print(f"    {'Peak VRAM (MB)':<22} {fp16_vram:>12.2f} {int8_vram:>12.2f} {(int8_vram - fp16_vram) / fp16_vram * 100:>+11.1f}%")


if __name__ == "__main__":
    main()
