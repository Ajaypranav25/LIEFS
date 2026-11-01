"""
Universal Computer Hardware & Model Benchmarking CLI.

Benchmarks your computer hardware across any Hugging Face model or local checkpoint.
Evaluates single-stream throughput, memory bandwidth (GB/s), 5-engine speedup matrix,
concurrency batch scaling, and computes a standardized Computer Performance Index Score.

Usage:
    python -m benchmarks.bench_computer
    python -m benchmarks.bench_computer --model meta-llama/Llama-3.2-1B-Instruct
    python -m benchmarks.bench_computer --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --export report.json
"""

import argparse
import json
import time

import torch

from liefs.hardware_profiler import (
    calculate_computer_score,
    calculate_effective_memory_bandwidth,
    get_hardware_profile,
)
from liefs.kv_cache_engine import KVCacheEngine
from liefs.model_loader import (
    DEFAULT_MODEL_NAME,
    format_chat_prompt,
    get_model_metadata,
    load_model_and_tokenizer,
)
from liefs.naive_engine import NaiveEngine
from liefs.paged_engine import PagedEngine
from liefs.scheduler import ContinuousBatchScheduler
from liefs.utils import reset_vram_stats


def print_banner():
    banner = r"""
====================================================================================
   __    _______________________   ____                               _            
  / /   /  _/ ____/ ____/ ___/  / __ )___  ____  _____/ /_  ____ ___  (_)____  _____
 / /    / // __/ / __/  \__ \  / __  / _ \/ __ \/ ___/ __ \/ __ `__ \/ / __ \/ ___/
/ /____/ // /___/ /___ ___/ / / /_/ /  __/ / / / /__/ / / / / / / / / / /_/ / /    
/_____/___/_____/_____//____/ /_____/\___/_/ /_/\___/_/ /_/_/ /_/ /_/_/\__,_/_/     
           Universal Computer Hardware & Model Benchmarking Platform
====================================================================================
"""
    print(banner)


def run_benchmark(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    precision: str = "float16",
    hf_token: str | None = None,
    quick: bool = False,
    export_path: str | None = None,
):
    print_banner()

    # 1. Profile Host Hardware
    print("[1/5] Profiling host computer hardware...")
    hw = get_hardware_profile()
    print(f"  OS:        {hw.os_name} {hw.os_release} ({hw.architecture})")
    print(
        f"  CPU:       {hw.cpu_model} ({hw.cpu_physical_cores} Physical / {hw.cpu_logical_cores} Logical Cores)"
    )
    print(
        f"  RAM:       {hw.ram_total_gb} GB Total ({hw.ram_available_gb} GB Available)"
    )
    print(
        f"  GPU:       {hw.gpu_name} (Compute Capability: {hw.gpu_compute_capability or 'N/A'})"
    )
    print(
        f"  VRAM:      {hw.vram_total_mb:.1f} MB Total ({hw.vram_free_mb:.1f} MB Free)"
    )
    print(f"  PyTorch:   {hw.pytorch_version} (CUDA: {hw.cuda_version or 'N/A'})\n")

    # 2. Load Model
    print(f"[2/5] Loading model {model_name!r} on {device} ({precision})...")
    t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        dtype=precision,
        device=device,
        hf_token=hf_token,
    )
    load_time_sec = round(time.perf_counter() - t0, 2)
    meta = get_model_metadata(model, tokenizer, model_name)
    print(f"  Load Time: {load_time_sec}s")
    print(f"  Parameters: {meta.parameter_count:,} ({meta.parameter_count_m}M)")
    print(
        f"  Architecture: {meta.num_layers} Layers | {meta.num_attention_heads} Q Heads | {meta.num_kv_heads} KV Heads | Head Dim {meta.head_dim}"
    )
    print(
        f"  Memory Footprint: {meta.memory_footprint_mb:.1f} MB (FP16 Estimate: {meta.fp16_vram_estimate_mb:.1f} MB)\n"
    )

    # 3. Instantiate Engines
    print("[3/5] Initializing inference engine architectures...")
    kv_engine = KVCacheEngine(model, tokenizer)
    naive_engine = NaiveEngine(model, tokenizer)
    try:
        paged_engine = PagedEngine(model, tokenizer, block_size=16, max_num_blocks=256)
    except Exception as e:  # noqa: BLE001
        print(f"  Paged Engine initialization notice: {e}")
        paged_engine = None

    # Benchmark test parameters
    max_tokens = 32 if quick else 128
    test_prompt = (
        "Explain how transformer self-attention works step by step in plain terms."
    )
    input_ids = format_chat_prompt(tokenizer, test_prompt, device=meta.device_str)
    prompt_tokens = input_ids.shape[1]

    print(
        f"\n[4/5] Running hardware & engine benchmark ({max_tokens} tokens per test)..."
    )
    print(f"  Prompt length: {prompt_tokens} tokens")

    # Warm-up pass
    print("  Warming up GPU execution pipeline...")
    kv_engine.generate(input_ids, max_new_tokens=16)

    results = []

    # 1. KV-Cache Engine (Stage 2)
    print("  Testing Stage 2: KV-Cache Engine...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    reset_vram_stats()
    _gen_ids, m_kv = kv_engine.generate(input_ids, max_new_tokens=max_tokens)
    bw_kv = calculate_effective_memory_bandwidth(
        meta.memory_footprint_mb, m_kv.tokens_per_sec
    )
    results.append(
        {
            "engine": "kv_cache",
            "name": "KV-Cache (Stage 2)",
            "throughput_tok_s": round(m_kv.tokens_per_sec, 2),
            "ttft_ms": round(m_kv.ttft_ms, 2),
            "tpot_ms": round(m_kv.tpot_ms, 2),
            "total_time_ms": round(m_kv.total_time_ms, 2),
            "peak_vram_mb": round(m_kv.peak_vram_mb, 2),
            "memory_bandwidth_gbs": bw_kv,
            "speedup": 1.0,
        }
    )

    # 2. Naive Baseline (Stage 1)
    print("  Testing Stage 1: Naive Full-Recomputation Baseline...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    reset_vram_stats()
    _gen_ids, m_naive = naive_engine.generate(input_ids, max_new_tokens=max_tokens)
    bw_naive = calculate_effective_memory_bandwidth(
        meta.memory_footprint_mb, m_naive.tokens_per_sec
    )
    speedup_kv = round(m_naive.total_time_ms / max(1.0, m_kv.total_time_ms), 2)
    results[0]["speedup"] = speedup_kv
    results.append(
        {
            "engine": "naive",
            "name": "Naive Recomputation (Stage 1)",
            "throughput_tok_s": round(m_naive.tokens_per_sec, 2),
            "ttft_ms": round(m_naive.ttft_ms, 2),
            "tpot_ms": round(m_naive.tpot_ms, 2),
            "total_time_ms": round(m_naive.total_time_ms, 2),
            "peak_vram_mb": round(m_naive.peak_vram_mb, 2),
            "memory_bandwidth_gbs": bw_naive,
            "speedup": 1.0,
        }
    )

    # 3. Paged Attention (Stage 4)
    if paged_engine:
        print("  Testing Stage 4: Paged Attention Pool...")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        reset_vram_stats()
        _gen_ids, m_paged = paged_engine.generate(input_ids, max_new_tokens=max_tokens)
        bw_paged = calculate_effective_memory_bandwidth(
            meta.memory_footprint_mb, m_paged.tokens_per_sec
        )
        results.append(
            {
                "engine": "paged",
                "name": "Paged Attention (Stage 4)",
                "throughput_tok_s": round(m_paged.tokens_per_sec, 2),
                "ttft_ms": round(m_paged.ttft_ms, 2),
                "tpot_ms": round(m_paged.tpot_ms, 2),
                "total_time_ms": round(m_paged.total_time_ms, 2),
                "peak_vram_mb": round(m_paged.peak_vram_mb, 2),
                "memory_bandwidth_gbs": bw_paged,
                "speedup": round(
                    m_naive.total_time_ms / max(1.0, m_paged.total_time_ms), 2
                ),
            }
        )

    # Concurrency Batch Scaling (Stage 3)
    batch_scaling = []
    if not quick:
        print(
            "\n  Testing Stage 3 Concurrency & Batch Scaling (Batch Sizes 1, 2, 4, 8)..."
        )
        for b_size in [1, 2, 4, 8]:
            scheduler = ContinuousBatchScheduler(
                model, tokenizer, max_batch_size=b_size
            )
            for req_i in range(b_size):
                scheduler.add_request(f"What is {req_i} + {req_i}?", max_new_tokens=32)

            t_start = time.perf_counter()
            req_metrics = scheduler.run_until_complete()
            t_batch_elapsed = (time.perf_counter() - t_start) * 1000.0

            total_toks = sum(m.total_tokens_generated for m in req_metrics)
            agg_throughput = (
                round((total_toks / t_batch_elapsed * 1000.0), 2)
                if t_batch_elapsed > 0
                else 0.0
            )
            batch_scaling.append(
                {
                    "batch_size": b_size,
                    "total_tokens": total_toks,
                    "wall_time_ms": round(t_batch_elapsed, 1),
                    "aggregate_throughput_tok_s": agg_throughput,
                }
            )
            print(
                f"    Batch Size {b_size:>2}: {agg_throughput:>6.1f} tok/s across {total_toks} tokens"
            )

    # 4. Computer Score Calculation
    print("\n[5/5] Computing Computer Performance Index Score...")
    score_data = calculate_computer_score(
        tokens_per_sec=m_kv.tokens_per_sec,
        ttft_ms=m_kv.ttft_ms,
        model_param_count_m=meta.parameter_count_m,
        memory_bandwidth_gbs=bw_kv,
        is_gpu=(meta.device_str == "cuda"),
    )

    # Display Results Table
    print("\n" + "=" * 102)
    print(f"  BENCHMARK RESULTS: {model_name} on {hw.cpu_model} / {hw.gpu_name}")
    print("=" * 102)
    print(
        f"| {'Engine / Optimization':<30} | {'Throughput':<12} | {'TTFT':<10} | {'TPOT':<10} | {'VRAM':<10} | {'Bandwidth':<10} | {'Speedup':<8} |"
    )
    print(
        "|"
        + "-" * 32
        + "|"
        + "-" * 14
        + "|"
        + "-" * 12
        + "|"
        + "-" * 12
        + "|"
        + "-" * 12
        + "|"
        + "-" * 12
        + "|"
        + "-" * 10
        + "|"
    )
    for r in results:
        print(
            f"| {r['name']:<30} | {r['throughput_tok_s']:>8.1f} tok/s | {r['ttft_ms']:>7.1f} ms | {r['tpot_ms']:>7.1f} ms | {r['peak_vram_mb']:>7.1f} MB | {r['memory_bandwidth_gbs']:>6.1f} GB/s | {r['speedup']:>6.2f}x |"
        )
    print("=" * 102)

    print("\n" + "#" * 70)
    print(
        f"  LIEFS COMPUTER PERFORMANCE SCORE:  {score_data['score']} POINTS  [{score_data['badge']}]"
    )
    print(f"  Hardware Tier:                     {score_data['tier']}")
    print(f"  Effective Memory Bandwidth:        {bw_kv} GB/s")
    print(f"  KV-Cache Elimination Speedup:      {speedup_kv}x faster than Naive")
    print("#" * 70 + "\n")

    report_payload = {
        "timestamp": int(time.time()),
        "model": meta.to_dict(),
        "hardware": hw.to_dict(),
        "score": score_data,
        "results": results,
        "batch_scaling": batch_scaling,
    }

    if export_path:
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"Benchmark report exported successfully to: {export_path}")

    return report_payload


def main():
    parser = argparse.ArgumentParser(
        description="LIEFS Universal Computer & Model Benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face repo ID or local model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Weight precision",
    )
    parser.add_argument(
        "--token", type=str, default=None, help="Hugging Face API token"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run short 32-token benchmark"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Output JSON path to export benchmark results",
    )
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        device=args.device,
        precision=args.precision,
        hf_token=args.token,
        quick=args.quick,
        export_path=args.export,
    )


if __name__ == "__main__":
    main()
