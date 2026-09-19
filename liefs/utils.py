"""
Timing and logging utilities for LIEFS.

Provides GPU and CPU synchronized timers, memory peak tracking,
and multi-architecture EOS token resolution.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class GenerationMetrics:
    """Metrics collected from a single generation run.

    Attributes:
        ttft_ms: Time-to-first-token in milliseconds.
        tpot_ms: Time-per-output-token in milliseconds (average decode latency).
        tokens_per_sec: Total output tokens / total wall time.
        total_tokens_generated: Number of tokens generated.
        total_time_ms: Total wall time in milliseconds.
        peak_vram_mb: Peak GPU VRAM allocated during generation, in MB.
    """
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    tokens_per_sec: float = 0.0
    total_tokens_generated: int = 0
    total_time_ms: float = 0.0
    peak_vram_mb: float = 0.0


@contextmanager
def cuda_timer():
    """Context manager that yields a callable returning elapsed time in ms.

    Uses torch.cuda.synchronize() when CUDA is available to ensure genuine GPU execution timing.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    result = [0.0]

    def get_elapsed() -> float:
        return result[0]

    yield get_elapsed

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result[0] = (time.perf_counter() - start) * 1000.0  # Convert to ms


def get_peak_vram_mb() -> float:
    """Return peak VRAM usage in MB since last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_vram_stats():
    """Reset peak VRAM tracking. Call before each benchmark run."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_eos_token_ids(tokenizer) -> set[int]:
    """Extract all relevant EOS token IDs for generation termination across model families.

    Handles Qwen (<|im_end|>), Llama-3 (<|eot_id|>), Gemma (<end_of_turn>),
    GPT-2/NeoX (<|endoftext|>), and standard tokenizer.eos_token_id.
    """
    eos_ids: set[int] = set()

    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.update(tokenizer.eos_token_id)
        else:
            eos_ids.add(tokenizer.eos_token_id)

    # Common chat stop tokens across popular models
    common_stop_tokens = [
        "<|im_end|>",
        "<|eot_id|>",
        "<end_of_turn>",
        "</s>",
        "<|endoftext|>",
        "<|end|>",
    ]

    for stop_str in common_stop_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(stop_str)
            if isinstance(token_id, int) and token_id != tokenizer.unk_token_id and token_id > 0:
                eos_ids.add(token_id)
        except Exception:
            pass

    return eos_ids


def compute_generation_metrics(
    token_times_ms: list[float], num_generated: int
) -> GenerationMetrics:
    """Compute generation metrics from per-token timing data."""
    if not token_times_ms:
        return GenerationMetrics()

    total_time = sum(token_times_ms)
    ttft = token_times_ms[0]

    # TPOT: average decode time (excludes prefill)
    if len(token_times_ms) > 1:
        tpot = sum(token_times_ms[1:]) / (len(token_times_ms) - 1)
    else:
        tpot = 0.0

    tokens_per_sec = (num_generated / total_time * 1000.0) if total_time > 0 else 0.0

    return GenerationMetrics(
        ttft_ms=ttft,
        tpot_ms=tpot,
        tokens_per_sec=tokens_per_sec,
        total_tokens_generated=num_generated,
        total_time_ms=total_time,
        peak_vram_mb=get_peak_vram_mb(),
    )


def print_metrics(metrics: GenerationMetrics, label: str = ""):
    """Pretty-print generation metrics."""
    header = f"=== {label} ===" if label else "=== Generation Metrics ==="
    print(header)
    print(f"  Tokens generated : {metrics.total_tokens_generated}")
    print(f"  Total time       : {metrics.total_time_ms:>10.2f} ms")
    print(f"  TTFT             : {metrics.ttft_ms:>10.2f} ms")
    print(f"  TPOT             : {metrics.tpot_ms:>10.2f} ms")
    print(f"  Throughput       : {metrics.tokens_per_sec:>10.2f} tok/s")
    print(f"  Peak VRAM        : {metrics.peak_vram_mb:>10.2f} MB")
    print()
