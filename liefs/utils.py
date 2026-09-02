"""
Timing and logging utilities for LIEFS.

Design note: We wrap all GPU timing with torch.cuda.synchronize() because
CUDA operations are *asynchronous* — the CPU queues work on the GPU and
returns immediately. Without synchronize(), time.perf_counter() measures
how fast the CPU can *enqueue* work, not how fast the GPU *executes* it.
This is a classic benchmarking pitfall.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class GenerationMetrics:
    """Metrics collected from a single generation run.

    Attributes:
        ttft_ms: Time-to-first-token in milliseconds. Measures the latency
            from submitting the prompt to producing the first output token.
            In a real serving scenario, this is the user-perceived "thinking"
            time. Dominated by the cost of the initial prefill pass.
        tpot_ms: Time-per-output-token in milliseconds (average, excluding
            first token). This determines the streaming speed — how fast
            tokens appear to the user after the first one.
        tokens_per_sec: Total output tokens / total wall time. The headline
            throughput number.
        total_tokens_generated: Number of tokens generated (excluding prompt).
        total_time_ms: Total wall time for the full generation in milliseconds.
        peak_vram_mb: Peak GPU VRAM allocated during generation, in MB.
            Measured via torch.cuda.max_memory_allocated(), which tracks the
            high-water mark of the PyTorch CUDA memory allocator.
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

    Uses torch.cuda.synchronize() to ensure accurate GPU timing.

    Usage:
        with cuda_timer() as elapsed:
            # ... GPU work ...
        print(f"Took {elapsed()} ms")
    """
    torch.cuda.synchronize()
    start = time.perf_counter()

    # We store the result in a list so the inner function can mutate it
    result = [0.0]

    def get_elapsed() -> float:
        return result[0]

    yield get_elapsed

    torch.cuda.synchronize()
    result[0] = (time.perf_counter() - start) * 1000.0  # Convert to ms


def get_peak_vram_mb() -> float:
    """Return peak VRAM usage in MB since last reset."""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_vram_stats():
    """Reset peak VRAM tracking. Call before each benchmark run."""
    torch.cuda.reset_peak_memory_stats()


def get_eos_token_ids(tokenizer) -> set[int]:
    """Extract all relevant EOS token IDs for generation termination.

    Includes standard tokenizer.eos_token_id and chat turn terminators (e.g. <|im_end|>).
    """
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        eos_ids.add(im_end_id)
    return eos_ids


def compute_generation_metrics(
    token_times_ms: list[float], num_generated: int
) -> GenerationMetrics:
    """Compute generation metrics from per-token timing data.

    Args:
        token_times_ms: List of durations (in ms) for each step (index 0 is prefill/TTFT).
        num_generated: Total number of generated tokens.
    """
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
