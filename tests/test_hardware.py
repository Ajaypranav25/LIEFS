"""
Tests for hardware profiling and computer scoring algorithms.
"""

import pytest
from liefs.hardware_profiler import (
    get_hardware_profile,
    calculate_effective_memory_bandwidth,
    calculate_computer_score,
    HardwareProfile,
)


def test_hardware_profile_detection():
    hw = get_hardware_profile()
    assert isinstance(hw, HardwareProfile)
    assert len(hw.os_name) > 0
    assert hw.cpu_physical_cores >= 1
    assert hw.cpu_logical_cores >= 1
    assert hw.ram_total_gb > 0.0
    assert isinstance(hw.cuda_available, bool)
    
    d = hw.to_dict()
    assert "cpu_model" in d
    assert "ram_total_gb" in d
    assert "vram_total_mb" in d


def test_effective_memory_bandwidth_calculation():
    # 1024 MB model (1 GB) generating 100 tok/s -> 100 GB/s
    bw = calculate_effective_memory_bandwidth(1024.0, 100.0)
    assert bw == 100.0

    # 512 MB model (0.5 GB) generating 50 tok/s -> 25 GB/s
    bw2 = calculate_effective_memory_bandwidth(512.0, 50.0)
    assert bw2 == 25.0

    # Edge cases
    assert calculate_effective_memory_bandwidth(0.0, 100.0) == 0.0
    assert calculate_effective_memory_bandwidth(1024.0, 0.0) == 0.0


def test_computer_score_calculation():
    # Fast GPU simulation: 100 tok/s, 15ms TTFT, 500M model, 95 GB/s
    res = calculate_computer_score(
        tokens_per_sec=100.0,
        ttft_ms=15.0,
        model_param_count_m=500.0,
        memory_bandwidth_gbs=95.0,
        is_gpu=True,
    )
    assert isinstance(res["score"], int)
    assert res["score"] > 800
    assert "tier" in res
    assert "badge" in res
    assert "breakdown" in res

    # Zero throughput
    zero_res = calculate_computer_score(0.0, 0.0, 500.0, 0.0)
    assert zero_res["score"] == 0
