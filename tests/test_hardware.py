"""
Tests for hardware profiling and computer scoring algorithms.
"""

from unittest.mock import mock_open, patch

from liefs.hardware_profiler import (
    HardwareProfile,
    calculate_computer_score,
    calculate_effective_memory_bandwidth,
    get_cpu_name,
    get_hardware_profile,
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


def test_computer_score_tiers():
    # Tier S (>= 1200)
    res_s = calculate_computer_score(
        tokens_per_sec=200.0,
        ttft_ms=5.0,
        model_param_count_m=7000.0,  # Large model -> higher weight
        memory_bandwidth_gbs=400.0,
        is_gpu=True,
    )
    assert res_s["badge"] == "Tier S"

    # Tier B (>= 450)
    res_b = calculate_computer_score(
        tokens_per_sec=40.0,
        ttft_ms=30.0,
        model_param_count_m=500.0,
        memory_bandwidth_gbs=60.0,
        is_gpu=True,
    )
    assert res_b["badge"] == "Tier B"

    # Tier C (>= 200)
    res_c = calculate_computer_score(
        tokens_per_sec=10.0,
        ttft_ms=50.0,
        model_param_count_m=500.0,
        memory_bandwidth_gbs=20.0,
        is_gpu=False,
    )
    assert res_c["badge"] == "Tier C"

    # Tier D (< 200)
    res_d = calculate_computer_score(
        tokens_per_sec=2.0,
        ttft_ms=500.0,
        model_param_count_m=500.0,
        memory_bandwidth_gbs=5.0,
        is_gpu=False,
    )
    assert res_d["badge"] == "Tier D"


@patch("platform.system", return_value="Windows")
def test_get_cpu_name_windows(mock_system):
    import sys
    from unittest.mock import MagicMock
    mock_winreg = MagicMock()
    _ = mock_winreg.OpenKey.return_value
    mock_winreg.QueryValueEx.return_value = (
        "Intel(R) Core(TM) i9 CPU  ", None)

    # Patch winreg in sys.modules so the import winreg in hardware_profiler gets the mock
    with patch.dict(sys.modules, {'winreg': mock_winreg}):
        assert get_cpu_name() == "Intel(R) Core(TM) i9 CPU"


@patch("platform.system", return_value="Darwin")
def test_get_cpu_name_darwin(mock_system):
    import sys
    from unittest.mock import MagicMock
    mock_subprocess = MagicMock()
    mock_subprocess.check_output.return_value = b"Apple M2 Max\n"

    with patch.dict(sys.modules, {'subprocess': mock_subprocess}):
        assert get_cpu_name() == "Apple M2 Max"


@patch("platform.system", return_value="Linux")
def test_get_cpu_name_linux(mock_system):
    mock_cpuinfo = "processor\t: 0\nmodel name\t: AMD Ryzen 9 5950X \n"
    with patch("builtins.open", mock_open(read_data=mock_cpuinfo)):
        assert get_cpu_name() == "AMD Ryzen 9 5950X"


@patch("platform.system", return_value="Unknown")
@patch("platform.processor", return_value="Generic CPU")
def test_get_cpu_name_fallback(mock_processor, mock_system):
    assert get_cpu_name() == "Generic CPU"


@patch("platform.system", return_value="Linux")
@patch("platform.processor", return_value="Generic CPU")
def test_get_cpu_name_exception(mock_processor, mock_system):
    # Simulate an error (e.g. /proc/cpuinfo not found) to trigger the exception block
    with patch("builtins.open", side_effect=FileNotFoundError):
        assert get_cpu_name() == "Generic CPU"
