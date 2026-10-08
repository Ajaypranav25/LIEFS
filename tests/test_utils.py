import io
import sys
import time
from unittest.mock import MagicMock, patch

from liefs.utils import (
    GenerationMetrics,
    compute_generation_metrics,
    cuda_timer,
    get_eos_token_ids,
    get_peak_vram_mb,
    print_metrics,
    reset_vram_stats,
)


def test_cuda_timer():
    with cuda_timer() as get_elapsed:
        time.sleep(0.01)

    elapsed = get_elapsed()
    assert elapsed > 0
    # Should be at least ~10ms
    assert elapsed >= 8.0


@patch("torch.cuda.is_available", return_value=False)
def test_get_peak_vram_mb_cpu(mock_is_available):
    assert get_peak_vram_mb() == 0.0


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.max_memory_allocated", return_value=104857600)  # 100 MB
def test_get_peak_vram_mb_gpu(mock_max_mem, mock_is_available):
    assert get_peak_vram_mb() == 100.0


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.reset_peak_memory_stats")
def test_reset_vram_stats_gpu(mock_reset, mock_is_available):
    reset_vram_stats()
    mock_reset.assert_called_once()


@patch("torch.cuda.is_available", return_value=False)
def test_reset_vram_stats_cpu(mock_is_available):
    # Should not raise any errors
    reset_vram_stats()


def test_get_eos_token_ids():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 42
    tokenizer.unk_token_id = 0

    # Mocking convert_tokens_to_ids to return specific IDs for common tokens
    def mock_convert(token):
        mapping = {
            "<|im_end|>": 100,
            "<|eot_id|>": 101,
            "</s>": 102,
        }
        return mapping.get(token, tokenizer.unk_token_id)

    tokenizer.convert_tokens_to_ids.side_effect = mock_convert

    eos_ids = get_eos_token_ids(tokenizer)

    assert 42 in eos_ids
    assert 100 in eos_ids
    assert 101 in eos_ids
    assert 102 in eos_ids
    assert 0 not in eos_ids


def test_get_eos_token_ids_list():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = [42, 43]
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 0

    eos_ids = get_eos_token_ids(tokenizer)
    assert 42 in eos_ids
    assert 43 in eos_ids


def test_compute_generation_metrics():
    # Empty case
    empty_metrics = compute_generation_metrics([], 0)
    assert empty_metrics.ttft_ms == 0.0
    assert empty_metrics.tpot_ms == 0.0

    # Single token
    single_metrics = compute_generation_metrics([50.0], 1)
    assert single_metrics.ttft_ms == 50.0
    assert single_metrics.tpot_ms == 0.0
    assert single_metrics.total_time_ms == 50.0
    assert single_metrics.tokens_per_sec == 20.0

    # Multiple tokens
    multi_metrics = compute_generation_metrics([50.0, 10.0, 20.0], 3)
    assert multi_metrics.ttft_ms == 50.0
    assert multi_metrics.tpot_ms == 15.0  # (10 + 20) / 2
    assert multi_metrics.total_time_ms == 80.0
    assert multi_metrics.tokens_per_sec == 3 / 0.08  # 37.5


def test_print_metrics():
    metrics = GenerationMetrics(
        ttft_ms=50.0,
        tpot_ms=15.0,
        tokens_per_sec=37.5,
        total_tokens_generated=3,
        total_time_ms=80.0,
        peak_vram_mb=100.0,
    )

    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        print_metrics(metrics, label="Test Label")
    finally:
        sys.stdout = original_stdout

    output = captured_output.getvalue()

    assert "=== Test Label ===" in output
    assert "3" in output
    assert "80.00" in output
    assert "50.00" in output
    assert "15.00" in output
    assert "37.50" in output
    assert "100.00" in output
