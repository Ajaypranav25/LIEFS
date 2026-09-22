"""
Tests for universal model loading helpers, resolution, presets, and metadata.
"""

import torch

from liefs.model_loader import (
    POPULAR_MODEL_PRESETS,
    resolve_device,
    resolve_dtype,
)


def test_resolve_dtype():
    assert resolve_dtype("float16") == torch.float16
    assert resolve_dtype("fp16") == torch.float16
    assert resolve_dtype("float32") == torch.float32
    assert resolve_dtype("fp32") == torch.float32


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("auto") == "cuda"
    else:
        assert resolve_device("cuda") == "cpu"
        assert resolve_device("auto") == "cpu"


def test_popular_model_presets():
    assert len(POPULAR_MODEL_PRESETS) >= 5
    for preset in POPULAR_MODEL_PRESETS:
        assert "id" in preset
        assert "name" in preset
        assert "params_m" in preset
        assert "recommended_vram_mb" in preset
