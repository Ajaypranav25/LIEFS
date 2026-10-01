"""
Tests for universal model loading helpers, resolution, presets, and metadata.
"""

from unittest.mock import MagicMock
import torch

from liefs.model_loader import (
    POPULAR_MODEL_PRESETS,
    resolve_device,
    resolve_dtype,
    get_model_metadata,
    format_chat_prompt,
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


def test_get_model_metadata():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_config = MagicMock()
    mock_config.num_hidden_layers = 12
    mock_config.hidden_size = 768
    mock_config.num_attention_heads = 12
    mock_config.vocab_size = 50000
    mock_config.max_position_embeddings = 1024
    mock_config.architectures = ["DummyCausalLM"]

    mock_model.config = mock_config

    mock_param = MagicMock()
    mock_param.numel.return_value = 1000000
    mock_param.is_cuda = False
    mock_param.dtype = torch.float32

    # Needs to be an iterator/generator when called
    def params_generator():
        for _ in range(10):
            yield mock_param

    mock_model.parameters.side_effect = params_generator

    mock_tokenizer.__len__.return_value = 50000

    metadata = get_model_metadata(mock_model, mock_tokenizer, "dummy-model")

    assert metadata.model_name == "dummy-model"
    assert metadata.parameter_count == 10000000
    assert metadata.num_layers == 12
    assert metadata.hidden_size == 768
    assert metadata.num_attention_heads == 12
    assert metadata.vocab_size == 50000
    assert metadata.max_position_embeddings == 1024
    assert metadata.device_str == "cpu"
    assert metadata.architectures == ["DummyCausalLM"]


def test_format_chat_prompt_success():
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])

    tensor = format_chat_prompt(mock_tokenizer, "Hello")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3)
    mock_tokenizer.apply_chat_template.assert_called_once()


def test_format_chat_prompt_fallback():
    mock_tokenizer = MagicMock()
    # Force an exception to trigger the fallback block
    mock_tokenizer.apply_chat_template.side_effect = Exception("No template")
    mock_tokenizer.encode.return_value = torch.tensor([[4, 5, 6]])

    tensor = format_chat_prompt(mock_tokenizer, "Hello")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3)
    mock_tokenizer.encode.assert_called_once()
