"""
Correctness tests for Stage 5: INT8 Quantization.
"""

import pytest
import torch

from liefs.kv_cache_engine import KVCacheEngine
from liefs.model_loader import format_chat_prompt, load_model_and_tokenizer
from liefs.quantization import QuantizedLinear, quantize_model


@pytest.fixture(scope="module")
def loaded_components():
    """Load model, record original VRAM, quantize, and return engine + stats."""
    model, tokenizer = load_model_and_tokenizer()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_before = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        vram_before = 0.0

    quantize_model(model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_after = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        vram_after = 0.0

    engine = KVCacheEngine(model, tokenizer)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "engine": engine,
        "vram_before": vram_before,
        "vram_after": vram_after,
    }


class TestQuantization:
    def test_quantization_reduces_memory(self, loaded_components):
        """VRAM usage should decrease after INT8 weight quantization."""
        if not torch.cuda.is_available():
            pytest.skip("Test requires CUDA to measure VRAM.")
        vram_before = loaded_components["vram_before"]
        vram_after = loaded_components["vram_after"]
        assert vram_after < vram_before, (
            f"Expected VRAM reduction. Before: {vram_before:.1f} MB, After: {vram_after:.1f} MB"
        )

    def test_lm_head_not_quantized(self, loaded_components):
        """lm_head should remain standard nn.Linear in fp16 (to preserve logits quality)."""
        model = loaded_components["model"]
        assert isinstance(model.lm_head, torch.nn.Linear)
        assert not isinstance(model.lm_head, QuantizedLinear)

    def test_linear_layers_quantized(self, loaded_components):
        """At least the decoder layers should be replaced by QuantizedLinear."""
        model = loaded_components["model"]
        q_count = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
        assert q_count > 0, "No QuantizedLinear layers found in model"

    def test_quantized_output_coherent(self, loaded_components):
        """Quantized model should still produce coherent, sensible output."""
        engine = loaded_components["engine"]
        tokenizer = loaded_components["tokenizer"]

        input_ids = format_chat_prompt(tokenizer, "What is 2+2?")
        gen_ids, _ = engine.generate(input_ids, max_new_tokens=32)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        assert "4" in text, f"Expected '4' in output, got: {text!r}"

    def test_quantized_deterministic(self, loaded_components):
        """Greedy decoding with quantized weights should be deterministic."""
        engine = loaded_components["engine"]
        tokenizer = loaded_components["tokenizer"]

        input_ids = format_chat_prompt(tokenizer, "Explain gravity in one sentence.")
        ids_1, _ = engine.generate(input_ids, max_new_tokens=32)
        ids_2, _ = engine.generate(input_ids, max_new_tokens=32)

        assert ids_1 == ids_2, "Quantized greedy generation was non-deterministic"
