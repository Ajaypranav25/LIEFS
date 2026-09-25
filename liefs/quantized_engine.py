"""
Quantized Generation Engine for LIEFS.
"""

from liefs.kv_cache_engine import KVCacheEngine
from liefs.quantization import quantize_model


def create_quantized_engine(model, tokenizer):
    """
    Quantizes the model in-place and wraps it in a KVCacheEngine.
    """
    print("Quantizing model...")
    quantized_model = quantize_model(model)
    print("Model quantized. Creating engine...")

    engine = KVCacheEngine(quantized_model, tokenizer)
    return engine
