"""
Model and tokenizer loading for LIEFS.

Supports universal loading of any Hugging Face CausalLM model or local checkpoint,
dynamic precision and device allocation (CUDA / CPU), chat template parsing,
and model architecture telemetry extraction.
"""

import os
import torch
from dataclasses import dataclass, asdict
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default model — lightweight, GQA, SwiGLU, RoPE, RMSNorm
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Curated catalog of benchmark-ready models for quick testing
POPULAR_MODEL_PRESETS = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen 2.5 0.5B Instruct",
        "family": "Qwen",
        "size_label": "0.5B",
        "params_m": 494,
        "recommended_vram_mb": 1200,
        "description": "24 Layers, GQA with 2 KV heads, SwiGLU MLP. Highly optimized baseline for laptops and entry GPUs.",
        "badge": "Default"
    },
    {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "name": "SmolLM2 360M Instruct",
        "family": "SmolLM",
        "size_label": "360M",
        "params_m": 360,
        "recommended_vram_mb": 900,
        "description": "Ultra-lightweight high-speed model. Great for quick sanity checks and CPU benchmarking.",
        "badge": "Ultra Fast"
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama 1.1B Chat",
        "family": "Llama",
        "size_label": "1.1B",
        "params_m": 1100,
        "recommended_vram_mb": 2400,
        "description": "Classic Llama architecture with 22 layers, 32 Q heads, 4 KV heads. Standard open benchmark.",
        "badge": "Popular"
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen 2.5 1.5B Instruct",
        "family": "Qwen",
        "size_label": "1.5B",
        "params_m": 1540,
        "recommended_vram_mb": 3500,
        "description": "28 Layers, 12 Q heads, 2 KV heads. Excellent balance of inference speed and reasoning capability.",
        "badge": "Mid-Weight"
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 1B Instruct",
        "family": "Llama",
        "size_label": "1.2B",
        "params_m": 1235,
        "recommended_vram_mb": 2800,
        "description": "Meta's lightweight model with 128k context support, GQA, and RoPE scaling.",
        "badge": "Meta Llama"
    },
    {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "family": "DeepSeek",
        "size_label": "1.5B",
        "params_m": 1540,
        "recommended_vram_mb": 3500,
        "description": "Reasoning-distilled model trained on chain-of-thought traces. High compute density.",
        "badge": "Reasoning"
    },
]


@dataclass
class ModelArchitectureMetadata:
    """Detailed structural metadata of a loaded model."""
    model_name: str
    parameter_count: int
    parameter_count_m: float
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    dtype_str: str
    device_str: str
    memory_footprint_mb: float
    fp16_vram_estimate_mb: float
    int8_vram_estimate_mb: float
    int4_vram_estimate_mb: float
    architectures: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_dtype(dtype_str: str = "float16") -> torch.dtype:
    """Resolve string representation to torch.dtype."""
    d = dtype_str.lower().strip()
    if d in ("float16", "fp16"):
        return torch.float16
    elif d in ("bfloat16", "bf16"):
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    elif d in ("float32", "fp32"):
        return torch.float32
    return torch.float16


def resolve_device(device_str: str = "auto") -> str:
    """Resolve device preference based on hardware availability."""
    d = device_str.lower().strip()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return d


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    dtype: torch.dtype | str = torch.float16,
    device: str = "auto",
    hf_token: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Universal model loader for any Hugging Face model or local directory.

    Args:
        model_name: Hugging Face repo ID (e.g. "Qwen/Qwen2.5-0.5B-Instruct") or local path.
        dtype: Weight precision (torch.float16, torch.bfloat16, torch.float32 or str).
        device: Target device ("cuda", "cpu", or "auto").
        hf_token: Optional Hugging Face token for gated models.

    Returns:
        (model, tokenizer) tuple in eval mode.
    """
    target_device = resolve_device(device)

    if isinstance(dtype, str):
        target_dtype = resolve_dtype(dtype)
    else:
        target_dtype = dtype

    if target_device == "cpu" and target_dtype == torch.float16:
        # Use float32 on CPU to prevent numerical instability differences
        target_dtype = torch.float32

    print(f"Loading model: {model_name}")
    print(f"  Target precision: {target_dtype}, Target device: {target_device}")

    # Set Hugging Face token if provided
    token = hf_token or os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load causal LM
    load_kwargs: dict[str, Any] = {
        "dtype": target_dtype,
        "token": token,
        "trust_remote_code": True,
    }

    if target_device == "cuda":
        load_kwargs["device_map"] = "cuda"
    else:
        load_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded successfully! Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    if target_device == "cuda" and torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  GPU VRAM occupied: {vram_mb:.1f} MB")

    return model, tokenizer


def get_model_metadata(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str = DEFAULT_MODEL_NAME,
) -> ModelArchitectureMetadata:
    """Extract full architecture specifications and memory calculations."""
    config = model.config
    param_count = sum(p.numel() for p in model.parameters())
    param_count_m = round(param_count / 1e6, 2)

    # Generalized config extraction across Llama, Qwen, Gemma, Mistral, Phi, GPT-2
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 24))
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 1024))
    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 16))
    num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", num_heads))
    head_dim = getattr(config, "head_dim", hidden_size // num_heads if num_heads > 0 else 64)
    vocab_size = getattr(config, "vocab_size", len(tokenizer))
    max_pos = getattr(config, "max_position_embeddings", getattr(config, "max_sequence_length", 4096))
    archs = getattr(config, "architectures", ["CausalLM"])

    # Memory calculations
    # FP16: 2 bytes/param, INT8: 1 byte/param, INT4: 0.5 bytes/param
    fp16_mb = round((param_count * 2) / (1024 * 1024), 1)
    int8_mb = round((param_count * 1) / (1024 * 1024), 1)
    int4_mb = round((param_count * 0.5) / (1024 * 1024), 1)

    device_str = "cuda" if next(model.parameters()).is_cuda else "cpu"
    dtype_str = str(next(model.parameters()).dtype).replace("torch.", "")

    if device_str == "cuda" and torch.cuda.is_available():
        memory_footprint_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
    else:
        memory_footprint_mb = fp16_mb

    return ModelArchitectureMetadata(
        model_name=model_name,
        parameter_count=param_count,
        parameter_count_m=param_count_m,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        dtype_str=dtype_str,
        device_str=device_str,
        memory_footprint_mb=memory_footprint_mb,
        fp16_vram_estimate_mb=fp16_mb,
        int8_vram_estimate_mb=int8_mb,
        int4_vram_estimate_mb=int4_mb,
        architectures=archs,
    )


def format_chat_prompt(
    tokenizer: AutoTokenizer,
    user_message: str,
    system_message: str = "You are a helpful assistant.",
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Format a user message using the tokenizer's chat template with fallback.

    Args:
        tokenizer: The loaded tokenizer.
        user_message: The user's prompt text.
        system_message: System prompt.
        device: Target device for output tensor.

    Returns:
        Token IDs tensor of shape (1, seq_len) on the requested device.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
    except Exception:
        # Fallback for tokenizers without chat templates or with syntax incompatibilities
        fallback_text = f"System: {system_message}\nUser: {user_message}\nAssistant:"
        input_ids = tokenizer.encode(fallback_text, return_tensors="pt")

    if isinstance(device, str):
        target_dev = resolve_device(device)
    else:
        target_dev = device

    return input_ids.to(target_dev)
