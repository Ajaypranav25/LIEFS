"""
Model and tokenizer loading for LIEFS.

This is the ONLY module that touches HuggingFace's model loading machinery.
All generation logic lives in the engine modules.

Design decisions:
- We load in float16, not bfloat16. Both are supported on RTX 4060 (Ada
  Lovelace), but fp16 has slightly more mature kernel support and identical
  precision for our inference-only use case.
- We use device_map="cuda" to load directly onto GPU, avoiding a CPU→GPU
  copy of ~1GB of weights.
- model.eval() disables dropout (not that Qwen2.5 uses it in inference mode,
  but it's good practice and signals intent).
- We never call model.generate() — that's the whole point of this project.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default model — small enough to iterate fast, complex enough to be
# interesting (GQA, SwiGLU, RoPE, RMSNorm, tied embeddings).
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model weights and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.
        dtype: Weight precision. Default fp16.
        device: Target device. Default "cuda".

    Returns:
        (model, tokenizer) tuple. Model is in eval mode with gradients disabled.
    """
    print(f"Loading model: {model_name}")
    print(f"  dtype: {dtype}, device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
    )
    model.eval()

    # Sanity checks
    param_count = sum(p.numel() for p in model.parameters())
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
    print(f"  VRAM after load: {vram_mb:.1f} MB")
    print(f"  EOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print()

    return model, tokenizer


def format_chat_prompt(
    tokenizer: AutoTokenizer,
    user_message: str,
    system_message: str = "You are a helpful assistant.",
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Format a user message using the model's chat template and tokenize.

    Uses the Qwen2.5 ChatML format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant

    Args:
        tokenizer: The loaded tokenizer.
        user_message: The user's prompt text.
        system_message: System prompt. Default is a simple assistant prompt.
        device: Target device for output tensor. Default "cuda".

    Returns:
        Token IDs as a tensor of shape (1, seq_len) on the requested device.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # apply_chat_template returns token IDs directly when tokenize=True.
    # add_generation_prompt=True appends "<|im_start|>assistant\n" so the
    # model knows it should start generating the assistant's response.
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    return input_ids.to(device)
