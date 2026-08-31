"""
INT8 Quantization Module for LIEFS.

This module provides a from-scratch implementation of INT8 weight-only quantization.
We use symmetric per-channel quantization for the weight matrices.

Design Decisions & Interview Talking Points:
1. Why per-channel vs per-tensor?
   - Per-tensor quantization computes a single scale for the entire weight matrix.
   - Per-channel computes a separate scale for each output channel (row) of the matrix.
   - Per-channel preserves significantly more information because different channels often
     have different dynamic ranges (outliers). This greatly reduces quantization error.

2. Why symmetric?
   - Symmetric quantization maps the maximum absolute value to 127 and uses 0 as the origin.
   - It is computationally simpler because there is no zero-point offset to add/subtract
     during dequantization/matmul.
   - Weights in neural networks are usually roughly zero-centered anyway.

3. Why skip lm_head?
   - The lm_head is the final projection from the hidden state size to the vocabulary size.
   - Quantization error here directly and immediately affects the final token logits and probabilities.
   - Since lm_head is tied with embed_tokens, quantizing it is complex and often hurts generation
     quality disproportionately compared to the memory saved.

4. Memory Savings:
   - A float16 weight takes 2 bytes.
   - An int8 weight takes 1 byte, plus a float16 scale factor per channel.
   - For a layer of size (out_features, in_features), memory goes from:
     2 * out * in  =>  1 * out * in + 2 * out.
   - This results in roughly a 50% compression ratio.

5. Simplification vs Production:
   - This is naive RTN (Round-To-Nearest) quantization.
   - Real production systems use calibration-based methods like GPTQ or AWQ to compute optimal
     scales/weights that minimize activation error.
   - Production systems also use custom INT8 GEMM (General Matrix Multiply) CUDA kernels that
     compute the dot product in integer math, returning float16/float32.
   - We do "on-the-fly dequantization" where we cast int8 to float16 and multiply by scale
     before a standard F.linear(). This saves memory but doesn't speed up compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

class QuantizedLinear(nn.Module):
    """
    A quantized linear layer that stores weights in INT8 and dequantizes on the fly.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights as int8
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8))
        # Store scales as original dtype (usually float16)
        self.register_buffer("scale", torch.zeros(out_features, dtype=dtype))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        """
        Create a QuantizedLinear from a standard nn.Linear.
        Performs per-channel symmetric quantization.
        """
        weight = linear.weight.data
        
        # 1. Compute scale per output channel: max(abs(row)) / 127
        max_val = torch.max(torch.abs(weight), dim=1, keepdim=True).values
        scale = max_val.clamp(min=1e-5) / 127.0
        
        # 2. Quantize weight: round(weight / scale).clamp(-128, 127)
        quantized_weight = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
        
        # Create new module
        q_linear = cls(linear.in_features, linear.out_features, linear.bias is not None, dtype=weight.dtype)
        
        # Assign buffers
        q_linear.weight.copy_(quantized_weight)
        q_linear.scale.copy_(scale.squeeze(1)) # shape: (out_features,)
        
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data)
            
        return q_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly dequantization.
        Dequantization formula: weight_fp = weight_int8.float() * scale
        """
        # Dequantize
        weight_deq = self.weight.to(input.dtype) * self.scale.unsqueeze(1)
        
        # Linear operation
        return F.linear(input, weight_deq, self.bias)

def quantize_model(model: nn.Module) -> nn.Module:
    """
    Recursively replaces all nn.Linear layers in the model with QuantizedLinear.
    Skips the lm_head to preserve generation quality.
    """
    replaced_count = 0
    
    def _replace_linear(module: nn.Module, name_prefix: str = ""):
        nonlocal replaced_count
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            # Skip lm_head
            if full_name == "lm_head":
                continue
                
            if isinstance(child, nn.Linear):
                # Replace
                q_linear = QuantizedLinear.from_linear(child)
                q_linear = q_linear.to(child.weight.device)
                setattr(module, name, q_linear)
                replaced_count += 1
            else:
                # Recurse
                _replace_linear(child, full_name)
                
    _replace_linear(model)
    
    # Free up memory from old weights
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Replaced {replaced_count} nn.Linear layers with QuantizedLinear.")
    return model
