"""
INT8 Quantization Module for LIEFS.
"""

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        max_val = torch.max(torch.abs(weight), dim=1, keepdim=True).values
        scale = max_val.clamp(min=1e-5) / 127.0
        
        quantized_weight = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
        
        q_linear = cls(linear.in_features, linear.out_features, linear.bias is not None, dtype=weight.dtype)
        
        q_linear.weight.copy_(quantized_weight)
        q_linear.scale.copy_(scale.squeeze(1))
        
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data)
            
        return q_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly dequantization.
        Dequantization formula: weight_fp = weight_int8.float() * scale
        """
        weight_deq = self.weight.to(input.dtype) * self.scale.unsqueeze(1)
        return F.linear(input, weight_deq, self.bias)


@torch.no_grad()
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
            
            if full_name == "lm_head":
                continue
                
            if isinstance(child, nn.Linear):
                q_linear = QuantizedLinear.from_linear(child)
                q_linear = q_linear.to(child.weight.device)
                setattr(module, name, q_linear)
                replaced_count += 1
            else:
                _replace_linear(child, full_name)
                
    _replace_linear(model)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Replaced {replaced_count} nn.Linear layers with QuantizedLinear.")
    return model
