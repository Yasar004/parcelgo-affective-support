"""GPU capability checks shared by training and generation.

Colab's free T4 (compute capability 7.5) has no native bfloat16, yet
`torch.cuda.is_bf16_supported()` returns True there through emulation.
Choosing bf16 on a T4 makes TRL's fused loss kernel call a mixed-precision
matmul that only exists on capability >= 8.0 (A100, L4, H100), so training
crashes. We therefore decide on the real capability.
"""

from __future__ import annotations

import torch


def native_bf16() -> bool:
    """True only on GPUs with hardware bfloat16 (compute capability >= 8.0)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if native_bf16() else torch.float16


def patch_trl_for_older_gpus() -> None:
    """Make TRL's chunked log-prob kernel use plain fp32 matmuls on pre-Ampere GPUs.

    TRL accumulates the loss with `torch.addmm(..., out_dtype=float32)` on
    half-precision inputs, which requires compute capability >= 8.0. Its
    fallback path upcasts to fp32 first: numerically the same, slightly slower.
    """
    if not torch.cuda.is_available() or native_bf16():
        return
    try:
        import trl.kernels.chunked_logprob as chunked
    except ImportError:  # older TRL without the fused kernel
        return
    if hasattr(chunked, "_MM_OUT_DTYPE"):
        chunked._MM_OUT_DTYPE = False
        print("older GPU detected: using TRL's fp32 fallback for the loss kernel")


def upcast_trainable_params(model) -> None:
    """Keep trainable (LoRA) weights in fp32 on pre-Ampere GPUs.

    TRL casts the adapter weights of a quantized model to bf16 (QLoRA paper
    recommendation). On a T4 we train with fp16 mixed precision, whose gradient
    scaler cannot unscale bf16 gradients, so training crashes at the first
    optimizer step. fp32 adapters with fp16 autocast is the standard setup on
    these GPUs. Call this after building the trainer and before `train()`.
    """
    if not torch.cuda.is_available() or native_bf16():
        return
    n = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
            n += 1
    if n:
        print(f"older GPU detected: keeping {n} trainable tensors in fp32")
