"""Stability fix for three-factor learning rule.

Step 12 showed loss explosion with layer-wise error third factor.
These functions prevent numerical instability by:
1. Clipping error signals to bounded magnitude (preserving sign)
2. Normalizing eligibility traces when their norm grows too large

Applied within the training loop wrapping Step 12's ThreeFactorRule.
"""

import torch
from torch import Tensor


def clip_error_signal(error: Tensor, threshold: float = 10.0) -> Tensor:
    """Clip error signal magnitude while preserving sign.

    Args:
        error: Error signal tensor of any shape.
        threshold: Maximum absolute value allowed (default 10.0).

    Returns:
        Clipped tensor with all elements in [-threshold, threshold].
    """
    return torch.clamp(error, -threshold, threshold)


def normalize_eligibility(
    trace: Tensor,
    norm_threshold: float = 100.0,
    safe_constant: float = 1.0,
) -> Tensor:
    """Normalize eligibility trace when its norm exceeds threshold.

    When the Frobenius norm of the trace exceeds norm_threshold,
    rescale the trace so its norm equals safe_constant. This preserves
    the direction (unit vector) of the trace while bounding magnitude.

    Args:
        trace: Eligibility trace tensor of any shape.
        norm_threshold: Norm above which normalization is applied.
        safe_constant: Target norm after normalization.

    Returns:
        Original trace if norm <= threshold, normalized trace otherwise.
    """
    norm = trace.norm()
    if norm > norm_threshold:
        return trace * (safe_constant / norm)
    return trace
