"""Hebbian learning rule — the simplest local learning baseline.

Update: Δw = η · mean_over_batch(outer(post, pre)) − λ · weights

This is pure correlation-based learning with weight decay to prevent
unbounded growth. No error signal, no normalization beyond decay.
"""

import torch

from code.rules.base import LayerState


class HebbianRule:
    """Basic Hebbian learning rule with weight decay.

    Attributes:
        name: Human-readable identifier.
        lr: Learning rate η.
        weight_decay: Decay rate λ.
        max_norm: Maximum allowed weight matrix norm (explosion guard).
    """

    name = "hebbian"

    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0.001,
        max_norm: float = 100.0,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_norm = max_norm

    def compute_update(self, state: LayerState) -> torch.Tensor:
        """Compute Hebbian weight update.

        Δw = η · mean_over_batch(outer(post, pre)) − λ · weights

        Args:
            state: Layer state with pre/post activations and weights.

        Returns:
            Weight delta of shape (out_features, in_features).
        """
        # post: (batch, out), pre: (batch, in)
        # outer product averaged over batch
        hebbian_term = torch.einsum("bo,bi->oi", state.post_activation, state.pre_activation)
        hebbian_term /= state.pre_activation.size(0)  # mean over batch

        decay_term = self.weight_decay * state.weights

        delta = self.lr * hebbian_term - decay_term

        # Weight explosion guard: clip if applying update would exceed max_norm
        new_weights = state.weights + delta
        norm = torch.norm(new_weights)
        if norm > self.max_norm:
            # Scale delta so resulting weights have norm = max_norm
            delta = (new_weights / norm) * self.max_norm - state.weights

        return delta

    def reset(self) -> None:
        """No internal state to reset."""
        pass
