"""Oja's learning rule — self-normalizing Hebbian that extracts principal components.

Update: Δw = η · mean_over_batch(post · (pre − post · w))

This is equivalent to: Δw = η · (post · pre − post² · w)
The subtraction of post² · w provides automatic normalization,
causing weight vectors to converge toward unit norm (first PC direction).
"""

import torch

from code.rules.base import LayerState


class OjaRule:
    """Oja's normalized Hebbian learning rule.

    Self-normalizing — no explicit weight decay needed.
    Each output neuron's weight vector converges toward the
    first principal component of its input distribution.

    Attributes:
        name: Human-readable identifier.
        lr: Learning rate η.
    """

    name = "oja"

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def compute_update(self, state: LayerState) -> torch.Tensor:
        """Compute Oja's rule weight update.

        Δw = η · mean_over_batch(post · (pre − post · w))

        For each sample:
          post: (out_features,)
          pre: (in_features,)
          w: (out_features, in_features)
          post · w: (in_features,) — reconstruction of input
          pre - post · w: (in_features,) — residual
          post * (pre - post · w): (out_features, in_features) via outer product

        Args:
            state: Layer state with pre/post activations and weights.

        Returns:
            Weight delta of shape (out_features, in_features).
        """
        # post: (batch, out), pre: (batch, in), w: (out, in)
        # reconstruction: (batch, in) = (batch, out) @ (out, in)
        reconstruction = state.post_activation @ state.weights

        # residual: (batch, in)
        residual = state.pre_activation - reconstruction

        # Oja update: mean over batch of outer(post, residual)
        # (batch, out).T @ (batch, in) -> (out, in)
        delta = torch.einsum("bo,bi->oi", state.post_activation, residual)
        delta /= state.pre_activation.size(0)  # mean over batch

        return self.lr * delta

    def reset(self) -> None:
        """No internal state to reset."""
        pass
