"""Base protocols and data structures for local learning rules.

Defines the LayerState dataclass (all information available to a rule at one layer)
and the LocalLearningRule / ThirdFactorInterface protocols.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class LayerState:
    """All information available to a local learning rule at one layer.

    This is the complete "view" a rule has — no gradient information
    from other layers is included.
    """

    pre_activation: torch.Tensor  # Input to the layer (batch_size, in_features)
    post_activation: torch.Tensor  # Output after activation (batch_size, out_features)
    weights: torch.Tensor  # Current weight matrix (out_features, in_features)
    bias: torch.Tensor | None  # Current bias vector (out_features,) or None
    layer_index: int  # Position in the network (0-indexed)
    labels: torch.Tensor | None = None  # Batch labels (for rules that use them)
    global_loss: float | None = None  # Current batch loss (for reward-based rules)


class LocalLearningRule(Protocol):
    """Contract for all local learning rules.

    A local rule computes weight updates using only information
    available at the layer (no backward gradient flow through the network).
    """

    @property
    def name(self) -> str:
        """Human-readable name for results reporting."""
        ...

    def compute_update(self, state: LayerState) -> torch.Tensor:
        """Compute the weight update for this layer.

        Args:
            state: All locally available information at this layer.

        Returns:
            Weight delta tensor of same shape as state.weights.
            The caller applies: weights += delta.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (e.g., eligibility traces) between epochs."""
        ...


class ThirdFactorInterface(Protocol):
    """Pluggable interface for the third factor signal in three-factor learning.

    Step 12 provides three placeholder implementations.
    Step 13 will provide AstrocyteGate as a drop-in replacement.
    """

    @property
    def name(self) -> str:
        """Identifier for this third factor source."""
        ...

    def compute_signal(
        self,
        layer_activations: torch.Tensor,
        layer_index: int,
        labels: torch.Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> torch.Tensor:
        """Compute the third factor modulation signal.

        Args:
            layer_activations: Post-activation output (batch, out_features).
            layer_index: Which layer this is (0-indexed).
            labels: Ground truth labels for the batch.
            global_loss: Current batch loss value.
            prev_loss: Previous batch loss value (for reward computation).

        Returns:
            Modulation signal — scalar, (out_features,), or (out, in).
        """
        ...
