"""LocalLayer — a single linear layer that exposes activations and supports detachment.

In local learning mode, the output is detached from the computation graph
so that no gradient flows backward through the network. Each layer can
still compute its own local gradients (e.g., for forward-forward).
"""

import torch
import torch.nn as nn


class LocalLayer(nn.Module):
    """Single linear layer with activation, exposing pre/post activations.

    Attributes:
        linear: The nn.Linear weight matrix.
        use_activation: Whether to apply ReLU (False for output layer).
        last_pre: Most recent pre-activation input (set during forward).
        last_post: Most recent post-activation output (set during forward).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_activation: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_activation = use_activation
        self.activation = nn.ReLU() if use_activation else nn.Identity()

        # Stored activations (populated during forward pass)
        self.last_pre: torch.Tensor | None = None
        self.last_post: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """Forward pass through this layer.

        Args:
            x: Input tensor (batch_size, in_features).
            detach: If True, detach output from computation graph.

        Returns:
            Output tensor (batch_size, out_features).
        """
        self.last_pre = x
        out = self.activation(self.linear(x))
        self.last_post = out

        if detach:
            return out.detach()
        return out

    @property
    def weight(self) -> torch.Tensor:
        """Access the weight matrix."""
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Access the bias vector."""
        return self.linear.bias
