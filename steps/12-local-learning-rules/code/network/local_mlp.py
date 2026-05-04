"""LocalMLP — 4-hidden-layer MLP with detached inter-layer activations.

Architecture: 784 → 128 (ReLU) → 128 (ReLU) → 128 (ReLU) → 128 (ReLU) → 10
Same dimensions as Phase 1's DeeperMLP, but forward pass detaches
activations between layers to enforce locality.
"""

import torch
import torch.nn as nn

from code.network.local_layer import LocalLayer
from code.rules.base import LayerState


class LocalMLP(nn.Module):
    """4-hidden-layer MLP with optional inter-layer detachment.

    In local learning mode (detach=True), each layer's output is detached
    before being passed to the next layer. This prevents gradient flow
    across layer boundaries, enforcing that each layer can only learn
    from locally available information.

    Args:
        input_size: Input dimension (default 784 for flattened 28x28).
        hidden_size: Hidden layer width (default 128).
        n_classes: Output dimension (default 10).
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        n_classes: int = 10,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.layers = nn.ModuleList([
            LocalLayer(input_size, hidden_size, use_activation=True),
            LocalLayer(hidden_size, hidden_size, use_activation=True),
            LocalLayer(hidden_size, hidden_size, use_activation=True),
            LocalLayer(hidden_size, hidden_size, use_activation=True),
            LocalLayer(hidden_size, n_classes, use_activation=False),
        ])

    def forward(self, x: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """Forward pass with optional inter-layer detachment.

        Args:
            x: Input tensor (batch_size, 784) or (batch_size, 1, 28, 28).
            detach: If True, detach activations between layers (local mode).
                    If False, allow full gradient flow (backprop mode).

        Returns:
            Logits tensor (batch_size, n_classes).
        """
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x, detach=detach)
        return x

    def forward_with_states(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        global_loss: float | None = None,
    ) -> list[LayerState]:
        """Forward pass collecting LayerState for each layer.

        Used by local learning rules to access pre/post activations.
        Always detaches between layers (local learning mode).

        Args:
            x: Input tensor (batch_size, 784).
            labels: Optional batch labels.
            global_loss: Optional current loss value.

        Returns:
            List of LayerState objects, one per layer.
        """
        x = x.view(x.size(0), -1)
        states: list[LayerState] = []

        for i, layer in enumerate(self.layers):
            pre = x
            x = layer(x, detach=True)

            states.append(
                LayerState(
                    pre_activation=pre,
                    post_activation=x,
                    weights=layer.weight.data,
                    bias=layer.bias,
                    layer_index=i,
                    labels=labels,
                    global_loss=global_loss,
                )
            )

        return states

    def get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return post-activation outputs for each layer (detached).

        Args:
            x: Input tensor (batch_size, 784).

        Returns:
            List of activation tensors, one per layer.
        """
        x = x.view(x.size(0), -1)
        activations: list[torch.Tensor] = []

        for layer in self.layers:
            x = layer(x, detach=True)
            activations.append(x)

        return activations
