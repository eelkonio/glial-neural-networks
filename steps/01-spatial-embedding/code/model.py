"""Baseline MLP for MNIST classification.

Architecture: 784 → 256 (ReLU) → 256 (ReLU) → 10
Total weight parameters (excluding biases): 203,264
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class WeightInfo:
    """Metadata for a single weight parameter."""

    layer_idx: int
    source_neuron: int
    target_neuron: int
    flat_idx: int


class BaselineMLP(nn.Module):
    """2-hidden-layer MLP for classification.

    Default architecture: 784 → 256 (ReLU) → 256 (ReLU) → 10
    Configurable input size and number of classes for different tasks.

    Provides utility methods for accessing weight metadata, flat weight
    tensors, and flat gradient tensors needed by embedding strategies.
    """

    def __init__(self, input_size: int = 784, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @property
    def weight_layers(self) -> list[nn.Linear]:
        """Return the linear layers in order."""
        return [self.fc1, self.fc2, self.fc3]

    def get_weight_count(self) -> int:
        """Total number of weight parameters (excluding biases)."""
        return sum(layer.weight.numel() for layer in self.weight_layers)

    def get_weight_metadata(self) -> list[WeightInfo]:
        """Return metadata for each weight: layer, source neuron, target neuron.

        Weights are ordered: fc1 weights (row-major), fc2 weights, fc3 weights.
        For nn.Linear, weight shape is (out_features, in_features), so
        weight[target, source] is the connection from source to target.
        """
        metadata = []
        flat_idx = 0
        for layer_idx, layer in enumerate(self.weight_layers):
            out_features, in_features = layer.weight.shape
            for target in range(out_features):
                for source in range(in_features):
                    metadata.append(
                        WeightInfo(
                            layer_idx=layer_idx,
                            source_neuron=source,
                            target_neuron=target,
                            flat_idx=flat_idx,
                        )
                    )
                    flat_idx += 1
        return metadata

    def get_flat_weights(self) -> torch.Tensor:
        """Return all weights as a flat 1D tensor (detached copy)."""
        return torch.cat(
            [layer.weight.detach().flatten() for layer in self.weight_layers]
        )

    def get_flat_gradients(self) -> torch.Tensor:
        """Return all weight gradients as a flat 1D tensor after backward().

        Raises RuntimeError if backward() has not been called.
        """
        grads = []
        for layer in self.weight_layers:
            if layer.weight.grad is None:
                raise RuntimeError(
                    f"No gradient for {layer}. Call backward() first."
                )
            grads.append(layer.weight.grad.detach().flatten())
        return torch.cat(grads)

    def get_layer_info(self) -> list[tuple[int, int, int]]:
        """Return (layer_idx, in_features, out_features) for each layer."""
        return [
            (i, layer.in_features, layer.out_features)
            for i, layer in enumerate(self.weight_layers)
        ]


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
