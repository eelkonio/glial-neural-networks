"""Linear embedding strategy.

Maps weight indices directly to normalized 3D coordinates:
  x = layer_idx / total_layers
  y = source_neuron / max_neurons_in_layer
  z = target_neuron / max_neurons_in_layer

For nn.Linear, weight shape is (out_features, in_features), so
weight[target, source]. We iterate target (out) then source (in)
to match get_weight_metadata() ordering.
"""

import numpy as np
import torch.nn as nn


class LinearEmbedding:
    """Maps weight indices directly to normalized 3D coordinates."""

    @property
    def name(self) -> str:
        return "linear"

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Compute linear spatial positions for all weights.

        Args:
            model: Neural network with get_weight_metadata() and
                   get_layer_info() methods.

        Returns:
            ndarray of shape (N_weights, 3) with coordinates in [0, 1].
        """
        layer_info = model.get_layer_info()
        total_layers = len(layer_info)

        # Find max neurons across all layers (both in and out features)
        max_neurons = max(
            max(in_feat, out_feat) for _, in_feat, out_feat in layer_info
        )

        metadata = model.get_weight_metadata()
        n_weights = len(metadata)
        positions = np.zeros((n_weights, 3), dtype=np.float64)

        for i, w in enumerate(metadata):
            positions[i, 0] = w.layer_idx / total_layers
            positions[i, 1] = w.source_neuron / max_neurons
            positions[i, 2] = w.target_neuron / max_neurons

        return positions
