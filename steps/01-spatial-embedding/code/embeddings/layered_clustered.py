"""Layered-clustered embedding strategy.

Hybrid: layer depth on x-axis, spectral clustering within layers on y/z.

x-coordinate = layer_idx / total_layers (identical for all weights in same layer)
y, z coordinates: spectral embedding within each layer's weight matrix
  - Uses SVD of the weight matrix to find structure within each layer
  - Left singular vectors give target neuron coordinates
  - Right singular vectors give source neuron coordinates
  - Weight position y/z = interpolation of source and target neuron y/z

This preserves the natural layer structure while allowing within-layer
organization based on connectivity patterns.
"""

import numpy as np
import torch.nn as nn


class LayeredClusteredEmbedding:
    """Hybrid: layer depth on x-axis, spectral clustering within layers."""

    @property
    def name(self) -> str:
        return "layered_clustered"

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Compute layered-clustered spatial positions for all weights.

        Algorithm:
        1. x-coordinate = layer_idx / total_layers
        2. For each layer, compute SVD of weight matrix
        3. Use top singular vectors to assign y/z to source and target neurons
        4. Weight y/z = midpoint of source and target neuron y/z
        5. Normalize y, z to [0, 1] per layer

        Args:
            model: Neural network with get_weight_metadata(),
                   get_layer_info(), and weight_layers methods.

        Returns:
            ndarray of shape (N_weights, 3) with coordinates in [0, 1].
        """
        layer_info = model.get_layer_info()
        total_layers = len(layer_info)
        metadata = model.get_weight_metadata()
        n_weights = len(metadata)
        positions = np.zeros((n_weights, 3), dtype=np.float64)

        # Precompute per-layer spectral coordinates for neurons
        layer_source_coords = {}  # layer_idx -> (in_features,) y coords
        layer_target_coords = {}  # layer_idx -> (out_features,) z coords

        for layer_idx, layer in enumerate(model.weight_layers):
            weight_matrix = layer.weight.detach().cpu().numpy()
            out_features, in_features = weight_matrix.shape

            # Compute SVD for within-layer structure
            source_y, source_z, target_y, target_z = (
                self._compute_layer_spectral_coords(weight_matrix)
            )

            layer_source_coords[layer_idx] = (source_y, source_z)
            layer_target_coords[layer_idx] = (target_y, target_z)

        # Assign positions to each weight
        for i, w in enumerate(metadata):
            # x = layer depth (identical for all weights in same layer)
            positions[i, 0] = w.layer_idx / total_layers

            # y, z from spectral coordinates of source and target neurons
            src_y, src_z = layer_source_coords[w.layer_idx]
            tgt_y, tgt_z = layer_target_coords[w.layer_idx]

            # Weight y = midpoint of source and target y-coordinates
            positions[i, 1] = (src_y[w.source_neuron] + tgt_y[w.target_neuron]) / 2.0
            # Weight z = midpoint of source and target z-coordinates
            positions[i, 2] = (src_z[w.source_neuron] + tgt_z[w.target_neuron]) / 2.0

        # Normalize y and z to [0, 1] globally
        for dim in [1, 2]:
            col = positions[:, dim]
            rng = col.max() - col.min()
            if rng > 0:
                positions[:, dim] = (col - col.min()) / rng
            else:
                positions[:, dim] = 0.5

        return positions

    def _compute_layer_spectral_coords(
        self, weight_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectral coordinates for neurons within a layer.

        Uses SVD of the weight matrix:
        - Right singular vectors (V) give source neuron structure
        - Left singular vectors (U) give target neuron structure

        We use the first two non-trivial singular vectors for y and z.

        Args:
            weight_matrix: Shape (out_features, in_features).

        Returns:
            (source_y, source_z, target_y, target_z) each normalized to [0, 1].
        """
        out_features, in_features = weight_matrix.shape

        # Compute truncated SVD (only need first few components)
        n_components = min(3, min(out_features, in_features))

        try:
            U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback: uniform spacing
            source_y = np.linspace(0, 1, in_features)
            source_z = np.linspace(0, 1, in_features)
            target_y = np.linspace(0, 1, out_features)
            target_z = np.linspace(0, 1, out_features)
            return source_y, source_z, target_y, target_z

        # Source neuron coordinates from right singular vectors (V = Vt.T)
        # Use 1st and 2nd singular vectors for y and z
        if n_components >= 2:
            source_y = Vt[0, :]  # First right singular vector
            source_z = Vt[1, :]  # Second right singular vector
        else:
            source_y = Vt[0, :]
            source_z = np.linspace(0, 1, in_features)

        # Target neuron coordinates from left singular vectors (U)
        if n_components >= 2:
            target_y = U[:, 0]  # First left singular vector
            target_z = U[:, 1]  # Second left singular vector
        else:
            target_y = U[:, 0]
            target_z = np.linspace(0, 1, out_features)

        # Normalize each to [0, 1]
        source_y = self._normalize_1d(source_y)
        source_z = self._normalize_1d(source_z)
        target_y = self._normalize_1d(target_y)
        target_z = self._normalize_1d(target_z)

        return source_y, source_z, target_y, target_z

    def _normalize_1d(self, arr: np.ndarray) -> np.ndarray:
        """Normalize a 1D array to [0, 1]."""
        rng = arr.max() - arr.min()
        if rng > 0:
            return (arr - arr.min()) / rng
        return np.full_like(arr, 0.5)
