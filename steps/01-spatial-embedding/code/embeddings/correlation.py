"""Correlation-based embedding strategy.

Embeds weights so that those with correlated activations are close.

Computes pairwise activation correlations by running data through the
network, then uses MDS to embed the correlation distance matrix into 3D.

Due to O(N²) cost of full pairwise correlation on ~200K weights,
operates on a subsampled set and interpolates remaining positions.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import MDS


class CorrelationEmbedding:
    """Embeds weights so that those with correlated activations are close.

    Computes pairwise activation correlations by running data through the
    network, then uses MDS to embed the correlation distance matrix into 3D.

    Due to O(N²) cost of full pairwise correlation on ~200K weights,
    operates on a subsampled set and interpolates remaining positions.
    """

    def __init__(self, n_batches: int = 10, subsample_size: int = 5000):
        """Initialize correlation embedding.

        Args:
            n_batches: Number of data batches to use for correlation estimation.
            subsample_size: Number of weights to subsample for MDS (tractability).
        """
        self.n_batches = n_batches
        self.subsample_size = subsample_size

    @property
    def name(self) -> str:
        return "correlation"

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Compute correlation-based spatial positions for all weights.

        Requires data_loader in kwargs.

        Algorithm:
        1. Run data through the network, collecting per-weight gradient signals
        2. Subsample weights for tractability
        3. Compute pairwise correlation matrix on subsampled weights
        4. Convert correlations to distance matrix: d = 1 - |correlation|
        5. Use MDS to embed into 3D
        6. Interpolate remaining weight positions from subsampled set
        7. Normalize to [0, 1]

        Args:
            model: Neural network with get_weight_count() and weight_layers.
            **kwargs: Must include 'data_loader'.

        Returns:
            ndarray of shape (N_weights, 3) with coordinates in [0, 1].
        """
        data_loader = kwargs.get("data_loader")
        if data_loader is None:
            raise ValueError(
                "CorrelationEmbedding requires 'data_loader' in kwargs"
            )

        n_weights = model.get_weight_count()

        # Collect gradient signals for all weights across batches
        gradient_signals = self._collect_gradient_signals(model, data_loader)

        # Subsample weights for MDS
        rng = np.random.RandomState(42)
        subsample_size = min(self.subsample_size, n_weights)
        subsample_indices = rng.choice(
            n_weights, size=subsample_size, replace=False
        )
        subsample_indices.sort()

        # Compute correlation matrix for subsampled weights
        sub_signals = gradient_signals[subsample_indices]  # (subsample, n_batches)
        corr_matrix = self._compute_correlation_matrix(sub_signals)

        # Convert to distance matrix: d = 1 - |correlation|
        distance_matrix = 1.0 - np.abs(corr_matrix)
        # Ensure symmetry and zero diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(distance_matrix, 0.0)

        # MDS embedding of subsampled weights into 3D
        mds = MDS(
            n_components=3,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress="auto",
            max_iter=300,
            n_init=1,
        )
        sub_positions = mds.fit_transform(distance_matrix)

        # Interpolate remaining weight positions using nearest-neighbor
        # in correlation space
        all_positions = self._interpolate_positions(
            gradient_signals, subsample_indices, sub_positions, n_weights, rng
        )

        # Normalize to [0, 1]
        all_positions = self._normalize(all_positions)
        return all_positions

    def _collect_gradient_signals(
        self, model: nn.Module, data_loader
    ) -> np.ndarray:
        """Collect per-weight gradient magnitudes across batches.

        For each batch, computes the loss and backpropagates, then records
        the gradient magnitude for each weight. This gives a signal vector
        per weight that captures when/how each weight responds to data.

        Returns:
            ndarray of shape (N_weights, n_batches) with gradient signals.
        """
        model.eval()
        n_weights = model.get_weight_count()
        signals = np.zeros((n_weights, self.n_batches), dtype=np.float32)

        criterion = torch.nn.CrossEntropyLoss()

        batch_iter = iter(data_loader)
        for batch_idx in range(self.n_batches):
            try:
                inputs, targets = next(batch_iter)
            except StopIteration:
                # Restart iterator if we run out of batches
                batch_iter = iter(data_loader)
                inputs, targets = next(batch_iter)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Collect flat gradients
            grads = model.get_flat_gradients()
            signals[:, batch_idx] = grads.cpu().numpy()

        return signals

    def _compute_correlation_matrix(self, signals: np.ndarray) -> np.ndarray:
        """Compute pairwise Pearson correlation matrix.

        Args:
            signals: (n_weights, n_batches) array of gradient signals.

        Returns:
            (n_weights, n_weights) correlation matrix.
        """
        # Center the signals (subtract mean per weight)
        centered = signals - signals.mean(axis=1, keepdims=True)

        # Compute norms
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)

        # Normalized signals
        normalized = centered / norms

        # Correlation matrix = dot product of normalized signals
        corr_matrix = normalized @ normalized.T

        # Clip to [-1, 1] for numerical safety
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        return corr_matrix

    def _interpolate_positions(
        self,
        gradient_signals: np.ndarray,
        subsample_indices: np.ndarray,
        sub_positions: np.ndarray,
        n_weights: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Interpolate positions for non-subsampled weights.

        Uses nearest-neighbor in correlation space: for each non-subsampled
        weight, find the most correlated subsampled weight and assign its
        position (with small jitter to avoid exact overlaps).

        Args:
            gradient_signals: (N_weights, n_batches) full gradient signals.
            subsample_indices: Indices of subsampled weights.
            sub_positions: (subsample_size, 3) MDS positions.
            n_weights: Total number of weights.
            rng: Random state for jitter.

        Returns:
            (N_weights, 3) positions for all weights.
        """
        all_positions = np.zeros((n_weights, 3), dtype=np.float64)

        # Place subsampled weights at their MDS positions
        all_positions[subsample_indices] = sub_positions

        # For non-subsampled weights, find nearest subsampled neighbor
        # in correlation space
        sub_signals = gradient_signals[subsample_indices]

        # Precompute normalized subsampled signals
        sub_centered = sub_signals - sub_signals.mean(axis=1, keepdims=True)
        sub_norms = np.linalg.norm(sub_centered, axis=1, keepdims=True)
        sub_norms = np.maximum(sub_norms, 1e-10)
        sub_normalized = sub_centered / sub_norms

        # Create mask for non-subsampled indices
        is_subsampled = np.zeros(n_weights, dtype=bool)
        is_subsampled[subsample_indices] = True

        # Process non-subsampled weights in chunks for memory efficiency
        non_sub_indices = np.where(~is_subsampled)[0]
        chunk_size = 5000

        for chunk_start in range(0, len(non_sub_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(non_sub_indices))
            chunk_indices = non_sub_indices[chunk_start:chunk_end]

            # Get signals for this chunk
            chunk_signals = gradient_signals[chunk_indices]

            # Center and normalize
            chunk_centered = chunk_signals - chunk_signals.mean(
                axis=1, keepdims=True
            )
            chunk_norms = np.linalg.norm(chunk_centered, axis=1, keepdims=True)
            chunk_norms = np.maximum(chunk_norms, 1e-10)
            chunk_normalized = chunk_centered / chunk_norms

            # Compute correlations with subsampled weights
            # Shape: (chunk_size, subsample_size)
            correlations = chunk_normalized @ sub_normalized.T

            # Find nearest neighbor (highest absolute correlation)
            nearest_sub_idx = np.argmax(np.abs(correlations), axis=1)

            # Assign positions with small jitter
            jitter = rng.randn(len(chunk_indices), 3) * 0.001
            all_positions[chunk_indices] = (
                sub_positions[nearest_sub_idx] + jitter
            )

        return all_positions

    def _normalize(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to [0, 1] range per dimension."""
        for dim in range(3):
            col = positions[:, dim]
            rng = col.max() - col.min()
            if rng > 0:
                positions[:, dim] = (col - col.min()) / rng
            else:
                positions[:, dim] = 0.5
        return positions
