"""Spatial coherence metric (PCA-based).

Measures whether spatially close weights develop similar PCA projections.
This tests the MECHANISM directly: if spatial coupling produces spatially
organized representations, spatially close weights should have similar
projections onto the top principal components of the weight matrix.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


class SpatialCoherence:
    """Measures whether spatially close weights develop similar PCA projections.

    Distinguishes the strong claim (spatial structure matters) from the weak
    claim (spatial smoothing is just regularization).
    """

    def __init__(self, n_components: int = 10, max_pairs: int = 100_000):
        """
        Args:
            n_components: Number of PCA components to compute.
            max_pairs: Maximum number of pairs to sample for correlation.
        """
        self._n_components = n_components
        self._max_pairs = max_pairs

    @property
    def n_components(self) -> int:
        """Number of PCA components used."""
        return self._n_components

    @property
    def max_pairs(self) -> int:
        """Maximum pairs for sampling."""
        return self._max_pairs

    def compute_coherence(
        self, weights: np.ndarray, positions: np.ndarray
    ) -> float:
        """Compute spatial coherence score.

        1. Compute top-k PCA components of weight matrix
        2. For sampled pairs, compute spatial distance and PC similarity
           (dot product of PC projections)
        3. Return Pearson correlation between distances and similarities

        High positive correlation = spatially organized structure.

        Args:
            weights: (N,) flat array of weight values.
            positions: (N, 3) array of spatial positions.

        Returns:
            Pearson correlation between spatial distances and PC similarities.
            Returns 0.0 for degenerate cases.
        """
        weights = np.asarray(weights, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)

        n = len(weights)

        if n < self._n_components + 1:
            return 0.0

        # Reshape weights into a matrix for PCA
        # We treat each weight as a 1D feature; to get meaningful PCA,
        # we need a matrix. We use the positions as features alongside weights.
        # Actually, PCA on the weight matrix means we need multiple observations.
        # For a single weight vector, we compute PCA on the (N, 3) position-weight
        # combined space, or we use the weight values directly.
        #
        # The design says: "Compute top-k PCA components of weight matrix"
        # For a flat weight vector, we reshape it into a 2D matrix based on
        # the network structure. But since we have a flat vector, we'll
        # use a sliding window approach to create a feature matrix.
        #
        # Simpler interpretation: project each weight onto PCA components
        # of the weight vector reshaped as a matrix. Since we have (N,) weights
        # and (N, 3) positions, we compute PCA on a feature matrix where
        # each row is a weight's local context (its value + neighbors' values).
        #
        # Most natural interpretation for coherence: use PCA on the weight
        # values viewed as a matrix (e.g., layer weight matrices stacked).
        # But for generality with flat weights, we'll create features from
        # local neighborhoods.
        #
        # Simplest correct approach: treat the weight vector as a 1D signal,
        # create overlapping windows, and compute PCA on those windows.
        # OR: just use the weight values directly and compute similarity
        # as the absolute difference (simpler, still tests spatial organization).
        #
        # Following the design more literally: PCA on the weight matrix.
        # We'll reshape the flat weights into chunks and compute PCA.

        # Use a chunk-based approach: divide weights into chunks, compute PCA
        # on the chunk matrix, then project each weight onto PC space.
        n_components = min(self._n_components, n - 1)

        # Create a feature matrix: each weight gets a feature vector
        # based on its value. For meaningful PCA, we need multiple features.
        # Use position-augmented features: [weight_value, x, y, z]
        # Then PCA captures the joint structure.
        #
        # Actually, the cleanest interpretation: the "weight matrix" is the
        # collection of weight values. PCA finds the principal directions
        # of variation. We project each weight onto these directions and
        # measure if spatially close weights have similar projections.
        #
        # For a 1D weight vector, we need to create a 2D matrix.
        # Standard approach: use a sliding window of size w to create
        # an (N-w+1, w) matrix, then project all N weights.
        #
        # Even simpler: just use the raw weight values and compute
        # pairwise similarity as |w_i - w_j| (weight distance).
        # Then correlate with spatial distance.
        #
        # But the design specifically says PCA. Let's use a window approach.

        # Window-based feature extraction
        window_size = min(n_components * 2, n)
        if window_size < 2:
            return 0.0

        # Create feature matrix using circular padding
        padded = np.pad(weights, (window_size // 2, window_size // 2), mode='wrap')
        feature_matrix = np.lib.stride_tricks.sliding_window_view(
            padded, window_size
        )[:n]  # (N, window_size)

        # Compute PCA
        pca = PCA(n_components=n_components)
        projections = pca.fit_transform(feature_matrix)  # (N, n_components)

        # Sample pairs
        total_pairs = n * (n - 1) // 2
        rng = np.random.default_rng(42)

        if total_pairs > self._max_pairs:
            # Random subsampling
            idx_i = rng.integers(0, n, size=self._max_pairs)
            idx_j = rng.integers(0, n, size=self._max_pairs)
            mask = idx_i == idx_j
            while mask.any():
                idx_j[mask] = rng.integers(0, n, size=mask.sum())
                mask = idx_i == idx_j
        else:
            idx_i, idx_j = np.triu_indices(n, k=1)

        # Compute spatial distances
        pos_diff = positions[idx_i] - positions[idx_j]
        spatial_distances = np.sqrt(np.sum(pos_diff**2, axis=1))

        # Compute PC similarity (dot product of projections)
        proj_i = projections[idx_i]  # (n_pairs, n_components)
        proj_j = projections[idx_j]  # (n_pairs, n_components)
        pc_similarities = np.sum(proj_i * proj_j, axis=1)

        # Handle degenerate cases
        if np.std(spatial_distances) < 1e-12 or np.std(pc_similarities) < 1e-12:
            return 0.0

        # Pearson correlation
        corr, _ = pearsonr(spatial_distances, pc_similarities)

        return float(corr)
