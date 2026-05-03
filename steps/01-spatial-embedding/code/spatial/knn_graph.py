"""KNN graph construction using scipy.spatial.cKDTree.

Provides efficient neighbor lookups for spatial coupling and
quality measurement operations.
"""

import numpy as np
from scipy.spatial import cKDTree


class KNNGraph:
    """K-nearest-neighbor graph over spatial positions using cKDTree.

    Provides efficient neighbor lookups for spatial coupling and
    quality measurement operations.
    """

    def __init__(self, positions: np.ndarray, k: int = 10):
        """Build KNN graph using scipy.spatial.cKDTree.

        Args:
            positions: (N, 3) array of spatial coordinates.
            k: Number of nearest neighbors per node. Clamped to N-1 if k >= N.
        """
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (N, 3), got {positions.shape}"
            )

        n = positions.shape[0]
        # Clamp k to N-1 if k >= N
        self._k = min(k, n - 1) if n > 1 else 0

        self._positions = positions.copy()
        self._tree = cKDTree(positions)

        if self._k > 0:
            # Query k+1 neighbors (includes self), then exclude self
            distances, indices = self._tree.query(positions, k=self._k + 1)
            # Exclude the first column (self, distance=0)
            self._neighbor_indices = indices[:, 1:].astype(np.intp)
            self._neighbor_distances = distances[:, 1:]
        else:
            # Edge case: single point or k=0
            self._neighbor_indices = np.empty((n, 0), dtype=np.intp)
            self._neighbor_distances = np.empty((n, 0), dtype=np.float64)

    @property
    def k(self) -> int:
        """Effective k (may be clamped from requested value)."""
        return self._k

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._positions.shape[0]

    @property
    def neighbor_indices(self) -> np.ndarray:
        """(N, k) array of neighbor indices for each node."""
        return self._neighbor_indices

    @property
    def neighbor_distances(self) -> np.ndarray:
        """(N, k) array of distances to each neighbor."""
        return self._neighbor_distances

    def get_neighbors(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances) for neighbors of node idx.

        Args:
            idx: Index of the node to query.

        Returns:
            Tuple of (neighbor_indices, neighbor_distances) arrays of shape (k,).
        """
        return self._neighbor_indices[idx], self._neighbor_distances[idx]
