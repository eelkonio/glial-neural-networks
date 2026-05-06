"""Astrocyte domain assignment.

Partitions output neurons per layer into non-overlapping astrocyte domains.
Each domain governs a contiguous region and receives the same gating signal.

Modes:
- "spatial": Order neurons by spectral ordering (first eigenvector of the
  layer's weight matrix), then partition into contiguous groups.
- "random": Random assignment for ablation studies.
- Fallback: Contiguous index partitioning if no spatial info available.
"""

import math
import warnings
from typing import Optional

import torch
import numpy as np

from code.domains.config import DomainConfig


class DomainAssignment:
    """Assigns output neurons to astrocyte domains.

    Immutable after initialization — domain assignments never change.

    Args:
        layer_sizes: List of (in_features, out_features) per layer.
        config: DomainConfig with assignment parameters.
        weight_matrices: Optional list of weight tensors per layer,
            used for spectral ordering in "spatial" mode.
    """

    def __init__(
        self,
        layer_sizes: list[tuple[int, int]],
        config: DomainConfig | None = None,
        weight_matrices: Optional[list[torch.Tensor]] = None,
    ):
        self.config = config or DomainConfig()
        self._layer_sizes = layer_sizes
        self._domain_size = self.config.domain_size

        # Compute assignments for each layer
        self._assignments: list[list[list[int]]] = []  # [layer][domain] -> [neuron_indices]
        self._neuron_to_domain: list[torch.Tensor] = []  # [layer] -> (out_features,)
        self._domain_distances: list[torch.Tensor] = []  # [layer] -> (n_domains, n_domains)

        for layer_idx, (in_feat, out_feat) in enumerate(layer_sizes):
            weight = weight_matrices[layer_idx] if weight_matrices else None
            domains = self._assign_layer(out_feat, weight, layer_idx)
            self._assignments.append(domains)

            # Build neuron-to-domain mapping
            n2d = torch.zeros(out_feat, dtype=torch.long)
            for domain_idx, indices in enumerate(domains):
                for idx in indices:
                    n2d[idx] = domain_idx
            self._neuron_to_domain.append(n2d)

            # Compute domain distances (based on mean neuron index position)
            n_domains = len(domains)
            distances = torch.zeros(n_domains, n_domains)
            centers = []
            for indices in domains:
                centers.append(np.mean(indices) if indices else 0.0)
            for i in range(n_domains):
                for j in range(n_domains):
                    distances[i, j] = abs(centers[i] - centers[j])
            self._domain_distances.append(distances)

    def _assign_layer(
        self,
        out_features: int,
        weight: Optional[torch.Tensor],
        layer_idx: int,
    ) -> list[list[int]]:
        """Assign neurons in one layer to domains.

        Args:
            out_features: Number of output neurons in this layer.
            weight: Weight matrix (out_features, in_features) if available.
            layer_idx: Index of this layer.

        Returns:
            List of lists of neuron indices per domain.
        """
        if self.config.mode == "random":
            return self._random_assignment(out_features)
        elif self.config.mode == "spatial":
            return self._spatial_assignment(out_features, weight, layer_idx)
        else:
            return self._contiguous_assignment(out_features)

    def _spatial_assignment(
        self,
        out_features: int,
        weight: Optional[torch.Tensor],
        layer_idx: int,
    ) -> list[list[int]]:
        """Spatial assignment using spectral ordering.

        Uses the first eigenvector of the weight matrix's correlation
        structure to order neurons, then partitions into contiguous groups.
        Falls back to contiguous if weight matrix not available.
        """
        if weight is None:
            warnings.warn(
                f"No weight matrix for layer {layer_idx}, "
                f"falling back to contiguous partitioning.",
                stacklevel=2,
            )
            return self._contiguous_assignment(out_features)

        try:
            # Compute correlation matrix of output neurons
            # weight shape: (out_features, in_features)
            w = weight.detach().cpu().float()

            # Use SVD of weight matrix to get spectral ordering
            # The first left singular vector orders output neurons
            # by their principal direction of variation
            if out_features <= 1:
                return self._contiguous_assignment(out_features)

            # Compute W @ W^T to get output neuron similarity
            gram = w @ w.T  # (out_features, out_features)

            # Get first eigenvector for ordering
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            # Last eigenvector (largest eigenvalue) gives principal ordering
            ordering_vector = eigenvectors[:, -1].numpy()

            # Sort neurons by their position in the spectral ordering
            sorted_indices = np.argsort(ordering_vector).tolist()

            # Partition sorted indices into domains
            return self._partition_indices(sorted_indices, out_features)

        except Exception:
            warnings.warn(
                f"Spectral ordering failed for layer {layer_idx}, "
                f"falling back to contiguous partitioning.",
                stacklevel=2,
            )
            return self._contiguous_assignment(out_features)

    def _random_assignment(self, out_features: int) -> list[list[int]]:
        """Random assignment for ablation."""
        rng = np.random.RandomState(self.config.seed)
        indices = list(range(out_features))
        rng.shuffle(indices)
        return self._partition_indices(indices, out_features)

    def _contiguous_assignment(self, out_features: int) -> list[list[int]]:
        """Contiguous index partitioning (fallback)."""
        indices = list(range(out_features))
        return self._partition_indices(indices, out_features)

    def _partition_indices(
        self, indices: list[int], out_features: int
    ) -> list[list[int]]:
        """Partition a list of indices into domains of domain_size."""
        n_domains = math.ceil(out_features / self._domain_size)
        domains = []
        for i in range(n_domains):
            start = i * self._domain_size
            end = min(start + self._domain_size, len(indices))
            domains.append(indices[start:end])
        return domains

    @property
    def n_domains_per_layer(self) -> list[int]:
        """Number of domains in each layer."""
        return [len(domains) for domains in self._assignments]

    @property
    def total_domains(self) -> int:
        """Total domains across all layers."""
        return sum(self.n_domains_per_layer)

    def get_domain_indices(self, layer_index: int) -> list[list[int]]:
        """Return list of output neuron indices per domain for a layer.

        Args:
            layer_index: Which layer (0-indexed).

        Returns:
            List of lists, where each inner list contains the output neuron
            indices belonging to that domain.
        """
        return self._assignments[layer_index]

    def get_domain_distances(self, layer_index: int) -> torch.Tensor:
        """Pairwise distances between domain centers for a layer.

        Args:
            layer_index: Which layer (0-indexed).

        Returns:
            Distance matrix (n_domains, n_domains).
        """
        return self._domain_distances[layer_index]

    def get_neuron_to_domain(self, layer_index: int) -> torch.Tensor:
        """Mapping from output neuron index to domain index.

        Args:
            layer_index: Which layer (0-indexed).

        Returns:
            Tensor of shape (out_features,) with domain index per neuron.
        """
        return self._neuron_to_domain[layer_index]
