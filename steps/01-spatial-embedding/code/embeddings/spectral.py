"""Spectral embedding strategy.

Topology-preserving embedding from graph Laplacian eigenvectors.

Connectivity is defined at the neuron level: two neurons are connected
if there's a weight between them. Edge weight = sum of absolute weight
magnitudes between those neurons. The graph Laplacian is built over this
neuron-level adjacency, then coordinates are assigned to weights based
on their source/target neuron positions (midpoint interpolation).

For a fully-connected MLP, direct weight-level adjacency is trivial
(all weights in adjacent layers connect). Instead, we build the graph
at the neuron level and interpolate to weights.
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import torch.nn as nn


class SpectralEmbedding:
    """Topology-preserving embedding from graph Laplacian eigenvectors."""

    @property
    def name(self) -> str:
        return "spectral"

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Compute spectral spatial positions for all weights.

        Algorithm:
        1. Build neuron-level adjacency from weight magnitudes
        2. Compute graph Laplacian
        3. Extract 3 smallest non-trivial eigenvectors
        4. Assign neuron positions from eigenvectors
        5. Assign weight positions as midpoint of source/target neurons
        6. Handle disconnected components with spatial offsets
        7. Normalize to [0, 1]

        Args:
            model: Neural network with get_weight_metadata(),
                   get_layer_info(), and weight_layers methods.

        Returns:
            ndarray of shape (N_weights, 3) with coordinates in [0, 1].
        """
        # Build neuron-level adjacency
        adjacency, neuron_offsets, total_neurons = self._build_neuron_adjacency(model)

        # Compute neuron positions from Laplacian eigenvectors
        neuron_positions = self._compute_neuron_positions(
            adjacency, total_neurons
        )

        # Assign weight positions as midpoint of source/target neuron positions
        metadata = model.get_weight_metadata()
        layer_info = model.get_layer_info()
        n_weights = len(metadata)
        positions = np.zeros((n_weights, 3), dtype=np.float64)

        for i, w in enumerate(metadata):
            # Get global neuron indices for source and target
            source_global = neuron_offsets[w.layer_idx] + w.source_neuron
            target_global = neuron_offsets[w.layer_idx + 1] + w.target_neuron

            # Weight position = midpoint of source and target neuron positions
            positions[i] = (
                neuron_positions[source_global] + neuron_positions[target_global]
            ) / 2.0

        # Normalize to [0, 1]
        positions = self._normalize(positions)
        return positions

    def _build_neuron_adjacency(
        self, model: nn.Module
    ) -> tuple[sparse.csr_matrix, list[int], int]:
        """Build adjacency matrix at neuron level from weight magnitudes.

        Neurons are indexed globally across all layers:
        - Layer 0 input neurons: 0..783
        - Layer 0 output / Layer 1 input neurons: 784..1039
        - Layer 1 output / Layer 2 input neurons: 1040..1295
        - Layer 2 output neurons: 1296..1305

        Two neurons are connected if there's a weight between them.
        Edge weight = sum of absolute weight magnitudes between those neurons.

        Returns:
            (adjacency_matrix, neuron_offsets, total_neurons)
            neuron_offsets[i] = global index of first neuron in layer i's input
        """
        layer_info = model.get_layer_info()

        # Compute neuron offsets
        # Each layer has input neurons; the final layer also has output neurons
        neuron_offsets = []
        offset = 0
        for layer_idx, in_feat, out_feat in layer_info:
            neuron_offsets.append(offset)
            offset += in_feat
        # Add the output neurons of the last layer
        neuron_offsets.append(offset)
        total_neurons = offset + layer_info[-1][2]  # + last layer's out_features

        # Build sparse adjacency
        rows = []
        cols = []
        data = []

        for layer_idx, layer in enumerate(model.weight_layers):
            weight_matrix = layer.weight.detach().cpu().numpy()
            out_features, in_features = weight_matrix.shape

            for target in range(out_features):
                target_global = neuron_offsets[layer_idx + 1] + target
                for source in range(in_features):
                    source_global = neuron_offsets[layer_idx] + source
                    edge_weight = abs(float(weight_matrix[target, source]))
                    if edge_weight > 0:
                        rows.append(source_global)
                        cols.append(target_global)
                        data.append(edge_weight)
                        # Symmetric
                        rows.append(target_global)
                        cols.append(source_global)
                        data.append(edge_weight)

        adjacency = sparse.csr_matrix(
            (data, (rows, cols)), shape=(total_neurons, total_neurons)
        )
        return adjacency, neuron_offsets, total_neurons

    def _compute_neuron_positions(
        self, adjacency: sparse.csr_matrix, total_neurons: int
    ) -> np.ndarray:
        """Compute neuron positions from Laplacian eigenvectors.

        Handles disconnected components by embedding each separately
        and offsetting spatially.

        Returns:
            ndarray of shape (total_neurons, 3) with neuron positions.
        """
        # Find connected components
        n_components, labels = sparse.csgraph.connected_components(
            adjacency, directed=False
        )

        neuron_positions = np.zeros((total_neurons, 3), dtype=np.float64)

        if n_components == 1:
            # Single connected component - standard spectral embedding
            neuron_positions = self._spectral_embed_component(
                adjacency, np.arange(total_neurons)
            )
        else:
            # Multiple components - embed each separately, offset spatially
            component_positions = []
            for comp_idx in range(n_components):
                node_indices = np.where(labels == comp_idx)[0]
                if len(node_indices) < 4:
                    # Too few nodes for spectral embedding, use simple layout
                    pos = np.zeros((len(node_indices), 3), dtype=np.float64)
                    for j, _ in enumerate(node_indices):
                        pos[j] = [
                            j / max(len(node_indices) - 1, 1),
                            0.5,
                            0.5,
                        ]
                    component_positions.append((node_indices, pos))
                else:
                    # Extract subgraph
                    sub_adjacency = adjacency[node_indices][:, node_indices]
                    pos = self._spectral_embed_component(
                        sub_adjacency, node_indices
                    )
                    component_positions.append((node_indices, pos))

            # Offset components along x-axis
            x_offset = 0.0
            for node_indices, pos in component_positions:
                # Normalize component positions to [0, 1] within component
                for dim in range(3):
                    col = pos[:, dim]
                    rng = col.max() - col.min()
                    if rng > 0:
                        pos[:, dim] = (col - col.min()) / rng
                    else:
                        pos[:, dim] = 0.5

                # Offset x by component index
                pos[:, 0] = pos[:, 0] / n_components + x_offset
                x_offset += 1.0 / n_components

                # Assign to global positions
                for j, node_idx in enumerate(node_indices):
                    neuron_positions[node_idx] = pos[j]

        return neuron_positions

    def _spectral_embed_component(
        self, adjacency: sparse.csr_matrix, node_indices: np.ndarray
    ) -> np.ndarray:
        """Compute spectral embedding for a single connected component.

        Uses the 3 smallest non-trivial eigenvectors of the graph Laplacian.

        Args:
            adjacency: Adjacency matrix for this component.
            node_indices: Global indices of nodes in this component.

        Returns:
            ndarray of shape (len(node_indices), 3) with positions.
        """
        n = adjacency.shape[0]

        # Compute graph Laplacian: L = D - A
        degree = np.array(adjacency.sum(axis=1)).flatten()
        D = sparse.diags(degree)
        laplacian = D - adjacency

        # Number of eigenvectors to compute (3 for 3D + 1 for trivial)
        n_eigvecs = min(4, n - 1)
        if n_eigvecs < 2:
            # Not enough nodes for meaningful spectral embedding
            pos = np.zeros((n, 3), dtype=np.float64)
            for i in range(n):
                pos[i] = [i / max(n - 1, 1), 0.5, 0.5]
            return pos

        # Compute smallest eigenvectors using shift-invert mode
        # sigma=0 finds eigenvalues closest to 0
        # Use a fixed random state for deterministic results
        try:
            rng = np.random.RandomState(42)
            v0 = rng.randn(n)
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                laplacian.astype(np.float64),
                k=n_eigvecs,
                which="SM",
                v0=v0,
            )
        except (scipy.sparse.linalg.ArpackNoConvergence, RuntimeError):
            # Fallback: use simple linear layout
            pos = np.zeros((n, 3), dtype=np.float64)
            for i in range(n):
                pos[i] = [i / max(n - 1, 1), 0.5, 0.5]
            return pos

        # Sort by eigenvalue (smallest first)
        sort_idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, sort_idx]

        # Skip the trivial eigenvector (constant, eigenvalue ≈ 0)
        # Use the next 3 non-trivial eigenvectors for 3D coordinates
        pos = np.zeros((n, 3), dtype=np.float64)
        available = min(3, n_eigvecs - 1)
        for dim in range(available):
            vec = eigenvectors[:, dim + 1]
            # Enforce deterministic sign: largest absolute element is positive
            max_abs_idx = np.argmax(np.abs(vec))
            if vec[max_abs_idx] < 0:
                vec = -vec
            pos[:, dim] = vec

        return pos

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
