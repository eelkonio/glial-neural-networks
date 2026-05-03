"""Spatially-structured benchmark task with topographic sensor array.

A 2D grid of sensors (16x16 = 256 inputs) where each sensor reads a local
patch of a larger signal field. Adjacent sensors have overlapping receptive
fields, creating natural spatial correlations. The classification task
requires integrating information across the sensor field in a spatially
structured way.

Unlike MNIST (permutation-invariant), this task has a known ground-truth
spatial structure that the embedding should discover.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class TopographicTask:
    """Benchmark task with inherent spatial structure."""

    def __init__(
        self,
        grid_size: int = 16,
        n_classes: int = 10,
        n_train: int = 50000,
        n_test: int = 10000,
        correlation_length: float = 3.0,
    ):
        self.grid_size = grid_size
        self.n_inputs = grid_size * grid_size
        self.n_classes = n_classes
        self.n_train = n_train
        self.n_test = n_test
        self.correlation_length = correlation_length

    def _make_spatial_kernel(self) -> np.ndarray:
        """Create a Gaussian spatial correlation kernel."""
        coords = np.arange(self.grid_size)
        xx, yy = np.meshgrid(coords, coords)
        positions = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Pairwise distance matrix between all sensor positions
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))

        # Gaussian kernel: nearby sensors are correlated
        kernel = np.exp(-0.5 * (dist / self.correlation_length) ** 2)
        return kernel

    def _generate_class_templates(
        self, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate spatially smooth class templates.

        Each class has a distinct spatial pattern on the sensor grid.
        Templates are smooth (spatially correlated) so that the task
        requires spatially structured processing.
        """
        kernel = self._make_spatial_kernel()
        # Cholesky decomposition for correlated sampling
        L = np.linalg.cholesky(kernel + 1e-6 * np.eye(self.n_inputs))

        templates = np.zeros((self.n_classes, self.n_inputs))
        for c in range(self.n_classes):
            # Generate spatially correlated random pattern
            z = rng.standard_normal(self.n_inputs)
            templates[c] = L @ z

        return templates

    def _generate_samples(
        self, n_samples: int, templates: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate samples by adding noise to class templates."""
        labels = rng.integers(0, self.n_classes, size=n_samples)
        noise_scale = 0.5

        kernel = self._make_spatial_kernel()
        L = np.linalg.cholesky(kernel + 1e-6 * np.eye(self.n_inputs))

        data = np.zeros((n_samples, self.n_inputs), dtype=np.float32)
        for i in range(n_samples):
            # Template + spatially correlated noise
            z = rng.standard_normal(self.n_inputs)
            noise = L @ z * noise_scale
            data[i] = templates[labels[i]] + noise

        # Normalize to [0, 1] range
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        return data, labels

    def generate_dataset(
        self, seed: int = 42, batch_size: int = 128
    ) -> tuple[DataLoader, DataLoader]:
        """Generate train and test data loaders.

        Returns:
            (train_loader, test_loader) with spatially-structured inputs.
        """
        rng = np.random.default_rng(seed)
        templates = self._generate_class_templates(rng)

        train_data, train_labels = self._generate_samples(
            self.n_train, templates, rng
        )
        test_data, test_labels = self._generate_samples(
            self.n_test, templates, rng
        )

        train_dataset = TensorDataset(
            torch.from_numpy(train_data),
            torch.from_numpy(train_labels).long(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(test_data),
            torch.from_numpy(test_labels).long(),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader

    def get_ground_truth_embedding(self) -> np.ndarray:
        """Return the known-correct spatial structure for this task.

        For the topographic task, the ground truth is that weights connecting
        adjacent sensors should be spatially close. We return the 2D grid
        positions of the input sensors, normalized to [0, 1] and extended
        to 3D (z=0 for input layer).

        Returns:
            (grid_size^2, 3) array of sensor positions.
        """
        coords = np.arange(self.grid_size, dtype=np.float64)
        xx, yy = np.meshgrid(coords, coords)
        positions_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)
        # Normalize to [0, 1]
        positions_2d = positions_2d / (self.grid_size - 1)
        # Extend to 3D with z=0
        positions_3d = np.zeros((self.n_inputs, 3))
        positions_3d[:, 0] = positions_2d[:, 0]
        positions_3d[:, 1] = positions_2d[:, 1]
        return positions_3d.astype(np.float32)
