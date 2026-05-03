"""MNIST data loading utilities."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str | Path | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Load MNIST train and test datasets.

    Args:
        batch_size: Batch size for data loaders.
        data_dir: Directory to store/load MNIST data.
            Defaults to steps/01-spatial-embedding/data/mnist.
        num_workers: Number of data loading workers.

    Returns:
        (train_loader, test_loader) tuple.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "mnist"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to [0, 1] range
        ]
    )

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
