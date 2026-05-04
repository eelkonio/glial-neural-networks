"""FashionMNIST data loading and forward-forward data adaptation.

Provides standard data loaders and the ForwardForwardDataAdapter that
yields (x_pos, x_neg, labels) tuples for the forward-forward algorithm.
"""

from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_fashion_mnist_loaders(
    batch_size: int = 128,
    data_dir: str | Path | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Load FashionMNIST train and test datasets.

    Args:
        batch_size: Batch size for data loaders.
        data_dir: Directory to store/load data.
            Defaults to steps/12-local-learning-rules/data/fashionmnist.
        num_workers: Number of data loading workers.

    Returns:
        (train_loader, test_loader) tuple.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "fashionmnist"

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
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


def embed_label(
    x: torch.Tensor, labels: torch.Tensor, n_classes: int = 10
) -> torch.Tensor:
    """Embed labels into the first n_classes pixels of the flattened input.

    Sets the pixel at the label index to 1.0 (max intensity).
    Other label pixels are set to 0.0 to ensure a clean one-hot encoding.

    Args:
        x: Input tensor of shape (batch_size, 784) — flattened images.
        labels: Integer labels of shape (batch_size,).
        n_classes: Number of classes (default 10).

    Returns:
        Modified input with label embedded in first n_classes pixels.
    """
    x_embedded = x.clone()
    # Zero out the label region first
    x_embedded[:, :n_classes] = 0.0
    # Set the correct label pixel to 1.0
    x_embedded[torch.arange(x.size(0)), labels] = 1.0
    return x_embedded


def generate_negative(
    x: torch.Tensor, labels: torch.Tensor, n_classes: int = 10
) -> torch.Tensor:
    """Generate negative samples by embedding a random incorrect label.

    The image pixels (beyond the label region) remain unchanged.
    Only the label embedding is replaced with a random wrong label.

    Args:
        x: Input tensor of shape (batch_size, 784) — flattened images.
        labels: Correct integer labels of shape (batch_size,).
        n_classes: Number of classes (default 10).

    Returns:
        Negative sample with incorrect label embedded.
    """
    batch_size = x.size(0)
    # Generate random labels that differ from the correct ones
    random_labels = torch.randint(0, n_classes - 1, (batch_size,), device=x.device)
    # Shift labels >= correct label up by 1 to avoid matching
    random_labels = torch.where(
        random_labels >= labels, random_labels + 1, random_labels
    )
    return embed_label(x, random_labels, n_classes)


class ForwardForwardDataAdapter:
    """Wraps a FashionMNIST DataLoader to yield (x_pos, x_neg, labels) tuples.

    Positive samples have the correct label embedded in the first 10 pixels.
    Negative samples have a random incorrect label embedded.

    Usage:
        train_loader, _ = get_fashion_mnist_loaders()
        ff_adapter = ForwardForwardDataAdapter(train_loader)
        for x_pos, x_neg, labels in ff_adapter:
            # x_pos: (batch, 784) with correct label embedded
            # x_neg: (batch, 784) with wrong label embedded
            # labels: (batch,) original labels
            ...
    """

    def __init__(self, base_loader: DataLoader, n_classes: int = 10):
        self.base_loader = base_loader
        self.n_classes = n_classes

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield (x_pos, x_neg, labels) tuples."""
        for images, labels in self.base_loader:
            # Flatten images: (batch, 1, 28, 28) -> (batch, 784)
            x = images.view(images.size(0), -1)
            x_pos = embed_label(x, labels, self.n_classes)
            x_neg = generate_negative(x, labels, self.n_classes)
            yield x_pos, x_neg, labels

    def __len__(self) -> int:
        """Number of batches."""
        return len(self.base_loader)
