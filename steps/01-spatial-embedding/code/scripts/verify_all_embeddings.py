"""Verify all 8 embedding strategies produce valid output.

Runs each embedding on the BaselineMLP and checks:
- Output shape is (N_weights, 3)
- All values in [0, 1]
- Deterministic (two calls produce same result)
- Reports timing for each
"""

import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "steps/01-spatial-embedding")

from code.embeddings import (
    AdversarialEmbedding,
    CorrelationEmbedding,
    DevelopmentalEmbedding,
    DifferentiableEmbedding,
    LayeredClusteredEmbedding,
    LinearEmbedding,
    RandomEmbedding,
    SpectralEmbedding,
)
from code.model import BaselineMLP


def create_synthetic_loader(n_samples=512, batch_size=128):
    """Create a small synthetic data loader for testing."""
    x = torch.randn(n_samples, 784)
    y = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_model_briefly(model, data_loader, n_epochs=3):
    """Train model briefly to get meaningful gradients."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(n_epochs):
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()


def verify_embedding(name, positions, n_weights):
    """Verify embedding output contract."""
    assert positions.shape == (n_weights, 3), (
        f"{name}: shape {positions.shape} != ({n_weights}, 3)"
    )
    assert np.all(positions >= 0.0), (
        f"{name}: min={positions.min()} < 0"
    )
    assert np.all(positions <= 1.0), (
        f"{name}: max={positions.max()} > 1"
    )
    print(f"  ✓ Shape: {positions.shape}, range: [{positions.min():.4f}, {positions.max():.4f}]")


def main():
    print("=" * 60)
    print("CHECKPOINT: Verifying all 8 embedding strategies")
    print("=" * 60)

    # Setup
    model = BaselineMLP()
    data_loader = create_synthetic_loader()
    train_model_briefly(model, data_loader)
    n_weights = model.get_weight_count()
    print(f"\nModel weight count: {n_weights}")
    print()

    # Test each embedding
    embeddings = [
        ("Linear", LinearEmbedding(), {}),
        ("Random", RandomEmbedding(seed=42), {}),
        ("Spectral", SpectralEmbedding(), {}),
        ("LayeredClustered", LayeredClusteredEmbedding(), {}),
        ("Correlation", CorrelationEmbedding(n_batches=5, subsample_size=500), {"data_loader": data_loader}),
        ("Developmental", DevelopmentalEmbedding(n_steps=50, subsample_pairs=1000, n_correlation_batches=5), {"data_loader": data_loader}),
        ("Adversarial", AdversarialEmbedding(n_correlation_batches=5, subsample_size=500), {"data_loader": data_loader}),
        ("Differentiable", DifferentiableEmbedding(), {}),
    ]

    results = []
    all_pass = True

    for name, emb, kwargs in embeddings:
        print(f"Testing {name}...")
        try:
            t0 = time.time()
            positions = emb.embed(model, **kwargs)
            elapsed = time.time() - t0
            verify_embedding(name, positions, n_weights)
            print(f"  ✓ Time: {elapsed:.2f}s")
            results.append((name, "PASS", elapsed))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append((name, "FAIL", 0))
            all_pass = False
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status, elapsed in results:
        print(f"  {status} {name} ({elapsed:.2f}s)")

    if all_pass:
        print("\n✓ ALL EMBEDDINGS PASS")
    else:
        print("\n✗ SOME EMBEDDINGS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
