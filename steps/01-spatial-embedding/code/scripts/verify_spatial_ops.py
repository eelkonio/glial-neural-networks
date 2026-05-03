"""Verification script for spatial operations (Task 8).

Tests that all spatial modules work together correctly with the
BaselineMLP and embedding strategies.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from code.model import BaselineMLP
from code.embeddings.linear import LinearEmbedding
from code.spatial import (
    KNNGraph,
    SpatialLRCoupling,
    QualityMeasurement,
    SpatialCoherence,
    TemporalQualityTracker,
)


def create_dummy_data_loader(n_samples=200, batch_size=32):
    """Create a dummy data loader for testing."""
    x = torch.randn(n_samples, 1, 28, 28)
    y = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_knn_graph():
    """Test KNNGraph construction and queries."""
    print("Testing KNNGraph...")
    positions = np.random.uniform(0, 1, size=(100, 3))

    graph = KNNGraph(positions, k=10)
    assert graph.neighbor_indices.shape == (100, 10)
    assert graph.neighbor_distances.shape == (100, 10)

    # Test get_neighbors
    indices, distances = graph.get_neighbors(0)
    assert len(indices) == 10
    assert len(distances) == 10
    assert all(d >= 0 for d in distances)

    # Test k clamping
    graph_small = KNNGraph(positions[:5], k=10)
    assert graph_small.k == 4  # clamped to N-1

    print("  ✓ KNNGraph works correctly")


def test_spatial_lr_coupling():
    """Test SpatialLRCoupling formula."""
    print("Testing SpatialLRCoupling...")
    positions = np.random.uniform(0, 1, size=(50, 3))
    base_lr = np.full(50, 1.0)  # Use LR within clamp range [0.01, 10.0]

    graph = KNNGraph(positions, k=5)
    coupling = SpatialLRCoupling(graph, alpha=0.5)

    effective_lr = coupling.compute_effective_lr(base_lr)
    assert effective_lr.shape == (50,)

    # With uniform LR, coupling should not change values
    np.testing.assert_allclose(effective_lr, base_lr, rtol=1e-10)

    # Test with varying LR (within clamp range)
    varying_lr = np.random.uniform(0.1, 5.0, size=50)
    effective_varying = coupling.compute_effective_lr(varying_lr)
    assert effective_varying.shape == (50,)
    assert np.all(effective_varying >= 0.01)  # clamped lower bound
    assert np.all(effective_varying <= 10.0)  # clamped upper bound

    print("  ✓ SpatialLRCoupling works correctly")


def test_quality_measurement():
    """Test QualityMeasurement with a small model."""
    print("Testing QualityMeasurement...")
    model = BaselineMLP()
    embedding = LinearEmbedding()
    positions = embedding.embed(model)

    # Use a small subset for speed
    n_subset = 100
    positions_subset = positions[:n_subset]

    qm = QualityMeasurement(positions_subset, max_pairs=1000, n_bootstrap=100)
    assert not qm.needs_subsampling or qm.n_pairs <= 1000

    # Test with dummy data
    data_loader = create_dummy_data_loader()
    result = qm.compute_quality_score(model, data_loader, n_batches=5)

    assert result.ci_lower <= result.score <= result.ci_upper
    assert result.n_pairs_sampled > 0
    assert result.computation_time_seconds >= 0

    print(f"  ✓ QualityMeasurement: score={result.score:.4f}, "
          f"CI=[{result.ci_lower:.4f}, {result.ci_upper:.4f}]")


def test_spatial_coherence():
    """Test SpatialCoherence computation."""
    print("Testing SpatialCoherence...")
    n = 200
    weights = np.random.randn(n)
    positions = np.random.uniform(0, 1, size=(n, 3))

    coherence = SpatialCoherence(n_components=5, max_pairs=1000)
    score = coherence.compute_coherence(weights, positions)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

    print(f"  ✓ SpatialCoherence: score={score:.4f}")


def test_temporal_quality_tracker():
    """Test TemporalQualityTracker."""
    print("Testing TemporalQualityTracker...")
    tracker = TemporalQualityTracker(record_interval_epochs=2)

    # Simulate recording with a small model
    model = BaselineMLP()
    positions = np.random.uniform(0, 1, size=(50, 3))
    qm = QualityMeasurement(positions, max_pairs=500, n_bootstrap=50)
    data_loader = create_dummy_data_loader()

    # Record a few points
    score = tracker.record(0, 0, qm, model, data_loader)
    assert isinstance(score, float)

    score2 = tracker.record(2, 100, qm, model, data_loader)
    assert isinstance(score2, float)

    trajectory = tracker.get_trajectory()
    assert len(trajectory) == 2
    assert trajectory[0][0] == 0  # epoch
    assert trajectory[1][0] == 2  # epoch

    # Test degradation detection
    degraded = tracker.detect_degradation(threshold=0.5)
    assert isinstance(degraded, bool)

    print(f"  ✓ TemporalQualityTracker: trajectory length={len(trajectory)}, "
          f"degraded={degraded}")


def test_optimizer_integration():
    """Test SpatialLRCoupling with Adam optimizer."""
    print("Testing optimizer integration...")
    model = BaselineMLP()
    embedding = LinearEmbedding()
    positions = embedding.embed(model)

    # Build KNN graph over all weight positions
    graph = KNNGraph(positions, k=5)
    coupling = SpatialLRCoupling(graph, alpha=0.5)

    # Only optimize weight parameters (not biases) for this test
    weight_params = [layer.weight for layer in model.weight_layers]
    optimizer = torch.optim.Adam(weight_params, lr=0.001)

    # Run a training step
    data_loader = create_dummy_data_loader()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Apply coupling (modifies gradients)
        coupling.apply_to_optimizer(optimizer)

        optimizer.step()
        break  # Just one step

    print("  ✓ Optimizer integration works correctly")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    test_knn_graph()
    test_spatial_lr_coupling()
    test_quality_measurement()
    test_spatial_coherence()
    test_temporal_quality_tracker()
    test_optimizer_integration()

    print("\n✅ All spatial operations verified successfully!")
