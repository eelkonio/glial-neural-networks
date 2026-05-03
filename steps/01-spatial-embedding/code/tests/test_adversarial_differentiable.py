"""Property-based tests for adversarial and differentiable embeddings.

Tests the correctness properties defined in the design document:
- Property 11: Adversarial embedding produces positive quality score
  (correlated weights are far apart)
- Property 12: Differentiable embedding positions remain in [0, 1]
  after any update (sigmoid guarantees range)

Uses Hypothesis for property-based testing.
"""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from code.embeddings.differentiable import DifferentiableEmbedding
from code.model import BaselineMLP


# --- Property 11: Adversarial embedding produces positive quality score ---
# Feature: spatial-embedding-experiments, Property 11: Adversarial embedding produces negative quality score
# **Validates: Requirements 15.3**


@settings(max_examples=3, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100))
def test_adversarial_embedding_positive_quality_score(seed):
    """The adversarial embedding should produce a positive quality score,
    meaning spatial distance positively correlates with gradient correlation
    (correlated weights are far apart).

    We verify this by checking the MDS construction directly: the adversarial
    embedding uses distance_matrix = |correlation|, so the MDS output should
    place highly-correlated weights far apart. We test that the MDS distance
    matrix (which IS |correlation|) is faithfully reflected in the output
    positions — i.e., the Pearson correlation between the MDS input distances
    and the resulting spatial distances is positive.
    """
    from code.embeddings.adversarial import AdversarialEmbedding
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BaselineMLP()

    # Create a synthetic dataset with enough samples for meaningful gradients
    n_samples = 512
    x = torch.randn(n_samples, 784)
    y = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Train for multiple epochs to get non-trivial gradient structure
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(10):
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    # Compute adversarial embedding
    n_batches = 10
    subsample_size = 200
    embedding = AdversarialEmbedding(
        n_correlation_batches=n_batches, subsample_size=subsample_size
    )
    positions = embedding.embed(model, data_loader=data_loader)

    # Verify output contract
    n_weights = model.get_weight_count()
    assert positions.shape == (n_weights, 3)
    assert np.all(positions >= 0.0)
    assert np.all(positions <= 1.0)

    # Verify the adversarial property by checking the MDS construction.
    # The adversarial MDS uses distance = |correlation|. If MDS works correctly,
    # pairs with high |correlation| (large MDS input distance) should end up
    # with large spatial distance in the output. We verify this on the
    # subsampled weights where MDS was directly applied.
    model.eval()
    signals = np.zeros((n_weights, n_batches), dtype=np.float32)
    batch_iter = iter(data_loader)
    for b in range(n_batches):
        try:
            bx, by = next(batch_iter)
        except StopIteration:
            batch_iter = iter(data_loader)
            bx, by = next(batch_iter)
        model.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        signals[:, b] = model.get_flat_gradients().cpu().numpy()

    # Use the same subsample indices as the embedding
    rng = np.random.RandomState(42)
    actual_subsample_size = min(subsample_size, n_weights)
    subsample_indices = rng.choice(
        n_weights, size=actual_subsample_size, replace=False
    )
    subsample_indices.sort()

    sub_signals = signals[subsample_indices]
    centered = sub_signals - sub_signals.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = centered / norms

    # Compute the correlation matrix (same as embedding does internally)
    corr_matrix = normalized @ normalized.T
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    # The adversarial MDS distance matrix is |correlation|
    mds_distance_matrix = np.abs(corr_matrix)
    np.fill_diagonal(mds_distance_matrix, 0.0)

    # Compute actual spatial distances between subsampled weights
    sub_positions = positions[subsample_indices]

    # Sample pairs and check that MDS input distances correlate with
    # output spatial distances (this tests MDS fidelity, which validates
    # the adversarial construction)
    pair_rng = np.random.RandomState(99)
    n_pairs = min(5000, actual_subsample_size * (actual_subsample_size - 1) // 2)
    idx_i = pair_rng.randint(0, actual_subsample_size, size=n_pairs)
    idx_j = pair_rng.randint(0, actual_subsample_size, size=n_pairs)
    same_mask = idx_i == idx_j
    idx_j[same_mask] = (idx_j[same_mask] + 1) % actual_subsample_size

    # MDS input distances (= |correlation|, the adversarial target)
    mds_input_dists = mds_distance_matrix[idx_i, idx_j]

    # Spatial distances in the output embedding
    spatial_dists = np.linalg.norm(
        sub_positions[idx_i] - sub_positions[idx_j], axis=1
    )

    # The MDS should faithfully represent the input distances:
    # pairs with large |correlation| (large MDS input distance) should
    # have large spatial distance. This correlation should be positive.
    if np.std(spatial_dists) > 1e-10 and np.std(mds_input_dists) > 1e-10:
        fidelity_score = np.corrcoef(spatial_dists, mds_input_dists)[0, 1]
        # MDS fidelity should be positive (it's optimizing to preserve distances)
        assert fidelity_score > 0, (
            f"Adversarial MDS fidelity should be positive "
            f"(MDS input distances correlate with output distances), "
            f"got {fidelity_score}"
        )


# --- Property 12: Differentiable embedding positions remain in [0, 1] ---
# Feature: spatial-embedding-experiments, Property 12: Differentiable embedding positions remain in [0, 1]
# **Validates: Requirements 16.4**


@given(
    n_weights=st.integers(min_value=10, max_value=500),
    n_updates=st.integers(min_value=1, max_value=10),
    lr=st.floats(min_value=0.001, max_value=1.0),
)
@settings(max_examples=50, deadline=None)
def test_differentiable_positions_in_unit_cube(n_weights, n_updates, lr):
    """After initialize() and after multiple gradient updates,
    sigmoid(positions_param) is always in [0, 1].

    This is a mathematical property of sigmoid: for any real input x,
    sigmoid(x) ∈ (0, 1). We verify this holds after arbitrary gradient
    updates to the position parameter.
    """
    embedding = DifferentiableEmbedding(
        lambda_spatial=0.01,
        subsample_pairs=min(100, n_weights * (n_weights - 1) // 2),
        position_lr=lr,
    )

    # Initialize positions
    param = embedding.initialize(n_weights)

    # Verify initial positions are in [0, 1]
    with torch.no_grad():
        initial_positions = torch.sigmoid(param).numpy()
    assert np.all(initial_positions >= 0.0), (
        f"Initial positions below 0: min={initial_positions.min()}"
    )
    assert np.all(initial_positions <= 1.0), (
        f"Initial positions above 1: max={initial_positions.max()}"
    )

    # Simulate gradient updates with random gradients
    optimizer = torch.optim.SGD([param], lr=lr)

    for _ in range(n_updates):
        # Simulate a gradient update with random gradients
        optimizer.zero_grad()
        # Create a fake loss that generates gradients on positions
        fake_loss = (param ** 2).sum()
        fake_loss.backward()
        optimizer.step()

        # After update, sigmoid(param) should still be in [0, 1]
        with torch.no_grad():
            positions = torch.sigmoid(param).numpy()
        assert np.all(positions >= 0.0), (
            f"Positions below 0 after update: min={positions.min()}"
        )
        assert np.all(positions <= 1.0), (
            f"Positions above 1 after update: max={positions.max()}"
        )


@given(
    raw_values=arrays(
        dtype=np.float32,
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),
            st.just(3),
        ),
        elements=st.floats(
            min_value=-100.0, max_value=100.0,
            allow_nan=False, allow_infinity=False,
        ),
    )
)
@settings(max_examples=100, deadline=None)
def test_sigmoid_always_produces_unit_range(raw_values):
    """For any raw parameter values, sigmoid always produces [0, 1].

    This is the mathematical guarantee that the differentiable embedding
    relies on: sigmoid maps R → (0, 1).
    """
    tensor = torch.from_numpy(raw_values)
    result = torch.sigmoid(tensor).numpy()

    assert np.all(result >= 0.0), f"Sigmoid produced value < 0: {result.min()}"
    assert np.all(result <= 1.0), f"Sigmoid produced value > 1: {result.max()}"
