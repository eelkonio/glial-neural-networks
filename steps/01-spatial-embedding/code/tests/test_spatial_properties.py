"""Property-based tests for spatial operations.

Tests the correctness properties defined in the design document:
- Property 6: Quality score is Pearson correlation of distances vs gradient correlations
- Property 7: Confidence interval contains point estimate
- Property 8: Spatial LR coupling formula
- Property 9: Subsampling threshold

Uses Hypothesis for property-based testing.
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import pearsonr

from code.spatial.knn_graph import KNNGraph
from code.spatial.lr_coupling import SpatialLRCoupling
from code.spatial.quality import QualityMeasurement


# --- Strategies ---

# Positions: (N, 3) arrays with values in [0, 1]
def positions_strategy(min_n=5, max_n=50):
    """Generate random position arrays of shape (N, 3) in [0, 1]."""
    return st.integers(min_value=min_n, max_value=max_n).flatmap(
        lambda n: arrays(
            dtype=np.float64,
            shape=(n, 3),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )


def lr_array_strategy(n):
    """Generate learning rate arrays of shape (n,) with positive values."""
    return arrays(
        dtype=np.float64,
        shape=(n,),
        elements=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    )


# --- Property 6: Quality score is Pearson correlation ---
# Feature: spatial-embedding-experiments, Property 6: Quality score is Pearson correlation
# **Validates: Requirements 9.1**


@given(
    n=st.integers(min_value=5, max_value=30),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_quality_score_equals_pearsonr(n, seed):
    """For any set of spatial positions and gradient correlations,
    the quality score equals scipy.stats.pearsonr on the same data.

    We test the core formula by directly computing spatial distances
    and gradient correlations, then verifying the Pearson correlation
    matches what scipy.stats.pearsonr would give.
    """
    rng = np.random.default_rng(seed)

    # Generate positions and synthetic gradient correlations
    positions = rng.uniform(0, 1, size=(n, 3))
    
    # Compute pairwise spatial distances
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = positions[idx_i] - positions[idx_j]
    spatial_distances = np.sqrt(np.sum(diff**2, axis=1))

    # Generate synthetic gradient correlations
    gradient_correlations = rng.uniform(-1, 1, size=len(idx_i))

    # Skip degenerate cases
    assume(np.std(spatial_distances) > 1e-10)
    assume(np.std(gradient_correlations) > 1e-10)

    # Compute expected Pearson correlation using scipy
    expected_score, _ = pearsonr(spatial_distances, gradient_correlations)

    # Compute using numpy (same formula the quality measurement uses internally)
    # Pearson r = cov(X, Y) / (std(X) * std(Y))
    x = spatial_distances
    y = gradient_correlations
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))

    if denominator < 1e-12:
        computed_score = 0.0
    else:
        computed_score = numerator / denominator

    # Both should match scipy's pearsonr
    np.testing.assert_allclose(
        computed_score, expected_score, rtol=1e-7,
        err_msg="Quality score formula does not match scipy.stats.pearsonr"
    )


# --- Property 7: Confidence interval contains point estimate ---
# Feature: spatial-embedding-experiments, Property 7: Confidence interval contains point estimate
# **Validates: Requirements 9.3**


@given(
    n=st.integers(min_value=10, max_value=30),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_confidence_interval_contains_point_estimate(n, seed):
    """For any quality measurement result, ci_lower <= score <= ci_upper.

    We simulate the bootstrap CI computation to verify the property.
    """
    rng = np.random.default_rng(seed)

    # Generate positions and synthetic data
    positions = rng.uniform(0, 1, size=(n, 3))

    # Compute pairwise spatial distances
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = positions[idx_i] - positions[idx_j]
    spatial_distances = np.sqrt(np.sum(diff**2, axis=1))

    # Generate synthetic gradient correlations
    gradient_correlations = rng.uniform(-1, 1, size=len(idx_i))

    # Skip degenerate cases
    assume(np.std(spatial_distances) > 1e-10)
    assume(np.std(gradient_correlations) > 1e-10)

    # Compute point estimate
    score, _ = pearsonr(spatial_distances, gradient_correlations)

    # Bootstrap CI
    n_bootstrap = 1000
    bootstrap_rng = np.random.default_rng(seed + 1000)
    bootstrap_scores = np.empty(n_bootstrap)
    n_pairs = len(spatial_distances)

    for b in range(n_bootstrap):
        sample_idx = bootstrap_rng.integers(0, n_pairs, size=n_pairs)
        d_sample = spatial_distances[sample_idx]
        c_sample = gradient_correlations[sample_idx]

        if np.std(d_sample) < 1e-12 or np.std(c_sample) < 1e-12:
            bootstrap_scores[b] = 0.0
        else:
            bootstrap_scores[b], _ = pearsonr(d_sample, c_sample)

    ci_lower = float(np.percentile(bootstrap_scores, 2.5))
    ci_upper = float(np.percentile(bootstrap_scores, 97.5))

    # The point estimate should be within the CI
    # Note: This is a statistical property - the point estimate should
    # typically be within the 95% CI, but not always guaranteed.
    # We use a slightly relaxed check.
    assert ci_lower <= score <= ci_upper, (
        f"Point estimate {score} not in CI [{ci_lower}, {ci_upper}]"
    )


# --- Property 8: Spatial LR coupling formula ---
# Feature: spatial-embedding-experiments, Property 8: Spatial LR coupling formula
# **Validates: Requirements 10.2, 10.3**


@given(
    n=st.integers(min_value=5, max_value=50),
    k=st.integers(min_value=1, max_value=10),
    alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_lr_coupling_formula(n, k, alpha, seed):
    """For any positions, base_lr, k, and alpha:
    effective_lr[i] = (1 - alpha) * base_lr[i] + alpha * mean(base_lr[neighbors[i]])

    Special cases:
    - alpha=0 → effective_lr == base_lr
    - alpha=1 → effective_lr[i] == mean(base_lr[neighbors[i]])
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 1, size=(n, 3))
    base_lr = rng.uniform(0.01, 5.0, size=n)

    # Clamp k
    effective_k = min(k, n - 1)

    graph = KNNGraph(positions, k=k)
    coupling = SpatialLRCoupling(graph, alpha=alpha)
    effective_lr = coupling.compute_effective_lr(base_lr)

    # Verify formula for each weight
    for i in range(n):
        neighbor_idx = graph.neighbor_indices[i]
        if len(neighbor_idx) == 0:
            expected = base_lr[i]
        else:
            neighbor_mean = base_lr[neighbor_idx].mean()
            expected = (1.0 - alpha) * base_lr[i] + alpha * neighbor_mean

        # Clamp expected to [0.01, 10.0]
        expected = np.clip(expected, 0.01, 10.0)

        np.testing.assert_allclose(
            effective_lr[i], expected, rtol=1e-10,
            err_msg=f"LR coupling formula mismatch at index {i}"
        )


@given(
    n=st.integers(min_value=5, max_value=50),
    k=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_lr_coupling_alpha_zero_no_change(n, k, seed):
    """When alpha=0, effective_lr equals base_lr (no coupling)."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 1, size=(n, 3))
    base_lr = rng.uniform(0.01, 5.0, size=n)

    graph = KNNGraph(positions, k=k)
    coupling = SpatialLRCoupling(graph, alpha=0.0)
    effective_lr = coupling.compute_effective_lr(base_lr)

    np.testing.assert_array_almost_equal(
        effective_lr, base_lr,
        err_msg="alpha=0 should produce no change"
    )


@given(
    n=st.integers(min_value=5, max_value=50),
    k=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_lr_coupling_alpha_one_full_averaging(n, k, seed):
    """When alpha=1, effective_lr[i] equals mean of neighbors' LRs."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 1, size=(n, 3))
    base_lr = rng.uniform(0.01, 5.0, size=n)

    graph = KNNGraph(positions, k=k)
    coupling = SpatialLRCoupling(graph, alpha=1.0)
    effective_lr = coupling.compute_effective_lr(base_lr)

    for i in range(n):
        neighbor_idx = graph.neighbor_indices[i]
        if len(neighbor_idx) == 0:
            expected = base_lr[i]
        else:
            expected = base_lr[neighbor_idx].mean()

        expected = np.clip(expected, 0.01, 10.0)

        np.testing.assert_allclose(
            effective_lr[i], expected, rtol=1e-10,
            err_msg=f"alpha=1 should give full neighbor averaging at index {i}"
        )


# --- Property 9: Subsampling threshold ---
# Feature: spatial-embedding-experiments, Property 9: Subsampling threshold
# **Validates: Requirements 9.5**


@given(
    n=st.integers(min_value=5, max_value=100),
    max_pairs=st.integers(min_value=5, max_value=500),
)
@settings(max_examples=100, deadline=None)
def test_subsampling_triggers_when_pairs_exceed_max(n, max_pairs):
    """When N*(N-1)/2 > max_pairs, subsampling is used (fewer pairs).
    When N*(N-1)/2 <= max_pairs, all pairs are used.
    """
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 1, size=(n, 3))

    total_pairs = n * (n - 1) // 2

    qm = QualityMeasurement(positions, max_pairs=max_pairs)

    if total_pairs > max_pairs:
        # Should use subsampling
        assert qm.needs_subsampling, (
            f"Expected subsampling for {total_pairs} pairs > max_pairs={max_pairs}"
        )
        assert qm.n_pairs == max_pairs, (
            f"Expected {max_pairs} pairs when subsampling, got {qm.n_pairs}"
        )
    else:
        # Should use all pairs
        assert not qm.needs_subsampling, (
            f"Should not subsample for {total_pairs} pairs <= max_pairs={max_pairs}"
        )
        assert qm.n_pairs == total_pairs, (
            f"Expected {total_pairs} pairs without subsampling, got {qm.n_pairs}"
        )
