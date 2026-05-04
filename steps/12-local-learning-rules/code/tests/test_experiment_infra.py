"""Tests for experiment infrastructure: metrics, deficiency, spatial quality, comparison."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure correct imports
step_dir = str(Path(__file__).parent.parent.parent)
if step_dir not in sys.path:
    sys.path.insert(0, step_dir)

from code.experiment.metrics import (
    PerformanceMetrics,
    compute_convergence_epoch,
    compute_stability,
    compute_weight_norms,
)
from code.experiment.deficiency import (
    compute_credit_assignment_reach,
    compute_weight_stability,
    compute_representation_redundancy,
    compute_inter_layer_coordination,
    _linear_cka,
)
from code.experiment.spatial_quality import (
    _get_weight_positions,
    compute_spatial_quality,
)
from code.network.local_mlp import LocalMLP
from code.rules.hebbian import HebbianRule


class TestConvergenceEpoch:
    """Tests for compute_convergence_epoch."""

    def test_basic_convergence(self):
        """Convergence epoch is first reaching 90% of max."""
        history = [0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.92, 0.93, 0.94, 0.95]
        # max = 0.95, threshold = 0.855
        # First epoch >= 0.855 is index 5 (0.9)
        assert compute_convergence_epoch(history) == 5

    def test_immediate_convergence(self):
        """If first epoch already meets threshold."""
        history = [0.9, 0.91, 0.92]
        # max = 0.92, threshold = 0.828
        # First epoch >= 0.828 is index 0
        assert compute_convergence_epoch(history) == 0

    def test_empty_history(self):
        """Empty history returns None."""
        assert compute_convergence_epoch([]) is None

    def test_single_value(self):
        """Single value always converges at epoch 0."""
        assert compute_convergence_epoch([0.5]) == 0

    def test_monotonic_increase(self):
        """Monotonically increasing sequence."""
        history = [i / 100 for i in range(100)]
        # max = 0.99, threshold = 0.891
        # First >= 0.891 is index 90 (0.90)
        result = compute_convergence_epoch(history)
        assert result == 90


class TestStability:
    """Tests for compute_stability."""

    def test_constant_accuracy(self):
        """Constant accuracy has zero stability (std=0)."""
        history = [0.9] * 20
        assert compute_stability(history) == pytest.approx(0.0)

    def test_varying_accuracy(self):
        """Varying accuracy has non-zero stability."""
        history = [0.8, 0.85, 0.82, 0.88, 0.84, 0.86, 0.83, 0.87, 0.85, 0.84]
        result = compute_stability(history, window=10)
        assert result > 0.0

    def test_short_history(self):
        """Short history uses all available values."""
        history = [0.5, 0.6, 0.7]
        result = compute_stability(history, window=10)
        assert result > 0.0


class TestWeightNorms:
    """Tests for compute_weight_norms."""

    def test_returns_correct_count(self):
        """Returns one norm per layer."""
        model = LocalMLP()
        norms = compute_weight_norms(model)
        assert len(norms) == 5  # 5 layers

    def test_norms_positive(self):
        """All norms are positive."""
        model = LocalMLP()
        norms = compute_weight_norms(model)
        assert all(n > 0 for n in norms)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    def test_record_and_retrieve(self):
        """Can record epochs and retrieve history."""
        metrics = PerformanceMetrics(rule_name="test", seed=42)
        metrics.record_epoch(0, 0.5, 0.4, 1.0, 1.1)
        metrics.record_epoch(1, 0.6, 0.55, 0.8, 0.9)

        assert len(metrics.epochs) == 2
        assert metrics.accuracy_history == [0.4, 0.55]

    def test_to_csv_rows(self):
        """CSV rows have correct structure."""
        metrics = PerformanceMetrics(rule_name="hebbian", seed=42)
        metrics.record_epoch(0, 0.5, 0.4, 1.0, 1.1, weight_norms=[1.0, 2.0])

        rows = metrics.to_csv_rows()
        assert len(rows) == 1
        assert rows[0]["rule"] == "hebbian"
        assert rows[0]["seed"] == 42
        assert rows[0]["epoch"] == 0

    def test_save_to_csv(self, tmp_path):
        """Can save to CSV file."""
        metrics = PerformanceMetrics(rule_name="test", seed=42)
        metrics.record_epoch(0, 0.5, 0.4, 1.0, 1.1)

        path = tmp_path / "test.csv"
        PerformanceMetrics.save_all_to_csv([metrics], path)
        assert path.exists()


class TestWeightStability:
    """Tests for compute_weight_stability."""

    def test_stable_weights(self):
        """Constant weight norms are stable."""
        history = [[1.0, 1.0, 1.0]] * 20
        result = compute_weight_stability(history)
        assert result["overall_stable"] is True

    def test_exploding_weights(self):
        """Exponentially growing norms are unstable."""
        history = [[2**i, 2**i, 2**i] for i in range(20)]
        result = compute_weight_stability(history)
        assert result["overall_stable"] is False

    def test_empty_history(self):
        """Empty history is considered stable."""
        result = compute_weight_stability([])
        assert result["overall_stable"] is True


class TestRepresentationRedundancy:
    """Tests for compute_representation_redundancy."""

    def test_identical_columns_high_redundancy(self):
        """Identical unit activations give redundancy = 1.0."""
        # All units have the same activation pattern
        act = torch.ones(10, 5)  # 10 samples, 5 units all identical
        result = compute_representation_redundancy([act])
        assert result[0] == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_low_redundancy(self):
        """Orthogonal unit activations give redundancy ≈ 0."""
        act = torch.eye(10)  # 10 samples, 10 units, identity matrix
        result = compute_representation_redundancy([act])
        assert abs(result[0]) < 0.1

    def test_bounds(self):
        """Redundancy is in [-1, 1]."""
        act = torch.randn(50, 20)
        result = compute_representation_redundancy([act])
        assert -1.0 <= result[0] <= 1.0


class TestInterLayerCoordination:
    """Tests for compute_inter_layer_coordination (CKA)."""

    def test_self_cka_is_one(self):
        """CKA of a matrix with itself is 1.0."""
        X = torch.randn(50, 20)
        cka = _linear_cka(X, X)
        assert cka == pytest.approx(1.0, abs=0.01)

    def test_cka_bounds(self):
        """CKA is in [0, 1]."""
        X = torch.randn(50, 20)
        Y = torch.randn(50, 15)
        cka = _linear_cka(X, Y)
        assert 0.0 <= cka <= 1.0

    def test_returns_correct_count(self):
        """Returns n_layers - 1 CKA values."""
        activations = [torch.randn(50, 128) for _ in range(5)]
        result = compute_inter_layer_coordination(activations)
        assert len(result) == 4


class TestCreditAssignmentReach:
    """Tests for compute_credit_assignment_reach."""

    def test_returns_per_layer_correlations(self):
        """Returns one correlation per layer."""
        model = LocalMLP()
        rule = HebbianRule()
        x = torch.randn(32, 784)
        labels = torch.randint(0, 10, (32,))

        correlations = compute_credit_assignment_reach(model, rule, x, labels)
        assert len(correlations) == 5

    def test_correlations_bounded(self):
        """Correlations are in [-1, 1]."""
        model = LocalMLP()
        rule = HebbianRule()
        x = torch.randn(32, 784)
        labels = torch.randint(0, 10, (32,))

        correlations = compute_credit_assignment_reach(model, rule, x, labels)
        for c in correlations:
            assert -1.0 <= c <= 1.0


class TestSpatialPositions:
    """Tests for _get_weight_positions."""

    def test_correct_count(self):
        """Returns one position per weight."""
        model = LocalMLP()
        positions = _get_weight_positions(model)

        # Count total weights: 784*128 + 128*128 + 128*128 + 128*128 + 128*10
        expected = 784 * 128 + 128 * 128 * 3 + 128 * 10
        assert positions.shape == (expected, 3)

    def test_positions_normalized(self):
        """Positions are in [0, 1]."""
        model = LocalMLP()
        positions = _get_weight_positions(model)
        assert positions.min() >= 0.0
        assert positions.max() <= 1.0
