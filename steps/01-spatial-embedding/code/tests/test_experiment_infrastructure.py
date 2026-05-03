"""Tests for experiment infrastructure (Task 9).

Tests reproducibility utilities, ExperimentRunner, and visualization functions.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from code.experiment.reproducibility import (
    get_git_hash,
    get_hardware_info,
    get_library_versions,
    set_seeds,
)
from code.experiment.runner import (
    ComparisonResult,
    ConditionResult,
    CouplingConfig,
    ExperimentRunner,
)
from code.visualization.plots import (
    plot_boundary_regression,
    plot_developmental_trajectory,
    plot_quality_vs_performance,
    plot_spatial_coherence_comparison,
    plot_temporal_quality,
    plot_three_point_curve,
)


# --- Reproducibility tests ---


class TestSetSeeds:
    """Tests for set_seeds function."""

    def test_deterministic_random(self):
        """Setting same seed produces same random values."""
        set_seeds(42)
        val1 = np.random.rand(5)
        torch_val1 = torch.rand(5)

        set_seeds(42)
        val2 = np.random.rand(5)
        torch_val2 = torch.rand(5)

        np.testing.assert_array_equal(val1, val2)
        assert torch.allclose(torch_val1, torch_val2)

    def test_different_seeds_differ(self):
        """Different seeds produce different values."""
        set_seeds(42)
        val1 = np.random.rand(5)

        set_seeds(123)
        val2 = np.random.rand(5)

        assert not np.array_equal(val1, val2)


class TestGetHardwareInfo:
    """Tests for get_hardware_info function."""

    def test_returns_dict(self):
        """Returns a dictionary with expected keys."""
        info = get_hardware_info()
        assert isinstance(info, dict)
        assert "cpu" in info
        assert "device_type" in info
        assert "gpu_name" in info
        assert "platform" in info

    def test_device_type_valid(self):
        """Device type is one of the expected values."""
        info = get_hardware_info()
        assert info["device_type"] in ("mps", "cuda", "cpu")


class TestGetLibraryVersions:
    """Tests for get_library_versions function."""

    def test_returns_dict(self):
        """Returns a dictionary with expected keys."""
        versions = get_library_versions()
        assert isinstance(versions, dict)
        assert "torch" in versions
        assert "numpy" in versions
        assert "scipy" in versions

    def test_versions_are_strings(self):
        """All version values are strings."""
        versions = get_library_versions()
        for key, value in versions.items():
            assert isinstance(value, str), f"{key} version is not a string"


class TestGetGitHash:
    """Tests for get_git_hash function."""

    def test_returns_string_or_none(self):
        """Returns a string hash or None."""
        result = get_git_hash()
        assert result is None or isinstance(result, str)

    def test_hash_is_short(self):
        """If a hash is returned, it's a short hash (7-12 chars)."""
        result = get_git_hash()
        if result is not None:
            assert 4 <= len(result) <= 12


# --- ExperimentRunner tests ---


class TestConditionResult:
    """Tests for ConditionResult dataclass."""

    def test_creation(self):
        """Can create a ConditionResult with all fields."""
        result = ConditionResult(
            final_test_accuracy=0.95,
            steps_to_95pct=500,
            quality_score=0.3,
            wall_clock_seconds=10.5,
            embedding_method="linear",
            coupling_enabled=True,
            seed=42,
            coherence_score=0.1,
        )
        assert result.final_test_accuracy == 0.95
        assert result.steps_to_95pct == 500
        assert result.coupling_enabled is True


class TestCouplingConfig:
    """Tests for CouplingConfig dataclass."""

    def test_defaults(self):
        """Default values are correct."""
        config = CouplingConfig()
        assert config.k == 10
        assert config.alpha == 0.5

    def test_custom_values(self):
        """Can set custom values."""
        config = CouplingConfig(k=20, alpha=0.8)
        assert config.k == 20
        assert config.alpha == 0.8


class TestExperimentRunnerMetadata:
    """Tests for ExperimentRunner metadata logging."""

    def test_log_metadata_creates_file(self):
        """log_metadata creates a JSON file with expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal data loaders
            train_loader = _make_tiny_loader()
            test_loader = _make_tiny_loader()

            runner = ExperimentRunner(
                train_loader=train_loader,
                test_loader=test_loader,
                results_dir=tmpdir,
            )

            config = {"lr": 0.001, "epochs": 10, "seed": 42}
            path = runner.log_metadata("test_experiment", config)

            assert path.exists()
            with open(path) as f:
                metadata = json.load(f)

            assert metadata["experiment_name"] == "test_experiment"
            assert metadata["config"]["lr"] == 0.001
            assert "hardware" in metadata
            assert "library_versions" in metadata
            assert "git_hash" in metadata


# --- Visualization tests ---


class TestVisualizationPlots:
    """Tests for visualization functions (verify they produce files)."""

    def test_plot_quality_vs_performance(self, tmp_path):
        """Creates a PNG file."""
        output = tmp_path / "quality_vs_perf.png"
        result = plot_quality_vs_performance(
            quality_scores=[0.1, 0.3, 0.5, -0.2],
            performance_deltas=[0.01, 0.02, 0.03, -0.01],
            labels=["A", "B", "C", "D"],
            output_path=output,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_plot_boundary_regression(self, tmp_path):
        """Creates a PNG file with regression line."""
        output = tmp_path / "boundary.png"
        result = plot_boundary_regression(
            quality_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            performance_deltas=[0.01, 0.015, 0.02, 0.025, 0.03],
            output_path=output,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_plot_three_point_curve(self, tmp_path):
        """Creates a PNG file."""
        output = tmp_path / "three_point.png"
        result = plot_three_point_curve(
            adversarial_delta=-0.02,
            random_delta=0.005,
            best_delta=0.03,
            output_path=output,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_plot_developmental_trajectory(self, tmp_path):
        """Creates a PNG file."""
        output = tmp_path / "trajectory.png"
        history = [(i * 50, 0.1 + 0.02 * i) for i in range(20)]
        result = plot_developmental_trajectory(
            quality_history=history,
            output_path=output,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_plot_temporal_quality(self, tmp_path):
        """Creates a PNG file with multiple lines."""
        output = tmp_path / "temporal.png"
        trajectories = {
            "linear": [(i, 0.3 + 0.01 * i) for i in range(10)],
            "spectral": [(i, 0.4 - 0.005 * i) for i in range(10)],
            "random": [(i, 0.1 + 0.005 * i) for i in range(10)],
        }
        result = plot_temporal_quality(
            trajectories_dict=trajectories,
            output_path=output,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_plot_spatial_coherence_comparison(self, tmp_path):
        """Creates a PNG file."""
        output = tmp_path / "coherence.png"
        result = plot_spatial_coherence_comparison(
            coupled_score=0.35,
            uncoupled_score=0.12,
            output_path=output,
        )
        assert result.exists()
        assert result.stat().st_size > 0


# --- Helpers ---


def _make_tiny_loader():
    """Create a tiny DataLoader for testing (no real data needed)."""
    # Create a minimal dataset with 32 samples
    data = torch.randn(32, 1, 28, 28)
    targets = torch.randint(0, 10, (32,))
    dataset = torch.utils.data.TensorDataset(data, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=16)
