"""Experiment runner for spatial embedding experiments.

Orchestrates experiment execution: seed management, metadata logging,
single-condition runs, and multi-seed comparisons.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from code.embeddings.base import EmbeddingStrategy
from code.experiment.reproducibility import (
    get_git_hash,
    get_hardware_info,
    get_library_versions,
    set_seeds,
)
from code.model import BaselineMLP, get_device
from code.spatial.coherence import SpatialCoherence
from code.spatial.knn_graph import KNNGraph
from code.spatial.lr_coupling import SpatialLRCoupling
from code.spatial.quality import QualityMeasurement


@dataclass
class CouplingConfig:
    """Configuration for spatial LR coupling."""

    k: int = 10
    alpha: float = 0.5


@dataclass
class ConditionResult:
    """Result from a single experiment condition run."""

    final_test_accuracy: float
    steps_to_95pct: int | None
    quality_score: float
    wall_clock_seconds: float
    embedding_method: str
    coupling_enabled: bool
    seed: int = 0
    coherence_score: float = 0.0


@dataclass
class ComparisonResult:
    """Aggregated result across multiple seeds for a condition."""

    condition_name: str
    embedding_method: str
    coupling_enabled: bool
    mean_accuracy: float
    std_accuracy: float
    mean_steps_to_95pct: float | None
    std_steps_to_95pct: float | None
    mean_quality_score: float
    std_quality_score: float
    mean_wall_clock: float
    individual_results: list[ConditionResult] = field(default_factory=list)


class ExperimentRunner:
    """Orchestrates spatial embedding experiment execution."""

    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        results_dir: str | Path | None = None,
    ):
        """
        Args:
            train_loader: Training data loader.
            test_loader: Test data loader.
            results_dir: Directory to save results. Defaults to results/.
        """
        self._train_loader = train_loader
        self._test_loader = test_loader

        if results_dir is None:
            results_dir = Path(__file__).parent.parent.parent / "results"
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._device = get_device()

    @property
    def device(self) -> torch.device:
        """Device used for training."""
        return self._device

    def log_metadata(
        self, experiment_name: str, config: dict[str, Any]
    ) -> Path:
        """Write experiment metadata to JSON file.

        Args:
            experiment_name: Name of the experiment.
            config: Hyperparameters and configuration dict.

        Returns:
            Path to the saved metadata file.
        """
        metadata = {
            "experiment_name": experiment_name,
            "config": config,
            "hardware": get_hardware_info(),
            "library_versions": get_library_versions(),
            "git_hash": get_git_hash(),
        }

        output_path = self._results_dir / f"{experiment_name}_metadata.json"
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return output_path

    def _evaluate_accuracy(self, model: nn.Module) -> float:
        """Evaluate model accuracy on test set.

        Args:
            model: Trained model to evaluate.

        Returns:
            Test accuracy as a fraction in [0, 1].
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self._test_loader:
                data, target = data.to(self._device), target.to(self._device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return correct / total if total > 0 else 0.0

    def run_condition(
        self,
        condition_name: str,
        model_factory: Callable[[], BaselineMLP],
        embedding: EmbeddingStrategy | None = None,
        coupling_config: CouplingConfig | None = None,
        n_epochs: int = 10,
        seed: int = 42,
        lr: float = 1e-3,
    ) -> ConditionResult:
        """Run a single experiment condition.

        Args:
            condition_name: Human-readable name for this condition.
            model_factory: Callable that creates a fresh model instance.
            embedding: Embedding strategy to use (None for uncoupled baseline).
            coupling_config: Coupling configuration (None to disable coupling).
            n_epochs: Number of training epochs.
            seed: Random seed for reproducibility.
            lr: Learning rate for Adam optimizer.

        Returns:
            ConditionResult with metrics from this run.
        """
        set_seeds(seed)
        start_time = time.time()

        # Create model and move to device
        model = model_factory()
        model = model.to(self._device)

        # Compute positions and set up coupling if configured
        positions = None
        coupling = None

        if embedding is not None:
            positions = embedding.embed(model, data_loader=self._train_loader)

            if coupling_config is not None:
                knn_graph = KNNGraph(positions, k=coupling_config.k)
                coupling = SpatialLRCoupling(
                    knn_graph, alpha=coupling_config.alpha
                )

        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        epoch_accuracies: list[float] = []
        steps_to_95pct: int | None = None
        total_steps = 0

        for epoch in range(n_epochs):
            model.train()
            for data, target in self._train_loader:
                data, target = data.to(self._device), target.to(self._device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # Apply spatial coupling before optimizer step
                if coupling is not None:
                    coupling.apply_to_optimizer(optimizer)

                optimizer.step()
                total_steps += 1

            # Evaluate at end of each epoch
            accuracy = self._evaluate_accuracy(model)
            epoch_accuracies.append(accuracy)

            # Check for steps to 95% of final accuracy
            # (We'll compute this after training completes)

        # Compute steps to 95% of final accuracy
        final_accuracy = epoch_accuracies[-1] if epoch_accuracies else 0.0
        target_accuracy = 0.95 * final_accuracy

        if target_accuracy > 0:
            for epoch_idx, acc in enumerate(epoch_accuracies):
                if acc >= target_accuracy:
                    # Approximate step count (steps per epoch * epoch index)
                    steps_per_epoch = len(self._train_loader)
                    steps_to_95pct = (epoch_idx + 1) * steps_per_epoch
                    break

        # Compute quality score if we have positions
        quality_score = 0.0
        coherence_score = 0.0

        if positions is not None:
            try:
                quality_measurement = QualityMeasurement(
                    positions, max_pairs=100_000
                )
                quality_result = quality_measurement.compute_quality_score(
                    model, self._train_loader, n_batches=20
                )
                quality_score = quality_result.score
            except Exception:
                quality_score = 0.0

            try:
                coherence = SpatialCoherence(n_components=10)
                weights = model.get_flat_weights().cpu().numpy()
                coherence_score = coherence.compute_coherence(weights, positions)
            except Exception:
                coherence_score = 0.0

        wall_clock = time.time() - start_time

        embedding_method = embedding.name if embedding is not None else "none"

        return ConditionResult(
            final_test_accuracy=final_accuracy,
            steps_to_95pct=steps_to_95pct,
            quality_score=quality_score,
            wall_clock_seconds=wall_clock,
            embedding_method=embedding_method,
            coupling_enabled=coupling_config is not None,
            seed=seed,
            coherence_score=coherence_score,
        )

    def run_comparison(
        self,
        conditions: list[
            tuple[str, Callable[[], BaselineMLP], EmbeddingStrategy | None, CouplingConfig | None]
        ],
        n_seeds: int = 3,
        seeds: list[int] | None = None,
        n_epochs: int = 10,
        lr: float = 1e-3,
    ) -> list[ComparisonResult]:
        """Run all conditions across multiple seeds and aggregate results.

        Args:
            conditions: List of (name, model_factory, embedding, coupling_config) tuples.
            n_seeds: Number of seeds to run per condition.
            seeds: Specific seeds to use. Defaults to [42, 123, 456, ...].
            n_epochs: Number of training epochs per run.
            lr: Learning rate for Adam optimizer.

        Returns:
            List of ComparisonResult with aggregated statistics.
        """
        if seeds is None:
            seeds = [42, 123, 456, 789, 1024][:n_seeds]

        comparison_results: list[ComparisonResult] = []

        for condition_name, model_factory, embedding, coupling_config in conditions:
            individual_results: list[ConditionResult] = []

            for seed in seeds:
                result = self.run_condition(
                    condition_name=condition_name,
                    model_factory=model_factory,
                    embedding=embedding,
                    coupling_config=coupling_config,
                    n_epochs=n_epochs,
                    seed=seed,
                    lr=lr,
                )
                individual_results.append(result)

            # Aggregate statistics
            accuracies = [r.final_test_accuracy for r in individual_results]
            quality_scores = [r.quality_score for r in individual_results]
            wall_clocks = [r.wall_clock_seconds for r in individual_results]

            steps_values = [
                r.steps_to_95pct
                for r in individual_results
                if r.steps_to_95pct is not None
            ]

            embedding_method = (
                embedding.name if embedding is not None else "none"
            )

            comparison_results.append(
                ComparisonResult(
                    condition_name=condition_name,
                    embedding_method=embedding_method,
                    coupling_enabled=coupling_config is not None,
                    mean_accuracy=float(np.mean(accuracies)),
                    std_accuracy=float(np.std(accuracies)),
                    mean_steps_to_95pct=(
                        float(np.mean(steps_values)) if steps_values else None
                    ),
                    std_steps_to_95pct=(
                        float(np.std(steps_values)) if steps_values else None
                    ),
                    mean_quality_score=float(np.mean(quality_scores)),
                    std_quality_score=float(np.std(quality_scores)),
                    mean_wall_clock=float(np.mean(wall_clocks)),
                    individual_results=individual_results,
                )
            )

        return comparison_results
