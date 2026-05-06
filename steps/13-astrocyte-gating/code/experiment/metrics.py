"""Metrics collection for Step 13 experiments.

Records per-epoch metrics and stores results in CSV format.
"""

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EpochResult:
    """Per-epoch metrics for a single training run."""

    epoch: int
    train_loss: float
    test_accuracy: float
    test_loss: float
    gate_fraction_open: float
    weight_norm: float = 0.0
    has_nan: bool = False
    wall_clock_seconds: float = 0.0


@dataclass
class ConditionResult:
    """Full results for one condition × seed combination."""

    condition_name: str
    seed: int
    n_epochs: int
    final_accuracy: float
    best_accuracy: float
    any_nan: bool
    epoch_results: list[EpochResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def save_epoch_results_csv(
    results: list[EpochResult],
    condition_name: str,
    seed: int,
    output_dir: str | Path,
    timestamp: str | None = None,
) -> Path:
    """Save per-epoch results to CSV.

    Args:
        results: List of EpochResult for each epoch.
        condition_name: Name of the experimental condition.
        seed: Random seed used.
        output_dir: Directory to write CSV file.
        timestamp: Optional timestamp string for filename. If None, uses current UTC.

    Returns:
        Path to the written CSV file.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{condition_name}_seed{seed}_{timestamp}.csv"
    filepath = output_dir / filename

    fieldnames = [
        "epoch", "train_loss", "test_accuracy", "test_loss",
        "gate_fraction_open", "weight_norm", "has_nan", "wall_clock_seconds",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "epoch": r.epoch,
                "train_loss": f"{r.train_loss:.6f}",
                "test_accuracy": f"{r.test_accuracy:.6f}",
                "test_loss": f"{r.test_loss:.6f}",
                "gate_fraction_open": f"{r.gate_fraction_open:.6f}",
                "weight_norm": f"{r.weight_norm:.6f}",
                "has_nan": r.has_nan,
                "wall_clock_seconds": f"{r.wall_clock_seconds:.2f}",
            })

    return filepath


def save_metadata_json(
    metadata: dict[str, Any],
    output_dir: str | Path,
    timestamp: str | None = None,
) -> Path:
    """Save experiment metadata to JSON.

    Args:
        metadata: Dict with hyperparams, seeds, versions, hardware info.
        output_dir: Directory to write JSON file.
        timestamp: Optional timestamp string for filename.

    Returns:
        Path to the written JSON file.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"metadata_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return filepath


def load_epoch_results_csv(filepath: str | Path) -> list[EpochResult]:
    """Load per-epoch results from CSV.

    Args:
        filepath: Path to CSV file.

    Returns:
        List of EpochResult.
    """
    results = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(EpochResult(
                epoch=int(row["epoch"]),
                train_loss=float(row["train_loss"]),
                test_accuracy=float(row["test_accuracy"]),
                test_loss=float(row["test_loss"]),
                gate_fraction_open=float(row["gate_fraction_open"]),
                weight_norm=float(row["weight_norm"]),
                has_nan=row["has_nan"] == "True",
                wall_clock_seconds=float(row["wall_clock_seconds"]),
            ))
    return results
