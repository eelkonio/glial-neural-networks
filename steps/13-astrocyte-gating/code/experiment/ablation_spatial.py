"""Spatial domain structure ablation experiment.

Tests whether spatial coherence of domain assignment matters.

Two strategies (both using directional gate):
(a) Spatial assignment (DomainConfig(mode="spatial"))
(b) Random assignment (DomainConfig(mode="random"))
"""

import time
from datetime import datetime, timezone
from pathlib import Path

from code.calcium.config import CalciumConfig
from code.domains.config import DomainConfig
from code.experiment.config import GateConfig, ExperimentCondition
from code.experiment.metrics import (
    ConditionResult,
    save_epoch_results_csv,
)
from code.experiment.runner import (
    ExperimentRunner,
    run_condition,
    set_all_seeds,
)


def get_spatial_ablation_conditions() -> list[ExperimentCondition]:
    """Get the two spatial ablation conditions.

    Both use directional gate with CalciumConfig(d_serine_threshold=0.02).
    """
    # (a) Spatial assignment
    spatial = ExperimentCondition(
        name="ablation_spatial",
        gate_config=GateConfig(variant="directional"),
        calcium_config=CalciumConfig(d_serine_threshold=0.02),
        domain_config=DomainConfig(mode="spatial"),
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )

    # (b) Random assignment
    random_assign = ExperimentCondition(
        name="ablation_random_assign",
        gate_config=GateConfig(variant="directional"),
        calcium_config=CalciumConfig(d_serine_threshold=0.02),
        domain_config=DomainConfig(mode="random"),
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )

    return [spatial, random_assign]


def run_spatial_ablation(
    n_epochs: int = 50,
    seeds: list[int] = None,
    batch_size: int = 128,
    device: str = "cpu",
    output_dir: str = "results",
    verbose: bool = True,
) -> list[ConditionResult]:
    """Run the spatial domain structure ablation experiment.

    Args:
        n_epochs: Number of epochs per condition.
        seeds: Random seeds.
        batch_size: Batch size.
        device: Torch device.
        output_dir: Output directory for results.
        verbose: Print progress.

    Returns:
        List of ConditionResult.
    """
    seeds = seeds or [42, 123, 456]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_results = []

    conditions = get_spatial_ablation_conditions()

    for condition in conditions:
        for seed in seeds:
            # UTC timestamp before
            start_time = datetime.now(timezone.utc)
            print(f"\n[{start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
                  f"Starting: {condition.name} seed={seed}")

            t0 = time.time()
            set_all_seeds(seed)

            result = run_condition(
                condition=condition,
                seed=seed,
                n_epochs=n_epochs,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
            )

            elapsed = time.time() - t0
            end_time = datetime.now(timezone.utc)
            print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
                  f"Completed: {condition.name} seed={seed} ({elapsed:.1f}s) "
                  f"acc={result.final_accuracy:.4f}")

            # Save CSV
            save_epoch_results_csv(
                results=result.epoch_results,
                condition_name=condition.name,
                seed=seed,
                output_dir=str(output_dir),
                timestamp=timestamp,
            )

            all_results.append(result)

    return all_results
