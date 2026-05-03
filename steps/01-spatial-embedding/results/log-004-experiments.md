# Log 004: Experiment Implementation

**Date**: 2025-01-XX  
**Tasks**: 11, 12, 13 — Comparison experiment, analysis experiments, wiring

## Summary

Implemented the full experiment orchestration layer that composes all previously-built components (embeddings, spatial operations, visualization) into runnable experiments.

## What Was Built

### Task 11: Comparison Experiment (`code/experiment/comparison.py`)

- **10 experimental conditions** defined:
  1. Uncoupled baseline (Adam only, no embedding)
  2. Linear + coupling
  3. Random + coupling
  4. Spectral + coupling
  5. LayeredClustered + coupling
  6. Correlation + coupling
  7. Developmental + coupling
  8. Adversarial + coupling
  9. Differentiable (jointly trained)
  10. Differentiable + coupling

- Runs on both MNIST and TopographicTask
- 3 seeds per condition (42, 123, 456)
- Uses `ExperimentRunner.run_comparison()` for orchestration
- Saves results to `results/comparison_results.csv`
- Generates `results/embedding_vs_performance.png`
- Data-dependent embeddings use reduced parameters for tractability:
  - correlation: n_batches=5, subsample_size=500
  - adversarial: n_batches=5, subsample_size=500
  - developmental: n_steps=100, subsample_pairs=5000

### Task 11.2: Property Tests (`code/tests/test_convergence_properties.py`)

- **Property 10**: Convergence detection — converged=True iff max relative change in final 20% < 5%
- **Property 15**: Temporal quality degradation — degraded=True iff min < 50% of initial
- All 5 property tests pass (100 examples each via Hypothesis)

### Task 12: Analysis Experiments

#### 12.1 Boundary condition test (`code/experiment/boundary.py`)
- Pearson correlation between quality scores and performance deltas
- Scatter plot with regression line
- Saves to `results/boundary_condition.csv` and `results/boundary_regression.png`

#### 12.2 Convergence analysis (`code/experiment/convergence.py`)
- Runs developmental embedding with quality tracking
- `detect_convergence()` function implements the convergence criterion
- Plots trajectory, reports convergence status
- Saves to `results/developmental_convergence.csv` and `results/developmental_trajectory.png`

#### 12.3 Three-point validation (in `code/experiment/boundary.py`)
- Extracts adversarial, random, and best embedding deltas
- Verifies monotonicity: adversarial_delta < random_delta < best_delta
- Saves to `results/three_point_validation.csv` and `results/three_point_curve.png`

#### 12.4 Temporal quality tracking (`code/experiment/temporal.py`)
- Tracks quality at intervals during training for fixed embeddings (linear, random, spectral)
- Reports degradation status per method
- Saves to `results/temporal_quality.csv` and `results/temporal_quality_trajectories.png`

#### 12.5 Spatial coherence test (`code/experiment/spatial_coherence_test.py`)
- Trains with and without coupling using spectral embedding
- Compares coherence scores
- Saves to `results/spatial_coherence.csv` and `results/spatial_coherence_comparison.png`

### Task 13: Wiring

#### 13.1 Main runner script (`code/scripts/run_all_experiments.py`)
- Orchestrates all 6 experiments in sequence
- Generates `results/summary.md` with key findings
- Run with: `.venv/bin/python steps/01-spatial-embedding/code/scripts/run_all_experiments.py`

#### 13.2 Updated `code/embeddings/__init__.py`
- Added `get_all_strategies()` convenience function returning all 8 embedding instances
- Updated `code/experiment/__init__.py` to export all new modules

### Smoke Test (`code/scripts/smoke_test.py`)
- Runs 2 conditions × 1 seed × 2 epochs to verify pipeline end-to-end
- Completes in ~27 seconds
- Validates: data loading, embedding, comparison, boundary analysis, convergence detection, temporal tracking

## Test Results

```
38 passed in 96.66s
```

All existing tests continue to pass. The 5 new property tests (Properties 10 and 15) pass with 100 examples each.

## Smoke Test Output

```
SMOKE TEST PASSED in 27.4s
- 8 strategies loaded via get_all_strategies()
- Baseline accuracy: 0.9697 (2 epochs)
- Linear+coupling accuracy: 0.9698 (2 epochs)
- Convergence detection works on synthetic trajectory
- Temporal quality tracking works for 3 methods
```

## Design Decisions

1. **Reduced parameters for data-dependent embeddings**: Correlation, developmental, and adversarial embeddings use smaller subsample sizes and fewer steps to keep total experiment runtime reasonable (~30-60 min for full run).

2. **Spectral embedding for coherence test**: Used spectral embedding as the "good embedding" for the spatial coherence test since it's structure-preserving and doesn't require data.

3. **Separate temporal tracking module**: Created `temporal.py` as a standalone experiment module rather than embedding it in the comparison, for cleaner separation of concerns.

4. **Three-point validation in boundary.py**: Kept the three-point validation in the same module as boundary condition since they share the same input data (comparison results) and conceptually test the same hypothesis.

## How to Run

```bash
# Smoke test (fast, ~30s)
.venv/bin/python steps/01-spatial-embedding/code/scripts/smoke_test.py

# Full experiment suite (slow, ~30-60 min)
.venv/bin/python steps/01-spatial-embedding/code/scripts/run_all_experiments.py

# Property tests only
.venv/bin/pytest steps/01-spatial-embedding/code/tests/test_convergence_properties.py -v

# All tests
.venv/bin/pytest steps/01-spatial-embedding/code/tests/ -v
```
