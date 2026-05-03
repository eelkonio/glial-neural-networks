# Step 01: Spatial Embedding Experiments

## Purpose

Assign spatial coordinates to every weight in a neural network and validate
whether that spatial structure carries useful information for optimization.
This is the universal dependency for all subsequent research steps.

## Core Output

A positions array of shape `(N_weights, 3)` — every weight gets a 3D coordinate.

## How to Run

```bash
# From project root, activate venv
source .venv/bin/activate

# Run all tests
pytest steps/01-spatial-embedding/code/tests/

# Run the full comparison experiment
python -m code.experiment.comparison

# Run individual analyses
python -m code.experiment.boundary
python -m code.experiment.convergence
```

## How to Interpret Results

Results are written to `results/`:
- `comparison_results.csv` — all conditions × seeds × metrics
- `embedding_quality.csv` — quality scores per embedding method
- `three_point_validation.csv` — adversarial → random → good curve
- `summary.md` — key findings and implications for Step 02

The critical result is the **three-point validation curve**: if it's monotonic
(adversarial hurts, random neutral, good helps), the spatial structure
hypothesis is supported and we proceed to Step 02.

## Directory Structure

```
docs/       — design decisions and reasoning
code/       — all implementation code
data/       — model checkpoints, cached embeddings
results/    — CSV files, plots, metadata, summaries
```
