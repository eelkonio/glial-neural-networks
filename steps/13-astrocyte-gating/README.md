# Step 13: Astrocyte D-Serine Gating

## Purpose

Implements three variants of an astrocyte-derived gating signal as a drop-in replacement for Step 12's `ThirdFactorInterface`. The astrocyte gate transforms the three-factor learning rule from a weak local rule (~10% accuracy) into a competitive algorithm by providing spatially-local directional credit assignment.

## Gate Variants

| Variant | Signal Type | Credit Assignment |
|---------|------------|-------------------|
| A (Binary) | 0.0 or 1.0 | None — gates plasticity only |
| B (Directional) | signed float | Activity prediction error per domain |
| C (Volume Teaching) | signed float | Spatially-diffused error from label targets |

## Architecture

- **Network**: 784→128→128→128→128→10 (LocalMLP, detached forward)
- **Dataset**: FashionMNIST, 50 epochs, 3 seeds
- **Hardware**: MPS GPU on M4 Pro

## How to Run

```bash
# Run all tests
.venv/bin/pytest steps/13-astrocyte-gating/code/tests/ -v

# Run property tests only
.venv/bin/pytest steps/13-astrocyte-gating/code/tests/ -m "property" -v

# Run experiments (individual condition)
.venv/bin/python steps/13-astrocyte-gating/code/experiment/run_experiments.py --condition directional

# Run all experiments
.venv/bin/python steps/13-astrocyte-gating/code/experiment/run_experiments.py --all
```

## Interpreting Results

- Results are saved to `results/` with timestamped filenames
- `results/summary.md` contains the final analysis
- The **central prediction test** compares benefit under local rules vs backprop
- Ablation results show which components (calcium dynamics, spatial structure) matter

## Dependencies

- Step 12: ThreeFactorRule, ThirdFactorInterface, LocalMLP, data pipeline
- Step 01: SpectralEmbedding for spatial domain assignment
