# Step 12: Local Learning Rules

## Purpose

Implements five local learning rules WITHOUT glia to establish baselines for Step 13 (astrocyte gating). Tests the Phase 2 prediction that spatial structure matters more under local learning rules than under backpropagation.

## Learning Rules

1. **Hebbian** — Δw = η · pre · post − λ · w (simplest baseline)
2. **Oja's Rule** — Δw = η · post · (pre − post · w) (self-normalizing PCA)
3. **Three-Factor** — Δw = eligibility_trace · third_factor · η (substrate for Step 13)
4. **Forward-Forward** — Per-layer goodness maximization (Hinton 2022)
5. **Predictive Coding** — Top-down predictions + local error minimization
6. **Backprop Baseline** — Standard backpropagation (upper bound reference)

## Architecture

All rules use the same MLP: 784 → 128 → 128 → 128 → 128 → 10 (FashionMNIST).

## Running Experiments

```bash
# Run all tests
.venv/bin/pytest steps/12-local-learning-rules/code/tests/ -v

# Run verification script
.venv/bin/python steps/12-local-learning-rules/code/scripts/verify_setup.py

# Run full experiment (all rules, 50 epochs, 3 seeds)
.venv/bin/python -m code.experiment.runner --all
```

## Interpreting Results

- `results/summary_table.csv` — Mean accuracy per rule
- `results/accuracy_comparison.png` — Bar chart comparison
- `results/convergence_curves.png` — Training curves
- `results/deficiency_analysis.md` — What each rule lacks vs backprop
- `results/spatial_quality.csv` — Spatial embedding quality per rule

## Key Finding (Expected)

Local rules will show a credit assignment gap vs backprop. The three-factor rule with layer-wise error should perform best among local rules. Step 13 will test whether astrocyte gating can close this gap.
