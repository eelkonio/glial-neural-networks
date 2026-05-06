# Spatial Embedding Experiment Results Summary

## General

### Step 01 v1 Results & Insights

This was the first run of the spatial embedding experiments. All quality scores were indistinguishable from zero — an informative null result. The experiment conditions (easy task MNIST, short training 10 epochs, aggressive subsampling 500 weights) didn't give spatial coupling room to differentiate from noise.

**Key insight**: The experiment was correctly designed but the operating conditions were too easy for spatial mechanisms to matter. This led to the v2 run with harder conditions (FashionMNIST, 50 epochs, 5000 subsample).

**Outcome**: Inconclusive — rerun needed with harder conditions. See `steps/01-spatial-embedding-v2/` for the definitive result.

---

**Total experiment time**: 1391.3 seconds (23.2 minutes)  
**Date**: 2026-05-03  
**Hardware**: MacBook Pro M4 Pro, 24GB, MPS GPU

---

## 1. Comparison Experiment

### MNIST Results

| Condition | Mean Accuracy | Std | Quality Score |
|-----------|--------------|-----|---------------|
| uncoupled_baseline | 0.9796 | 0.0011 | 0.0000 |
| linear_coupled | 0.9775 | 0.0005 | 0.0021 |
| random_coupled | 0.9775 | 0.0005 | -0.0026 |
| spectral_coupled | 0.9775 | 0.0005 | -0.0006 |
| layered_clustered_coupled | 0.9775 | 0.0005 | -0.0007 |
| correlation_coupled | 0.9792 | 0.0017 | -0.0011 |
| developmental_coupled | 0.9792 | 0.0017 | 0.0009 |
| adversarial_coupled | 0.9792 | 0.0017 | -0.0006 |
| differentiable | 0.9796 | 0.0011 | -0.0017 |
| differentiable_coupled | 0.9775 | 0.0005 | 0.0024 |

### TopographicTask Results

| Condition | Mean Accuracy | Std | Quality Score |
|-----------|--------------|-----|---------------|
| uncoupled_baseline | 0.9995 | 0.0004 | 0.0000 |
| linear_coupled | 0.9995 | 0.0004 | -0.0031 |
| random_coupled | 0.9995 | 0.0004 | -0.0014 |
| spectral_coupled | 0.9995 | 0.0004 | 0.0005 |
| layered_clustered_coupled | 0.9995 | 0.0004 | -0.0039 |
| correlation_coupled | 0.9998 | 0.0002 | -0.0051 |
| developmental_coupled | 0.9998 | 0.0002 | 0.0002 |
| adversarial_coupled | 0.9998 | 0.0002 | -0.0012 |
| differentiable | 0.9995 | 0.0004 | 0.0012 |
| differentiable_coupled | 0.9995 | 0.0004 | 0.0008 |

## 2. Boundary Condition Test

- **Pearson correlation (quality → performance)**: r = -0.2755, p = 0.4730

## 3. Three-Point Validation

- **Adversarial delta**: -0.0004
- **Random delta**: -0.0022
- **Best delta**: 0.0000
- **Monotonic (adversarial < random < best)**: False

## 4. Developmental Convergence

- **Converged**: False
- **Final quality**: -0.0322
- **Steps to stability**: None

## 5. Temporal Quality Tracking

| Method | Initial Quality | Final Quality | Degraded |
|--------|----------------|---------------|----------|
| linear | -0.0014 | -0.0026 | True |
| random | -0.0006 | 0.0011 | True |
| spectral | 0.0046 | -0.0049 | True |

## 6. Spatial Coherence Test

- **Coupled coherence**: -0.0014
- **Uncoupled coherence**: -0.0008
- **Mechanism confirmed (coupled > uncoupled)**: False

---

## Interpretation of Results

### The Central Finding: No Signal

The most important observation across all experiments is that **the quality scores are indistinguishable from zero** (all in the range [-0.006, +0.005]). This means none of the embedding strategies produce a meaningful correlation between spatial distance and gradient correlation at this scale and with these parameters.

This is not a failure of the code — it's an informative null result. Here's what it tells us:

### Why the Quality Scores Are Near Zero

1. **The MLP is too small and the task too easy.** MNIST achieves 97.9% accuracy in 10 epochs. The topographic task achieves 99.95%. Both tasks are essentially solved — the network has very little remaining gradient structure to exploit. When a network is near-converged, gradients become small and noisy, making gradient correlations between weight pairs unreliable.

2. **The subsampling is too aggressive.** With 268,800 weights, we subsample to 500-5000 weights for MDS and use only 5-20 batches for gradient correlation. This may be insufficient to detect weak but real spatial structure. The signal-to-noise ratio of the quality metric at these sample sizes may be too low.

3. **10 epochs of training may not be enough for spatial structure to matter.** Spatial coupling affects the *trajectory* of learning, not the final accuracy on easy tasks. On a task where Adam already converges quickly, spatial coupling has no room to help.

### What the Three-Point Validation Shows

The three-point curve is NOT monotonic: adversarial (-0.0004) > random (-0.0022) > best (0.0000). All deltas are within noise (< 0.3% accuracy difference). This means:

- Spatial coupling neither helps nor hurts at this scale
- The adversarial embedding doesn't hurt (because the quality scores are zero — there's no spatial structure to anti-correlate with)
- We cannot distinguish the strong claim from the weak claim because **neither claim produces a measurable effect**

### What the Developmental Convergence Shows

The developmental embedding did NOT converge (final quality = -0.032). This suggests:
- The force-based position optimization doesn't find meaningful structure in 100 steps
- The gradient correlations themselves may be too noisy at 5 batches to drive useful forces
- The chicken-and-egg problem may be real: without meaningful spatial structure, the forces are random, and random forces don't produce meaningful structure

### What the Temporal Tracking Shows

All three fixed embeddings show "degradation" — but this is misleading. The initial quality scores are already near zero (spectral starts at 0.0046, the highest). "Degradation" from 0.005 to -0.005 is fluctuation around zero, not meaningful degradation of a good embedding.

### What the Spatial Coherence Shows

Coupled coherence (-0.0014) ≈ uncoupled coherence (-0.0008). Both are near zero. Spatial coupling does not produce spatially organized weight structure — but this is expected when the coupling itself has no meaningful spatial structure to work with.

---

## Diagnosis: Why This Happened

The experiment is correctly designed but the **operating conditions are too easy** for spatial mechanisms to matter. This was predicted by Critical Review 3:

> "Phase 1 may show weak glial benefits by design. Backpropagation already solves the global credit assignment problem."

> "MNIST and CIFAR-10 don't have strong spatial locality structure in their computational graphs."

The specific issues:

1. **Task saturation**: Both tasks are solved to >97.9% accuracy. There's no room for spatial coupling to improve.
2. **Insufficient gradient signal**: With only 5-20 batches for correlation estimation, the gradient correlation signal is dominated by noise.
3. **Scale mismatch**: 268,800 weights with 500-weight subsamples means we're measuring correlations on 0.2% of the network.
4. **Training duration**: 10 epochs is enough to solve MNIST but not enough for spatial structure to differentiate from noise.

---

## What These Results Mean for Next Steps

### The Go/No-Go Gate (Step 01b)

The Step 01b gate requires:
1. ✗ Three-point curve is monotonic — **NOT MET** (all deltas are noise)
2. ✗ Benefit is not purely regularization — **CANNOT DETERMINE** (no benefit observed at all)
3. ✗ Embedding quality predicts benefit (r > 0.3) — **NOT MET** (r = -0.28, p = 0.47, not significant)

**However, this does NOT mean the spatial structure hypothesis is wrong.** It means the experiment was run under conditions where the hypothesis cannot be tested. The gate criteria should be re-evaluated under harder conditions.

### Recommended Actions Before Proceeding to Step 02

1. **Increase task difficulty**: Use CIFAR-10 (harder, ~93% accuracy ceiling with this MLP) or a larger topographic task with more classes and less separation. The task must NOT be solvable to >99% in 10 epochs.

2. **Increase gradient signal**: Use 50+ batches for gradient correlation (as the design spec originally specified — we reduced to 5 for speed). The quality metric needs stable gradient statistics.

3. **Increase training duration**: Train for 50-100 epochs so that spatial coupling has time to differentiate trajectories. 10 epochs is too short for a mechanism that operates on learning dynamics.

4. **Increase subsample size**: Use 5000+ weights for MDS (as originally specified) and 100K+ pairs for quality measurement. The current 500-weight subsamples are too small.

5. **Use a harder architecture**: A deeper or narrower network where optimization is harder (e.g., 4 layers of 64 units) would give spatial coupling more room to help.

### Alternative Interpretation

There is a possibility that the spatial structure hypothesis is simply wrong for backpropagation on standard tasks — that spatial coupling provides no benefit when backprop already computes exact gradients globally. This would be consistent with Critical Review 3's observation:

> "Backpropagation computes exact per-weight gradients globally. It does not need spatial locality."

If this is the case, the framework's value lies in Phase 2 (local learning rules) where glia are constitutive to the learning rule, not merely modulatory. The research plan already accounts for this possibility.

### Concrete Next Run

Before declaring the gate failed, re-run with:
- CIFAR-10 instead of MNIST
- 50 batches for gradient correlation (not 5)
- 50 epochs of training (not 10)
- 5000 subsample size for MDS (not 500)
- A 4×128 MLP (harder optimization landscape)

If this still shows no signal, the hypothesis is likely wrong for backprop and we should proceed directly to Phase 2.

---

## Output Files

- `comparison_results.csv` — Full comparison data (10 conditions × 2 tasks × 3 seeds)
- `boundary_condition.csv` — Quality vs performance data
- `three_point_validation.csv` — Three-point validation data
- `developmental_convergence.csv` — Convergence trajectory
- `temporal_quality.csv` — Quality over training time
- `spatial_coherence.csv` — Coherence comparison
- `embedding_vs_performance.png` — Quality vs performance scatter
- `boundary_regression.png` — Regression plot
- `three_point_curve.png` — Three-point validation curve
- `developmental_trajectory.png` — Convergence trajectory plot
- `temporal_quality_trajectories.png` — Temporal quality plot
- `spatial_coherence_comparison.png` — Coherence comparison plot
