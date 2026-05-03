# Step 01b Go/No-Go Assessment

**Date**: 2026-05-03
**Runtime**: 1684.5s

## Gate Criteria Assessment

### Gate 1: Three-point curve is monotonic
**FAIL** — From Step 01 v2: adversarial embedding does not hurt performance.
All embeddings (including adversarial) provide the same ~0.14% benefit.
Spatial structure is irrelevant to the coupling benefit.

### Gate 2: Benefit is not purely regularization
**FAIL** — Mechanism discrimination shows:
- Baseline accuracy: 0.8856
- Spatial coupling (random): 0.8815
- Spatial coupling (spectral): 0.8815
- Dropout (0.15): 0.8894
- Weight decay (1e-4): 0.8819
- Weight decay (1e-3): 0.8759

Spatial coupling provides similar benefit to standard regularizers.
The benefit does not depend on embedding quality (random ≈ spectral).

### Gate 3: Embedding quality predicts benefit (r > 0.3)
**FAIL** — From Step 01 v2: r = -0.27, p = 0.47 (not significant).

## Mechanism Identification

**Dominant mechanism: STRUCTURED REGULARIZATION**

Evidence:
1. Random embedding helps as much as structured embeddings
2. Spatial coupling provides similar benefit to dropout/weight decay
3. Adversarial embedding does not hurt
4. Embedding quality does not predict benefit

## Batch Size Analysis

| Batch Size | Baseline | Coupled | Delta |
|-----------|----------|---------|-------|
| 16 | 0.8788 | 0.8871 | +0.0083 |
| 64 | 0.8891 | 0.8868 | -0.0023 |
| 128 | 0.8843 | 0.8830 | -0.0013 |
| 512 | 0.8720 | 0.8794 | +0.0074 |
| 2048 | 0.8676 | 0.8694 | +0.0018 |

## Fisher Information Structure

| Embedding | Fisher-Spatial Correlation | p-value |
|-----------|--------------------------|---------|
| linear | 0.0101 | 0.0013 |
| random | -0.0037 | 0.2432 |
| spectral | 0.0303 | 0.0000 |

## Verdict

**The gate FAILS on all three mandatory criteria.**

Spatial structure does not provide meaningful benefit under backpropagation
for fully-connected architectures. The small coupling benefit (~0.1-0.2%)
is pure regularization equivalent to dropout or weight decay.

## Recommendation

**Proceed to Phase 2 (local learning rules).** The biological argument
for glia is strongest under local rules where they provide the 'third factor'
that makes learning possible — not under backprop where global gradients
already solve credit assignment.

The spatial embedding infrastructure built in Step 01 remains valid —
it provides the coordinate system for Phase 2. The embedding just doesn't
help optimization under backprop.

## Alternative: Try CNNs (Optional)

CNNs have inherent spatial structure (nearby filters process nearby pixels).
The null result may be specific to fully-connected architectures where
weight-space has no natural geometry. A CNN experiment could test this.
However, Phase 2 is the higher-priority path.