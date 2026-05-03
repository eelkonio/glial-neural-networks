# Step 01 v2: Spatial Embedding Results — Comprehensive Report

**Date**: 2026-05-03  
**Total runtime**: 5945.8 seconds (99.1 minutes)  
**Hardware**: MacBook Pro M4 Pro, 24GB, MPS GPU  
**Task**: FashionMNIST (harder than MNIST, ~89% accuracy ceiling)  
**Architecture**: 784→128→128→128→128→10 (4 hidden layers, 150,784 weights)  
**Training**: 50 epochs, Adam lr=1e-3, batch_size=128  
**Seeds**: 42, 123, 456 (3 per condition)

---

## 1. Raw Results

### Aggregated Accuracy and Quality Scores

| Condition | Mean Acc | Std | Delta vs Baseline | Quality Score | Coherence |
|-----------|----------|-----|-------------------|---------------|-----------|
| uncoupled_baseline | 0.8899 | 0.0011 | — | 0.0000 | 0.0000 |
| linear_coupled | 0.8913 | 0.0012 | +0.0014 | -0.0035 | -0.0022 |
| random_coupled | 0.8913 | 0.0012 | +0.0014 | -0.0002 | +0.0028 |
| spectral_coupled | 0.8913 | 0.0012 | +0.0014 | -0.0045 | +0.0014 |
| layered_clustered_coupled | 0.8913 | 0.0012 | +0.0014 | -0.0078 | +0.0020 |
| correlation_coupled | 0.8909 | 0.0011 | +0.0009 | -0.0105 | -0.0009 |
| developmental_coupled | 0.8909 | 0.0011 | +0.0009 | -0.0002 | +0.0007 |
| adversarial_coupled | 0.8909 | 0.0011 | +0.0009 | +0.0004 | +0.0029 |
| differentiable | 0.8899 | 0.0011 | +0.0000 | +0.0004 | -0.0018 |
| differentiable_coupled | 0.8913 | 0.0012 | +0.0014 | -0.0005 | +0.0001 |

### Key Metrics

- **Boundary condition**: r = -0.2744, p = 0.4749 (not significant)
- **Three-point validation**: adversarial=+0.0009, random=+0.0014, best=+0.0014 — **NOT monotonic**
- **Spatial coherence**: coupled=0.0014, uncoupled=0.0000

---

## 2. Interpretation

### 2.1 The Accuracy Pattern: Coupling Provides a Tiny Uniform Boost

All coupled conditions achieve 0.8909-0.8913 accuracy vs the baseline's 0.8899. This is a consistent +0.1% improvement. However:

- **The improvement is identical regardless of embedding quality.** Linear, random, spectral, layered-clustered, adversarial, and differentiable-coupled all achieve the same 0.8913. The correlation and developmental conditions achieve 0.8909.
- **The adversarial embedding does NOT hurt.** It achieves the same accuracy as random coupling (+0.0009). This directly contradicts the spatial structure hypothesis.
- **The differentiable embedding without coupling matches baseline exactly** (0.8899). The spatial coherence loss alone provides no benefit.

This pattern is diagnostic: **the benefit comes from the coupling mechanism itself (spatial smoothing of learning rates), not from the spatial structure of the embedding.** Any embedding — even random, even adversarial — provides the same small benefit when coupled.

### 2.2 The Quality Scores: Negative and Near Zero

All quality scores are in the range [-0.0105, +0.0004]. The most notable pattern:

- **Correlation embedding has the most negative quality** (-0.0105): spatially close weights have *anti-correlated* gradients. This is the opposite of what a "good" embedding should produce.
- **Layered-clustered is also negative** (-0.0078): the layer-preserving structure doesn't align with gradient similarity.
- **Random and developmental are near zero** (~-0.0002): no spatial-gradient relationship.
- **Adversarial is slightly positive** (+0.0004): the anti-MDS construction barely works.

The negative quality scores for structured embeddings (spectral, layered-clustered, correlation) suggest that **the network's gradient structure does not respect the topological structure of the weight graph.** Weights that are "structurally similar" (connected to similar neurons) do NOT have correlated gradients in this architecture.

### 2.3 The Three-Point Validation: Failed

Expected: adversarial < random < best  
Observed: adversarial (+0.0009) < random (+0.0014) ≈ best (+0.0014)

The ordering is technically correct (adversarial < random ≤ best) but the differences are within noise (0.05% accuracy). The adversarial embedding doesn't hurt because:
1. The quality scores are all near zero — there's no meaningful spatial structure to anti-correlate with
2. The coupling benefit comes from LR smoothing, not from spatial structure

### 2.4 Spatial Coherence: Marginal

Coupled coherence (0.0014) vs uncoupled (0.0000). The difference is positive but tiny. Spatial coupling produces a barely detectable increase in spatial organization of weights — but this is likely an artifact of the smoothing itself, not evidence that the mechanism is working as intended.

---

## 3. Diagnosis: What This Tells Us

### 3.1 The Weak Claim is Supported, the Strong Claim is Not

Critical Review 3 distinguished two claims:
- **Weak claim**: "Spatially smoothed learning rates are a good regularizer" — **SUPPORTED**. All coupled conditions improve by ~0.1%, regardless of embedding quality.
- **Strong claim**: "Spatial locality captures functional structure, and glia exploit this" — **NOT SUPPORTED**. Embedding quality does not predict benefit. Adversarial embeddings don't hurt.

### 3.2 Why Spatial Structure Doesn't Matter Here

The fundamental issue: **in a fully-connected MLP trained with backpropagation, there is no inherent spatial structure in the gradient field.** 

- Every weight in a layer receives gradients from the same loss function via the same backward pass
- Gradient correlations between weights are determined by the data distribution and the network's current state, not by topological proximity
- The spectral embedding captures *connectivity* structure, but connectivity in a fully-connected network is trivial — every input neuron connects to every output neuron in each layer
- The correlation embedding captures *functional* similarity, but with only 50 batches of gradient signal, the correlation estimates are noisy

### 3.3 The Coupling Benefit is Pure Regularization

The +0.1% improvement from coupling is consistent with **structured regularization**:
- Averaging learning rates with neighbors reduces the effective degrees of freedom
- This is equivalent to a mild form of weight tying or structured dropout
- It helps slightly on FashionMNIST (which has some overfitting headroom) but doesn't depend on the embedding being meaningful

This confirms the mechanism identified in Step 01b's theoretical analysis: spatial coupling under backpropagation acts as a regularizer, not as a preconditioner or noise reducer.

---

## 4. Implications for the Research Program

### 4.1 The Go/No-Go Gate Assessment

The Step 01b gate criteria:

1. ✗ **Three-point curve is monotonic** — Technically yes (adversarial < random ≤ best) but differences are within noise. Not meaningfully monotonic.
2. ✗ **Benefit is not purely regularization** — The benefit IS purely regularization. Random embedding helps as much as any structured embedding.
3. ✗ **Embedding quality predicts benefit (r > 0.3)** — r = -0.27, p = 0.47. Not significant, wrong sign.

**The gate FAILS for the strong spatial structure claim under backpropagation.**

### 4.2 What This Means for Step 02 (Modulation Field)

The modulation field (reaction-diffusion PDE over spatial positions) is designed to exploit spatial structure. Since spatial structure doesn't carry meaningful information under backpropagation on this architecture:

- **The PDE dynamics will not outperform simple spatial smoothing** (as predicted by the Step 01b theoretical analysis)
- **The modulation field may still provide regularization benefit** (~0.1%) but this doesn't justify the computational overhead of a PDE solver
- **KFAC will likely outperform the modulation field** since it captures actual curvature structure, not arbitrary spatial structure

### 4.3 The Path Forward: Phase 2

The research plan explicitly anticipated this outcome:

> "If Phase 1 shows no benefit, Phase 2 is still worth pursuing — it's possible that glial mechanisms are specifically adapted to local learning and provide little benefit when backprop already solves the credit assignment problem globally."

The biological argument is strongest for Phase 2: **glia are constitutive to local learning rules, not merely modulatory of backpropagation.** Under local rules (STDP, three-factor), there IS no global gradient signal — the spatial structure of the glial system provides the "missing piece" (the third factor, the teaching signal) that makes local learning work.

### 4.4 Recommended Next Steps

**Option A: Skip to Phase 2 (recommended)**
- Implement local learning rules (Step 12)
- Test whether astrocyte gating makes local rules competitive (Step 13)
- The spatial embedding from Step 01 still provides the coordinate system — it just doesn't help under backprop

**Option B: Try a different architecture (optional)**
- CNNs have inherent spatial structure (nearby filters process nearby pixels)
- A CNN's weight space might have meaningful spatial-gradient correlations
- This would test whether the null result is architecture-specific

**Option C: Try a different coupling mechanism (optional)**
- Instead of LR smoothing, try gradient smoothing (average gradients with neighbors)
- Or try the meta-learner variant (field state as input to a learned update function)
- These might extract more value from spatial structure than simple LR averaging

---

## 5. What We Learned

1. **Spatial LR coupling provides ~0.1% accuracy improvement on FashionMNIST** — consistent, reproducible, but tiny and embedding-independent.

2. **No embedding strategy produces meaningful gradient-distance correlation** in a fully-connected MLP. Quality scores are all in [-0.01, +0.001].

3. **The spatial structure hypothesis is not supported under backpropagation** for fully-connected architectures. The benefit of coupling is regularization, not spatial exploitation.

4. **The adversarial embedding does not hurt** — confirming that spatial structure is irrelevant (not just unhelpful) under these conditions.

5. **The differentiable embedding (learnable positions) provides no benefit** — the spatial coherence loss doesn't find useful structure to optimize toward.

6. **The framework's value likely lies in Phase 2** (local learning rules) where glia play a constitutive role, not in Phase 1 (backpropagation) where they're merely modulatory.

---

## 6. Comparison: v1 vs v2

| Metric | v1 (MNIST, 10 epochs) | v2 (FashionMNIST, 50 epochs) |
|--------|----------------------|------------------------------|
| Baseline accuracy | 97.96% | 88.99% |
| Best coupled accuracy | 97.96% | 89.13% |
| Performance delta | 0.00% | +0.14% |
| Quality score range | [-0.006, +0.005] | [-0.011, +0.001] |
| Boundary r | -0.28 (p=0.47) | -0.27 (p=0.47) |
| Three-point monotonic | No | No (technically yes but within noise) |
| Conclusion | Null (too easy) | Weak regularization effect only |

The v2 results are more informative than v1: we now have a measurable (if tiny) coupling benefit and can definitively attribute it to regularization rather than spatial structure.

---

## Output Files

- `comparison_results.csv` — Full data (10 conditions × 3 seeds)
- `boundary_condition.csv` — Quality vs performance data
- `metadata.json` — Experiment configuration and environment
- `embedding_vs_performance.png` — Quality vs performance scatter
- `boundary_regression.png` — Regression plot (r=-0.27)
- `three_point_curve.png` — Three-point validation
- `spatial_coherence_comparison.png` — Coupled vs uncoupled
