# Step 01b: Theoretical Analysis — Why Spatial LR Coupling Might Help Backpropagation

```
SIMULATION FIDELITY: Level-independent (theoretical/analytical)
SIGNAL MODEL: N/A (analysis of the optimization dynamics)
NETWORK STATE DURING INFERENCE: N/A
GLIAL INTERACTION WITH SIGNALS: N/A
NOTE: This analysis informs all subsequent steps by establishing WHY
      spatial coupling could help, not just WHETHER it helps.
```

## Purpose

Critical Review 3 identified a fundamental gap: backpropagation computes exact per-weight gradients globally. It does not need spatial locality. So **why would spatially smoothing learning rates improve it?**

This document provides the theoretical analysis that makes downstream results interpretable. Without understanding the mechanism, positive results are unexplainable and negative results are uninformative.

## The Three Candidate Mechanisms

### Mechanism 1: Structured Regularization

Spatial smoothing of learning rates reduces the effective degrees of freedom in the optimization. Instead of N independent learning rates (one per weight), the system has fewer effective parameters — determined by the spatial correlation length of the smoothing kernel.

**Mathematical framing:**

```
Standard Adam: each weight has independent adaptive LR
  effective_dof = N (one per weight)

Spatially-coupled Adam: LRs are correlated within spatial neighborhoods
  effective_dof ≈ N / (domain_size)^d  where d = spatial dimension

This is analogous to:
  - Dropout (reduces effective capacity by randomly zeroing)
  - Weight decay (reduces effective capacity by penalizing magnitude)
  - Spatial smoothing (reduces effective capacity by correlating updates)
```

**Prediction**: If this is the mechanism, the benefit should:
- Scale with the ratio of effective_dof reduction
- Be equivalent to other regularizers at matched effective capacity
- Help more on tasks prone to overfitting
- NOT depend on embedding quality (any smoothing reduces dof)

**Test**: Compare spatial coupling to dropout/weight decay at matched effective capacity. If they perform identically, spatial coupling is "just regularization."

### Mechanism 2: Landscape Conditioning (Structured Preconditioning)

Spatial correlation of learning rates changes the effective curvature of the loss landscape. A weight update with spatially smoothed LRs is equivalent to:

```
delta_w = P @ gradient

where P is a preconditioning matrix with structure:
  P_ij = f(spatial_distance(i, j))
```

This is a spatially structured preconditioner. The question is: does this structure approximate something useful?

**Connection to KFAC:**

KFAC (Kronecker-Factored Approximate Curvature) preconditions the gradient using the network's Fisher information matrix, approximated as a Kronecker product of layer-wise statistics:

```
KFAC: P ≈ (A ⊗ G)^{-1}
  where A = input activation covariance
        G = output gradient covariance
```

The spatial preconditioner has a different structure:
```
Spatial: P_ij = kernel(||pos_i - pos_j||)
  where kernel is determined by the diffusion coefficient D
```

**Key question**: Under what conditions does the spatial preconditioner approximate the Fisher information? If the embedding places weights with correlated Fisher information entries spatially close, then spatial smoothing approximates natural gradient descent.

**Prediction**: If this is the mechanism, the benefit should:
- Depend strongly on embedding quality (good embedding ≈ good preconditioner)
- Be comparable to KFAC when the embedding is optimal
- Improve conditioning number of the effective Hessian
- Help more on ill-conditioned problems

**Test**: Compare spatial coupling to KFAC. Measure the conditioning number of the effective Hessian under both. If spatial coupling improves conditioning comparably to KFAC, it's acting as a preconditioner.

### Mechanism 3: Gradient Noise Reduction

Mini-batch gradients are noisy estimates of the true gradient. Averaging learning rates with spatial neighbors is equivalent to a form of gradient smoothing — if spatially close weights have correlated true gradients, averaging their adaptive LRs reduces the noise in the effective update direction.

**Mathematical framing:**

```
Noisy gradient: g_i = true_gradient_i + noise_i
Adam LR: lr_i = f(history of g_i)  — noisy because g_i is noisy

Spatial averaging: effective_lr_i = mean(lr_j for j in neighbors(i))
  If neighbors have correlated true gradients:
    noise in effective_lr is reduced by factor ~1/sqrt(k)
    signal is preserved (correlated signals don't cancel)
```

**Prediction**: If this is the mechanism, the benefit should:
- Depend on embedding quality (neighbors must have correlated gradients)
- Scale with k (more neighbors = more noise reduction)
- Help more with small batch sizes (more gradient noise)
- Disappear with large batch sizes (less noise to reduce)

**Test**: Sweep batch size. If spatial coupling helps more at small batch sizes and the benefit vanishes at large batch sizes, it's noise reduction.

## Experiment 01b.1: Distinguishing the Three Mechanisms

### Protocol

Train the same MLP (from Step 01) under conditions designed to distinguish the three mechanisms:

| Condition | Tests | Expected if mechanism is... |
|-----------|-------|---------------------------|
| Spatial coupling + good embedding | All three | Helps (all mechanisms predict this) |
| Spatial coupling + random embedding | Regularization only | Helps if regularization, neutral if preconditioning/noise |
| Spatial coupling + adversarial embedding | None (should hurt) | Hurts (all mechanisms predict this) |
| Dropout at matched effective capacity | Regularization equivalence | Same benefit if regularization is the mechanism |
| KFAC preconditioner | Preconditioning equivalence | Same benefit if preconditioning is the mechanism |
| Large batch size (4096) + spatial coupling | Noise reduction test | No benefit if noise reduction is the mechanism |
| Small batch size (16) + spatial coupling | Noise reduction test | Large benefit if noise reduction is the mechanism |

### Interpretation Matrix

| Observation | Implies |
|-------------|---------|
| Random embedding helps as much as good embedding | Regularization (weak claim) |
| Good embedding helps, random doesn't | Preconditioning or noise reduction (strong claim) |
| Benefit disappears at large batch size | Noise reduction |
| Benefit persists at large batch size | Preconditioning or regularization |
| Spatial coupling ≈ KFAC performance | Preconditioning |
| Spatial coupling ≈ dropout performance | Regularization |
| Spatial coupling > both KFAC and dropout | Novel mechanism (or combination) |

## Experiment 01b.2: Fisher Information Structure Analysis

### The Question

Does the spatial embedding predict the structure of the Fisher information matrix? If spatially close weights have correlated Fisher information entries, then spatial smoothing is implicitly approximating natural gradient descent.

### Protocol

1. Train the baseline MLP to convergence
2. Compute the empirical Fisher information matrix (or its diagonal approximation)
3. For each embedding method, compute:
   - Correlation between spatial distance and Fisher information similarity
   - Compare to the gradient correlation metric from Step 01

### Expected Result

If the Fisher correlation is high for good embeddings and low for random embeddings, it confirms the preconditioning interpretation. The spatial embedding is useful precisely because it captures the structure of the loss landscape's curvature.

## Experiment 01b.3: Effective Hessian Conditioning

### The Question

Does spatial LR coupling improve the conditioning of the effective Hessian? If so, it's acting as a preconditioner regardless of the specific mechanism.

### Protocol

1. Compute the top eigenvalues of the Hessian (using Lanczos iteration) for:
   - Standard Adam
   - Spatially-coupled Adam (good embedding)
   - Spatially-coupled Adam (random embedding)
   - KFAC
2. Compute the condition number (ratio of largest to smallest eigenvalue)
3. Compare conditioning across methods

### Expected Result

If spatial coupling with a good embedding reduces the condition number comparably to KFAC, the preconditioning interpretation is confirmed.

## Go/No-Go Gate: Step 01 → Step 02

**This analysis serves as the gate between Step 01 and Step 02.** Before proceeding to the full modulation field (Step 02), the following must be established:

### Mandatory Criteria (ALL must be met)

1. **Three-point curve is monotonic**: Adversarial embedding hurts performance, random is neutral (or slight regularization benefit), good embedding helps. If this fails, spatial structure is not the mechanism and the modulation field approach needs fundamental rethinking.

2. **Benefit is not purely regularization**: At least one of the following must hold:
   - Good embedding helps significantly more than random embedding (rules out pure regularization)
   - Spatial coupling outperforms dropout at matched effective capacity
   - Benefit persists at large batch sizes (rules out pure noise reduction)

3. **Embedding quality predicts benefit**: Pearson correlation between embedding quality score and performance delta must be > 0.3 (with p < 0.05).

### Informational Criteria (guide Step 02 design, not blocking)

4. **Mechanism identification**: Which of the three mechanisms dominates? This determines how to design the modulation field:
   - If preconditioning: optimize the PDE to approximate the Fisher information
   - If noise reduction: optimize the PDE for gradient smoothing
   - If regularization: the PDE is overkill; simpler smoothing suffices

5. **KFAC comparison**: How does spatial coupling compare to KFAC? If KFAC dominates, the modulation field must add something beyond preconditioning (e.g., temporal adaptation, source-driven dynamics).

### If the Gate Fails

If the three-point curve is NOT monotonic (random embedding helps as much as good embedding):
- The spatial structure hypothesis is not supported for this task/architecture
- Options:
  a. Try the topographic task (where spatial structure should matter more)
  b. Reframe the modulation field as a temporal regularizer (not spatial)
  c. Skip to Phase 2 (local learning rules) where glia are constitutive, not modulatory

## Success Criteria

- At least one mechanism is identified as dominant (not "all three contribute equally")
- The three-point validation curve is confirmed monotonic
- The relationship between embedding quality and Fisher information structure is quantified
- A clear recommendation for Step 02's design is produced (optimize for preconditioning vs. regularization vs. noise reduction)

## Deliverables

- `src/fisher_analysis.py`: Fisher information computation and comparison
- `src/hessian_analysis.py`: Effective Hessian conditioning measurement
- `experiments/mechanism_discrimination.py`: The 7-condition experiment
- `experiments/fisher_structure.py`: Fisher-embedding correlation analysis
- `results/mechanism_discrimination.csv`: Results of all conditions
- `results/fisher_correlation.csv`: Fisher-embedding correlation data
- `results/go_no_go_assessment.md`: Formal assessment of gate criteria
- `results/step02_design_recommendation.md`: Which mechanism to optimize for

## Estimated Timeline

1-2 weeks. Mostly analytical with targeted experiments. The Fisher information computation is the most expensive part (requires multiple forward passes for the empirical Fisher).

## Connection to Critical Review 3

This step directly addresses Gap 2: "No mechanistic theory for why spatial coupling helps backpropagation." It also formalizes the Step 01b recommendation: "Consider inserting a three-experiment mini-phase before proceeding to Step 02."

The review noted: "Without understanding the mechanism, you won't know how to respond to the 'just make it bigger' comparison (Step 11.6)." This analysis provides that theoretical ammunition.
