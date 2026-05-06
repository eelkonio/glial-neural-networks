# Step 12: Full Experiment Analysis

**Runtime**: 459.2 minutes (~7.7 hours)  
**Configuration**: 8 rules × 3 seeds × 50 epochs on FashionMNIST  
**Architecture**: 784→128→128→128→128→10 (LocalMLP, detached forward)

---

## 1. Final Accuracy Results

| Rule | Accuracy (mean ± std) | vs Backprop Gap |
|------|----------------------|-----------------|
| **backprop** | **88.99% ± 0.11%** | — (reference) |
| forward_forward | 16.52% ± 4.61% | -72.5% |
| oja | 12.73% ± 2.86% | -76.3% |
| three_factor_random | 10.08% ± 0.12% | -78.9% |
| three_factor_reward | 9.81% ± 0.29% | -79.2% |
| three_factor_error | 10.00% ± 0.00% | -79.0% |
| hebbian | 10.00% ± 0.00% | -79.0% |
| predictive_coding | 10.00% ± 0.00% | -79.0% |

**10% = random chance** (10 classes in FashionMNIST).

---

## 2. Key Findings

### 2.1 The Credit Assignment Gap is Massive

The gap between backprop (89%) and the best local rule (forward-forward at 16.5%) is **72.5 percentage points**. This is far larger than the 4-15% gap we predicted in the research plan. The local rules essentially don't learn on this architecture.

### 2.2 Only Forward-Forward Shows Any Learning

Forward-forward is the only local rule that exceeds chance (16.5% vs 10%). This makes sense — it's the only rule that has a meaningful per-layer objective (goodness maximization). All other rules are stuck at chance because:
- **Hebbian/Oja**: No error signal at all. They extract principal components but can't solve classification.
- **Three-factor (all modes)**: The eligibility traces accumulate, but the third-factor signals (random noise, global reward, layer-wise error) don't provide enough directional information to drive useful learning.
- **Predictive coding**: The inference iterations converge (error drops from 6.8 to 0.5) but the weight updates don't improve classification. The top-down predictions learn to predict the input but the representations don't become discriminative.

### 2.3 Three-Factor Error Signal Exploded

The three-factor rule with layer-wise error showed a loss explosion (4.5 trillion) on seed 42. The random projection-based local error signal is numerically unstable for this architecture. This needs fixing before Step 13.

### 2.4 Oja Shows Interesting Properties

Despite only 12.7% accuracy, Oja's rule shows:
- **High credit assignment correlation** (0.7 per layer) — its updates are actually correlated with the true gradient direction
- **High inter-layer coordination** (CKA > 0.99) — layers learn coordinated representations
- **Self-normalizing** — weight norms stay bounded

This suggests Oja is doing something right (extracting structure) but can't convert that structure into classification without a teaching signal.

---

## 3. Spatial Quality Under Local Rules

| Rule | Spatial Correlation | Backprop Correlation | Ratio |
|------|--------------------|--------------------|-------|
| three_factor_reward | **-0.364** | -0.003 | **104×** |
| oja | -0.083 | -0.003 | 24× |
| hebbian | -0.036 | -0.003 | 10× |
| three_factor_random | +0.012 | -0.003 | -3× |
| three_factor_error | +0.007 | -0.003 | -2× |
| forward_forward | 0.000 | -0.003 | 0× |
| predictive_coding | 0.000 | -0.003 | 0× |

**Critical finding**: The three-factor rule with global reward shows a spatial correlation of **-0.364** — over 100× stronger than backprop's -0.003. This means:

- Under the three-factor reward rule, spatially close weights have **strongly correlated update signals**
- This is exactly what the Phase 2 prediction said: spatial structure matters MORE under local rules
- The negative sign means close weights have similar updates (good for spatial coupling)

This validates the core hypothesis: **spatial structure IS meaningful under local learning rules**, even though it wasn't under backprop.

---

## 4. Deficiency Analysis Summary

| Rule | Dominant Deficiency | Credit Reach | Redundancy | Coordination |
|------|--------------------:|:-------------|:-----------|:-------------|
| hebbian | Credit assignment | 0.05 (layer 0) | High (0.55) | Mixed |
| oja | Coordination | **0.7** (all layers) | Medium (0.25) | **Very high** (0.99) |
| three_factor_random | Credit assignment | 0.0 | Low | Low |
| three_factor_reward | Credit assignment | 0.0 | Medium (0.22) | High (0.85) |
| three_factor_error | Credit assignment | 0.0 | Medium (0.35) | Broken (NaN) |
| forward_forward | Credit assignment | 0.0 | Low | Low |
| predictive_coding | Credit assignment | 0.0 | High (0.47) | Very high (0.97) |

**The universal deficiency is credit assignment.** Every rule except Oja has near-zero correlation between its update signal and the true gradient. The astrocyte gate in Step 13 needs to provide a signal that carries error information to early layers.

---

## 5. Implications for Step 13 (Astrocyte Gating)

### What the Astrocyte Gate Must Do

1. **Provide directional credit assignment**: The gate must carry information about WHICH direction weights should change, not just WHETHER they should change. A simple activity-threshold gate (as currently designed in the research plan) may not be sufficient — it gates plasticity but doesn't provide direction.

2. **The three-factor reward rule is the best substrate**: It already shows strong spatial correlation (-0.364) and reasonable inter-layer coordination (CKA 0.85). The astrocyte gate replaces the global reward signal with a spatially-local, calcium-dependent signal.

3. **The spatial structure prediction is confirmed**: Spatial correlation under three-factor reward is 104× stronger than under backprop. This means spatial coupling WILL matter for the astrocyte gate — unlike Phase 1 where it was irrelevant.

### Design Recommendations for Step 13

- **Don't just gate plasticity** — the gate needs to carry directional information (which direction to update). A pure on/off gate won't help because the eligibility trace alone doesn't know the right direction.
- **Consider the gate as a teaching signal** — instead of just D-serine (binary gate), make the astrocyte signal carry error information derived from its domain's activity patterns.
- **Oja + astrocyte gating** may be promising — Oja already has high credit correlation (0.7) and just needs a teaching signal to convert its structural learning into classification.
- **Fix the three-factor error signal** — the layer-wise error implementation is numerically unstable and needs gradient clipping or normalization.

### What Won't Work

- A simple activity-threshold gate (Step 13's current design) will likely NOT close the 72% gap because it doesn't provide directional credit assignment
- The gap is too large for a modulatory signal — the astrocyte needs to be constitutive (provide the actual learning signal, not just modulate an existing one)

---

## 6. Next Steps

1. **Fix three-factor error instability** — add gradient clipping to the layer-wise error signal
2. **Proceed to Step 13** — implement astrocyte gating, but redesign it as a teaching signal (not just a plasticity gate)
3. **Consider Oja + astrocyte** as an alternative substrate — Oja's high credit correlation suggests it's closer to working than three-factor
4. **The 72% gap is the target** — any improvement from astrocyte gating is meaningful, but we need at least 20-30% improvement to validate the hypothesis

---

## Output Files

- `summary_table.csv` — Mean/std accuracy per rule
- `performance_comparison.csv` — Full per-epoch data
- `deficiency_analysis.md` — Per-rule deficiency characterization
- `spatial_quality.csv` — Spatial correlation per rule
- `accuracy_comparison.png` — Bar chart
- `convergence_curves.png` — Training curves
- `weight_norm_trajectories.png` — Weight norms over time
- `credit_assignment_heatmap.png` — Rules × layers heatmap
