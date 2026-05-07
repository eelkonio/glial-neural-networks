# Future Experiment Ideas

A living document collecting experiment ideas that arise during the research process. These are not yet scheduled or specified — they represent promising directions identified through theoretical analysis, experimental results, or discussion.

---

## 1. Multi-Dimensional Embedding Ablation

**Origin**: Discussion during Step 14 design (2026-05-07)

**Question**: Does increasing the dimensionality of the spatial embedding (from 3D to 5D or 8D) improve the effectiveness of domain-level prediction and glial mechanisms?

**Hypothesis**: Higher-dimensional embeddings (4D-8D) will improve prediction quality because:
- Each domain has more neighbors → richer prediction context
- Chemical signals attenuate faster → sharper domain boundaries, less cross-talk
- More boundary neurons → better inter-domain communication

**Design**:
- Run the Step 14 predictive-BCM experiment with domain assignments computed from spectral embeddings in 3D, 4D, 5D, 6D, and 8D
- Same network architecture (784→128→128→128→128→10), same learning rule, same hyperparameters
- Only the spatial geometry changes (which neurons are grouped into which domains)
- Measure: prediction error convergence speed, final accuracy, domain-level prediction quality

**Estimated Time Cost**:

The dimensionality change affects only the *domain assignment* computation (done once at initialization) and the *distance calculations* used for spatial operations. It does NOT change the forward pass, the BCM computation, or the prediction error computation (which operate on domain activities regardless of how domains were assigned).

| Component | 3D baseline | Per additional dimension | Notes |
|-----------|:-----------:|:------------------------:|-------|
| Spectral embedding computation | ~2s | +0.5s per dim | One-time cost at init. SVD on (128×128) gram matrix, extract d eigenvectors instead of 3. |
| Domain assignment | ~0.1s | ~0s | Same algorithm regardless of d — just partition the d-dimensional coordinates. |
| Training loop (per epoch) | ~2s | ~0s | **No change.** Domain activities are computed the same way regardless of how domains were assigned. |
| Full experiment (50 epochs × 3 seeds × 6 conditions) | ~5 hours | ~0 additional | The embedding dimension doesn't affect training time. |

**Conclusion: Adding 4D-8D experiments costs essentially zero additional training time.** The only cost is running the experiment multiple times with different domain assignments — which is equivalent to adding more "conditions" to the experiment. If we add 5 dimensionality conditions (3D, 4D, 5D, 6D, 8D), the total experiment time increases by ~5× (from 5 hours to ~25 hours) — but this is because we're running 5× more conditions, not because higher dimensions are more expensive per-run.

The per-epoch training cost is identical across all dimensionalities because:
- The forward pass doesn't use spatial coordinates
- The BCM rule operates on domain activities (same shape regardless of how domains were assigned)
- The prediction matrices are (n_domains × n_domains) = (8×8) regardless of embedding dimension
- Only the *assignment* of neurons to domains changes — and that's computed once at initialization

**Priority**: Medium. Should be tested within Step 14 as an ablation, but only after the base predictive-BCM mechanism is shown to work (accuracy > 10%). If the base mechanism doesn't work, dimensionality won't save it.

**Interesting sub-question**: Does the *optimal domain size* change with dimensionality? In 3D with 12 neighbors, domain_size=16 gives 8 domains per layer. In 5D with 50 neighbors, maybe smaller domains (domain_size=8, giving 16 domains) would be better — more domains means more prediction dimensions, which might be useful when each domain has more neighbors to predict from.

---

## 2. Oscillatory Coordination of Prediction Windows

**Origin**: Biological plausibility discussion during Step 14 requirements (2026-05-07)

**Question**: Does adding theta-oscillation-like temporal windows (alternating "bottom-up processing" and "top-down prediction" phases) improve predictive coding?

**Hypothesis**: Separating the forward pass (bottom-up) from the prediction comparison (top-down) into distinct temporal phases might reduce interference and improve prediction quality.

**Design**:
- Even batches: forward pass only, collect activations, no weight updates
- Odd batches: prediction comparison, compute errors, apply updates
- Compare against the standard "everything happens simultaneously" approach

**Estimated Time Cost**: ~2× training time (half the batches are "observation only"). Could be reduced by using smaller observation windows.

**Priority**: Low. Interesting theoretically but unlikely to be the bottleneck. Test only if base mechanism works.

---

## 3. Dendritic Compartment Model

**Origin**: Biological plausibility analysis (2026-05-07)

**Question**: Does modeling separate basal (bottom-up) and apical (top-down) dendritic compartments per neuron improve the prediction error computation?

**Hypothesis**: Instead of computing prediction error as a simple subtraction, model each neuron as having two input streams that are compared internally. This might allow more nuanced error signals (e.g., the neuron can weight the comparison based on confidence).

**Design**:
- Each neuron has two "activation" values: basal (from forward input) and apical (from prediction)
- The learning signal is a function of both (not just their difference)
- The function could be learned or fixed (e.g., multiplicative gating)

**Estimated Time Cost**: ~1.5× per epoch (double the activation computation per neuron). Moderate.

**Priority**: Low-medium. Biologically interesting but adds complexity. Test after simpler approaches.

---

## 4. Reward-Modulated Theta (BCM Threshold Modulation)

**Origin**: Step 12b results analysis — alternative to predictive coding (2026-05-07)

**Question**: Instead of using prediction errors, can we inject task information by modulating the BCM theta threshold with a global reward signal?

**Hypothesis**: When reward is high → lower theta → more LTP → reinforce current representations. When reward is low → raise theta → more LTD → destabilize. This converts the undirected BCM signal into a reward-directed one.

**Design**:
- Compute global reward as (current_loss < previous_loss) → positive reward
- Modulate theta: theta_effective = theta × (1 - reward_strength × reward_signal)
- Compare against standard BCM (Step 12b baseline) and predictive-BCM

**Estimated Time Cost**: Negligible additional cost (one scalar multiplication per domain per step).

**Priority**: Medium. Simple to implement, could be tested alongside Step 14. If predictive coding works, this becomes less interesting. If predictive coding doesn't work, this is a simpler alternative worth trying.

---

## 5. Domain Size Sweep

**Origin**: Multi-dimensional embedding discussion (2026-05-07)

**Question**: What is the optimal domain size for the predictive-BCM rule? Does it interact with embedding dimensionality?

**Hypothesis**: Smaller domains (more domains per layer) provide finer-grained prediction but noisier signals. Larger domains provide more stable signals but coarser prediction. The optimum depends on the information content of the prediction error.

**Design**:
- Test domain_size ∈ {4, 8, 16, 32, 64} (giving 32, 16, 8, 4, 2 domains per 128-neuron layer)
- For each domain size, run the full predictive-BCM experiment
- Cross with dimensionality (3D, 5D) to test interaction

**Estimated Time Cost**: 5× conditions × ~1 hour each = ~5 hours additional.

**Priority**: Medium. Should be tested after the base mechanism works. Domain size is likely an important hyperparameter.

---

## 6. Scaling to Larger Networks

**Origin**: General research direction

**Question**: Do the glial mechanisms (BCM direction, domain prediction, competition) scale to larger networks (e.g., 784→512→512→512→10 or convolutional architectures)?

**Hypothesis**: Larger networks have more neurons per domain, potentially making domain-level statistics more stable. But they also have more parameters to learn, potentially requiring more epochs.

**Design**:
- Test on a wider network (512 hidden units instead of 128)
- Test on a deeper network (8 layers instead of 5)
- Test on a CNN (where spatial structure has natural meaning — nearby filters process nearby pixels)

**Estimated Time Cost**: 4-16× per epoch (depending on network size). A 512-unit network would take ~4× longer per epoch.

**Priority**: Low until base mechanism works on the small network. Then high — scaling is the key question for practical relevance.

---

## 7. Continual Learning with Domain Consolidation

**Origin**: Research plan Step 09 (topology-as-memory)

**Question**: Can the D-serine gating mechanism (driven by prediction error) naturally support continual learning by consolidating domains that have low prediction error (well-learned) and keeping domains with high error (new/changing) plastic?

**Hypothesis**: Domains that predict well → gate closed → no LTP → consolidated. Domains that predict poorly → gate open → active learning. When a new task arrives, it disrupts predictions in relevant domains → those domains become plastic again while others stay consolidated.

**Design**:
- Train on FashionMNIST task A (classes 0-4), then switch to task B (classes 5-9)
- Measure catastrophic forgetting with and without prediction-error-driven gating
- Compare against EWC and other continual learning baselines

**Estimated Time Cost**: ~2× a standard experiment (two training phases).

**Priority**: Low-medium. Fascinating direction but requires the base mechanism to work first. The prediction-error-driven gating is a natural fit for continual learning — it's essentially "learn where you're surprised, consolidate where you're not."

---

*Last updated: 2026-05-07*
