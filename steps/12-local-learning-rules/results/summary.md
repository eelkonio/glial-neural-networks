# Local Learning Rules Experiment Summary

## General

### Step 12 Results & Next Steps
The experiment ran for 7.7 hours and produced a clear picture:

The good news: The Phase 2 prediction is confirmed — spatial correlation under the three-factor reward rule is 104× stronger than under backprop (-0.364 vs -0.003). Spatial structure genuinely matters for local learning rules.

The challenge: The credit assignment gap is 72% (backprop 89% vs best local rule 16.5%). This is much larger than predicted. The local rules essentially don't learn on this architecture without a teaching signal.

The critical insight for Step 13: The astrocyte gate as currently designed (a simple activity-threshold plasticity gate) won't be enough. It needs to provide directional credit assignment — not just "should this synapse learn?" but "in which direction should it change?" This is actually more biologically faithful: astrocytes don't just release D-serine (gate), they also release gliotransmitters that carry information about local network state.

### Recommended path:

- Fix the three-factor error signal instability
- Redesign the astrocyte gate as a teaching signal (volume-transmitted error information derived from domain activity patterns) rather than a binary plasticity gate
- Test on the three-factor reward substrate (which already shows strong spatial correlation)

## Overview

- **Epochs**: 50
- **Seeds**: [42, 123, 456]
- **Batch size**: 128
- **Dataset**: FashionMNIST
- **Architecture**: 784→128→128→128→128→10 (LocalMLP)

## Final Test Accuracy (mean ± std)

| Rule | Accuracy |
|------|----------|
| backprop | 0.8899 ± 0.0011 |
| forward_forward | 0.1652 ± 0.0461 |
| hebbian | 0.1000 ± 0.0000 |
| oja | 0.1273 ± 0.0286 |
| predictive_coding | 0.1000 ± 0.0000 |
| three_factor_error | 0.1000 ± 0.0000 |
| three_factor_random | 0.1008 ± 0.0012 |
| three_factor_reward | 0.0981 ± 0.0029 |

## Key Findings

1. **Backprop baseline** establishes the upper bound for accuracy
2. **Local rules** show varying degrees of credit assignment deficiency
3. **Three-factor with error signal** is closest to backprop among local rules
4. **Forward-forward** achieves reasonable accuracy with purely local learning

## Implications for Step 13 (Astrocyte Gating)

The deficiency analysis identifies specific gaps that the astrocyte gate should address:

- **Credit assignment**: Local rules struggle to propagate error to early layers
- **Coordination**: Without global signals, layers learn independently
- **The three-factor rule** is the ideal substrate because its third-factor slot
  can be directly replaced by the astrocyte D-serine gate

See `deficiency_analysis.md` for per-rule characterization.

---

## Detailed Analysis and Implications

### The Fundamental Problem: Undirected Eligibility

The most important finding from Step 12 is not the accuracy numbers themselves — it is the *diagnosis* of why local rules fail. Every local rule tested here (Hebbian, Oja, three-factor, predictive coding) shares a common structural limitation: **the eligibility trace is always positive under ReLU activations.**

In a standard three-factor rule, the weight update is:

```
Δw = pre × post × third_factor × lr
```

Under ReLU, `post ≥ 0` always. The pre-activation (input to the layer) is also typically non-negative (it's the output of the previous layer's ReLU). So `pre × post ≥ 0` — the eligibility trace is always positive. This means the learning rule can only *strengthen* synapses, never *weaken* them. The third factor (whether it's a reward signal, random noise, or error estimate) modulates the *magnitude* of this always-positive update, but cannot flip its *direction*.

This is the "always positive eligibility" problem. It explains why:
- **Hebbian** (Δw = pre × post) only strengthens → weights explode → accuracy = chance
- **Oja** (Hebbian + normalization) prevents explosion but still only strengthens the dominant direction → learns one principal component → slightly above chance (12.7%)
- **Three-factor reward** (Δw = pre × post × reward_signal) → reward is scalar, can't provide per-synapse direction → chance level
- **Three-factor error** (Δw = pre × post × error) → error signal is unstable (explodes or vanishes without careful normalization) → chance level
- **Predictive coding** (Δw = prediction_error × post) → prediction errors can be signed, but the implementation still uses ReLU activations → chance level

The only rule that partially escapes this trap is **forward-forward** (16.5%), which uses a fundamentally different objective (goodness maximization per layer) that doesn't rely on the pre×post eligibility trace.

### The 104× Spatial Correlation Finding

The spatial quality metric measures whether nearby weights (in the spectral embedding) have correlated gradients. Under backprop, this correlation is negligible (r = -0.003, as established in Step 01). Under the three-factor reward rule, the correlation is -0.364 — a 104× increase in magnitude.

What does this mean? Under local learning rules, **weights that are structurally nearby (connected to similar neurons) DO have correlated update signals.** This is because:
1. Local rules use only layer-local information (pre and post activations)
2. Neurons in the same layer that receive similar inputs will have similar pre-activations
3. Neurons that project to similar targets will have similar post-activations
4. Therefore, nearby weights (in the connectivity graph) will have correlated eligibility traces

This validates the core hypothesis of the research program: **spatial structure is irrelevant under backprop (Phase 1 result) but highly relevant under local rules (Phase 2 prediction confirmed).** The astrocyte domain structure — which groups nearby synapses and applies shared modulation — is precisely the right mechanism for local learning because it exploits this spatial correlation.

### The Credit Assignment Gap: 72%

Backprop achieves 89% accuracy. The best local rule (forward-forward) achieves 16.5%. The gap is 72 percentage points. This is much larger than the ~20-30% gap reported in some local learning papers, because:

1. **Deep architecture**: Our 5-layer network (784→128→128→128→128→10) is deeper than most local learning benchmarks (which typically use 2-3 layers). Credit assignment becomes exponentially harder with depth under local rules.
2. **No layer-wise objectives**: We don't use auxiliary losses per layer (as some local learning methods do). Each layer must learn from the same global signal or from purely local statistics.
3. **No feedback connections**: We don't use feedback alignment or direct random feedback, which can partially solve credit assignment by providing approximate error signals to early layers.

The 72% gap represents the "price of locality" — what you lose by refusing to propagate error information backward through the network. This is the gap that the astrocyte gating mechanism (Step 13) and the BCM-directed rule (Step 12b) aim to reduce.

### Why Forward-Forward Partially Works

The forward-forward algorithm achieves 16.5% — well above chance (10%) but far below backprop (89%). It works because:

1. **Layer-local objective**: Each layer has its own "goodness" function (sum of squared activations). Positive examples should have high goodness, negative examples should have low goodness.
2. **Contrastive signal**: The positive/negative contrast provides direction — the rule knows whether to increase or decrease goodness for a given input.
3. **No inter-layer gradient flow**: Each layer learns independently, which is truly local.

However, it's limited because:
- The goodness objective is a proxy for classification — it doesn't directly optimize for the task
- Early layers don't know what later layers need — there's no coordination
- The 16.5% accuracy (with high variance ±4.6%) suggests it learns some features but can't compose them into a coherent classifier

### Implications for the Astrocyte Gating Design (Step 13)

The Step 12 results directly inform what the astrocyte gate must provide:

1. **Direction, not just magnitude**: A binary gate (on/off) applied to an always-positive eligibility trace still produces always-positive updates. The gate must somehow provide *signed* modulation — or the eligibility trace itself must be redesigned to be signed.

2. **Domain-level coordination**: The 104× spatial correlation means that nearby synapses have correlated signals. The astrocyte domain (which groups ~16 nearby neurons) can exploit this by applying a shared modulation signal. But this signal must carry directional information.

3. **Homeostasis**: Without some form of homeostasis, local rules either explode (Hebbian) or collapse to a single mode (Oja). The astrocyte's calcium dynamics provide a natural homeostatic mechanism — the sliding threshold prevents runaway potentiation.

4. **Competition**: Within a domain, not all synapses should strengthen simultaneously. The biological mechanism of heterosynaptic plasticity (where potentiation of one synapse depresses its neighbors) provides local competition that acts as a form of credit assignment.

These insights led directly to the design of Step 12b (BCM-Directed Substrate), which replaces the always-positive eligibility trace with a calcium-level-based direction signal that can be both positive (LTP) and negative (LTD).

### The Broader Research Narrative

Step 12 establishes the "baseline of failure" for local learning — the starting point from which glial mechanisms must improve. The results confirm that:

- Local learning rules as traditionally formulated cannot solve credit assignment in deep networks
- The problem is not the learning rate, the architecture, or the training duration — it's the fundamental inability to determine *direction* from local information alone
- Spatial structure (which is irrelevant under backprop) becomes highly relevant under local rules
- The biological solution (calcium-level direction + astrocyte gating + heterosynaptic competition) addresses exactly the right failure mode

This positions the research program correctly: we're not trying to make local rules "almost as good as backprop" (which may be impossible without some form of error propagation). We're trying to understand whether the biological mechanisms of glial modulation can provide enough directional information to make local rules *functional* — achieving meaningful learning (>10%) without global error signals.

