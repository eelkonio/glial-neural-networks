# Step 12b: BCM-Directed Substrate

A biologically faithful local learning rule where **direction** (LTP vs LTD) comes from local calcium levels relative to a sliding threshold (BCM theory).

## Motivation

Step 12 showed that local rules achieve only ~10% accuracy (chance) because the eligibility trace (`pre × post`) is always positive under ReLU — it's undirected. Step 13's astrocyte gates modulate magnitude but can't fix an undirected signal.

The biological answer is that direction emerges from the **level** of postsynaptic calcium relative to a sliding threshold:
- Above theta → LTP (strengthen synapse)
- Below theta → LTD (weaken synapse)

The astrocyte's D-serine release gates whether synapses can reach the high-calcium state needed for LTP. Heterosynaptic competition (zero-centering within domains) provides local winner-take-all credit assignment.

## Key Components

- **BCMDirectedRule**: Core learning rule implementing `LocalLearningRule` protocol from Step 12
- **Sliding Theta**: EMA-based threshold that adapts to recent activity (BCM homeostasis)
- **D-Serine Gating**: Astrocyte calcium dynamics gate LTP eligibility per domain
- **Heterosynaptic Competition**: Zero-centering within domains for local credit assignment

## Dependencies

- Step 12: `LocalMLP`, `LayerState`, `LocalLearningRule`, data loaders
- Step 13: `CalciumDynamics`, `CalciumConfig`, `DomainAssignment`, `DomainConfig`

## Structure

```
code/           — Implementation modules
data/           — FashionMNIST (auto-downloaded)
docs/           — Design decisions
results/        — Experiment outputs
```
