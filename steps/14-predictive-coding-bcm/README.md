# Step 14: Predictive Coding + BCM

A local learning rule combining **BCM-directed signed updates** with **inter-layer domain-level prediction errors** as the task-relevant information channel for local learning.

## Motivation

Step 12b showed that BCM direction (LTP/LTD from calcium vs sliding threshold) provides signed updates, but the rule still lacks task-relevant information — it doesn't know *which* neurons should strengthen or weaken to improve the network's output. The missing piece is a local signal that tells each layer what it's "getting wrong."

The solution: each layer maintains a small (8×8) linear prediction of the next layer's **domain-level** activation. The prediction error (actual − predicted) is signed, local, and informative. It modulates the BCM direction signal, making weight updates task-relevant:

- **Surprised domains** (high prediction error) → active learning
- **Unsurprised domains** (low prediction error) → consolidation

## Key Components

- **PredictiveBCMRule**: Core learning rule combining BCM direction × domain information signal
- **Domain-Level Prediction**: Small (8×8) prediction matrices between adjacent layers
- **Surprise-Driven Calcium**: Prediction error drives D-serine gating (not raw activity)
- **Heterosynaptic Competition**: Zero-centering within domains for local credit assignment

## Dependencies

- Step 12: `LocalMLP`, `LayerState`, `LocalLearningRule`, data loaders
- Step 13: `CalciumDynamics`, `CalciumConfig`, `DomainAssignment`, `DomainConfig`
- Step 12b: `BCMDirectedRule`, `BCMConfig`

## Structure

```
code/           — Implementation modules
code/scripts/   — Experiment runner scripts
code/tests/     — Property-based and unit tests
data/           — FashionMNIST (auto-downloaded)
docs/           — Design decisions
results/        — Experiment outputs
```

## Quick Start

```bash
# Smoke test (5 epochs, 1 seed)
cd steps/14-predictive-coding-bcm
python -m code.scripts.run_quick

# Full experiment (50 epochs, 3 seeds, 6 conditions)
python -m code.scripts.run_experiment
```
