# Design Decisions — Step 13: Astrocyte D-Serine Gating

## Decision Log

### 2024-01-XX: Domain Assignment Strategy

**Context**: Need to partition output neurons into astrocyte domains.

**Decision**: Use spectral ordering (first eigenvector of layer weight matrix) for spatial mode, with contiguous partitioning as fallback.

**Rationale**: Full k-means on Step 01 3D coordinates requires the model to have been trained with that embedding. For the 4-layer MLP, a simpler 1D spectral ordering of the weight matrix captures the essential spatial structure without external dependencies.

### 2024-01-XX: Stability Fix Approach

**Context**: Step 12 showed loss explosion with layer-wise error third factor.

**Decision**: Implement error clipping (±10.0) and eligibility trace normalization (norm > 100 → normalize to 1.0).

**Rationale**: These are minimal interventions that prevent numerical instability without changing the learning dynamics when values are in normal range.

### 2024-01-XX: Li-Rinzel Parameters

**Context**: Need biologically-inspired calcium dynamics that produce meaningful gating behavior.

**Decision**: Use standard Li-Rinzel parameters from computational neuroscience literature, with dt=0.01 for stability.

**Rationale**: These parameters produce calcium oscillations in the 0.1-2.0 μM range with sustained input, matching biological astrocyte behavior.
