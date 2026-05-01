# Step 01: Spatial Coordinate Assignment

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static
GLIAL INTERACTION WITH SIGNALS: Learning-only
NOTE: Spatial embedding is foundational and level-independent.
      The same embedding carries forward into Level 2 and Level 3 simulations.
```

## The Problem

Before any glial mechanism can operate, every weight in the network needs a position in simulated physical space. This is the most fundamental design decision in the entire framework, and the second critical review identified it as the most important open problem: the embedding determines whether locality bias helps or hurts.

## Why This Comes First

Every subsequent experiment depends on having a spatial embedding. If the embedding is arbitrary, all spatial mechanisms (diffusion, domains, patrol) operate on meaningless geometry. If the embedding is well-matched to the network's functional structure, spatial mechanisms gain their power. We need to understand this dependency before building anything on top of it.

## Experiment 1.1: Embedding Strategies

### Implementation

Build a small MLP (2 hidden layers, 256 units each) for MNIST classification. Implement multiple spatial embedding strategies and compare their effect on a simple spatially-coupled learning rate (precursor to full astrocyte system).

### Embedding Methods to Implement

**A. Naive linear mapping**
```python
def embed_linear(layer_idx, neuron_idx, weight_idx, total_layers, max_neurons):
    """Map weight indices directly to 3D coordinates."""
    x = layer_idx / total_layers
    y = neuron_idx / max_neurons
    z = weight_idx / max_neurons
    return (x, y, z)
```

**B. Random embedding**
```python
def embed_random(n_weights, seed=42):
    """Assign random 3D coordinates to each weight."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n_weights, 3))
```

**C. Topology-preserving embedding (spectral)**
```python
def embed_spectral(adjacency_matrix, dim=3):
    """Use spectral embedding of the network graph.
    Weights that connect neurons with similar connectivity
    patterns end up spatially close."""
    laplacian = compute_laplacian(adjacency_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    # Use smallest non-zero eigenvectors as coordinates
    return eigenvectors[:, 1:dim+1]
```

**D. Activation-correlation embedding**
```python
def embed_by_correlation(model, data_loader, dim=3):
    """Run data through network, compute activation correlations,
    embed so that weights with correlated activations are close."""
    # Collect activation statistics
    correlations = compute_weight_activation_correlations(model, data_loader)
    # Use MDS or t-SNE to embed correlation matrix into 3D
    embedding = MDS(n_components=dim).fit_transform(correlations)
    return embedding
```

**E. Layer-preserving with intra-layer clustering**
```python
def embed_layered_clustered(model, dim=3):
    """Preserve layer structure on one axis, cluster within layers
    based on fan-in/fan-out patterns."""
    # x-axis: layer depth
    # y,z-axes: spectral embedding within each layer's weight matrix
    pass
```

**F. Co-evolving embedding (developmental)**
```python
def embed_developmental(model, data_loader, n_steps=1000):
    """Start with random embedding, iteratively adjust positions
    so that weights with correlated gradient signals move closer."""
    positions = np.random.randn(n_weights, 3)
    for step in range(n_steps):
        gradients = compute_gradients(model, data_loader)
        correlations = compute_gradient_correlations(gradients)
        # Attractive force between correlated weights
        # Repulsive force between uncorrelated weights
        forces = compute_forces(positions, correlations)
        positions += learning_rate * forces
    return positions
```

### Measurement Protocol

For each embedding method:
1. Train the MLP with a simple spatially-coupled learning rate: each weight's LR is the average of its own Adam LR and its k-nearest spatial neighbors' LRs
2. Compare to baseline: standard Adam (no spatial coupling)
3. Measure:
   - Final test accuracy
   - Convergence speed (steps to 95% of final accuracy)
   - Weight correlation structure (do spatially close weights develop correlated values?)
   - Gradient correlation vs. spatial distance (does the embedding predict gradient similarity?)

### Key Metrics

```
Embedding quality score = correlation(spatial_distance(w_i, w_j), gradient_correlation(w_i, w_j))
```

A good embedding has high negative correlation: spatially close weights have correlated gradients. A bad embedding has zero correlation: spatial proximity is meaningless.

## Experiment 1.2: Does Embedding Quality Predict Downstream Benefit?

### Implementation

Take the best and worst embeddings from 1.1. Apply the same astrocyte-like modulation field (from Step 02) to both. Measure whether the quality of the embedding predicts the benefit of the glial mechanism.

### Hypothesis

If embedding quality (gradient-distance correlation) is high, glial mechanisms will improve performance. If embedding quality is low, glial mechanisms will hurt performance (because locality bias is misleading).

### Expected Outcome

This experiment establishes the **boundary condition** identified in the critical reviews: spatial mechanisms help when the embedding matches functional structure, and hurt otherwise. This is the most important result to establish early because it determines whether the entire framework is viable or requires solving the embedding problem first.

## Experiment 1.3: Self-Organizing Embedding

### Implementation

Implement the developmental embedding (method F above) and test whether it converges to a useful embedding during training. Key questions:

- How many training steps are needed before the embedding stabilizes?
- Does the embedding quality improve monotonically or oscillate?
- Is there a chicken-and-egg problem (you need good representations to compute good correlations, but you need good embedding to learn good representations)?

### Protocol

1. Initialize with random embedding
2. Every 100 training steps, update embedding positions based on recent gradient correlations
3. Track embedding quality score over time
4. Compare final performance to fixed-embedding baselines

## Success Criteria

- At least one embedding method produces a quality score significantly above random
- The developmental embedding converges to a stable, high-quality embedding
- Embedding quality predicts downstream benefit of spatial coupling (correlation > 0.5)

## Deliverables

- `src/embeddings.py`: All embedding methods implemented
- `src/spatial_coupling.py`: Simple nearest-neighbor LR coupling
- `experiments/embedding_comparison.py`: Full comparison experiment
- `results/embedding_quality.csv`: Quality scores for all methods
- `results/embedding_vs_performance.png`: Scatter plot of embedding quality vs. downstream benefit

## Estimated Timeline

2-3 weeks for implementation and initial results.

## Risk Assessment

**High risk**: The developmental embedding may not converge, or may converge too slowly to be practical. If this happens, we need a principled heuristic for embedding assignment.

**Mitigation**: The spectral embedding (method C) provides a reasonable fallback that requires no training data and captures topological structure.
