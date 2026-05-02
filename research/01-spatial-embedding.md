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

Additionally, run all experiments on a **topographic sensor task** — a classification task over a simulated 16×16 sensor grid with spatially correlated inputs — to test whether spatial mechanisms provide greater benefit on tasks with inherent spatial structure.

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

**G. Adversarial embedding (negative control)**
```python
def embed_adversarial(model, data_loader, n_batches=10):
    """Deliberately anti-correlated embedding. Computes gradient correlations
    then assigns positions that MAXIMIZE distance between correlated weights.
    This is the negative end of the three-point validation curve."""
    correlations = compute_gradient_correlations(model, data_loader)
    # Use MDS on NEGATED correlation matrix
    # Highly correlated weights get maximally distant positions
    embedding = MDS(n_components=3).fit_transform(-correlations)
    return normalize_to_unit_cube(embedding)
```

**H. Differentiable embedding (learnable positions)**
```python
def embed_differentiable(model, data_loader, lambda_spatial=0.01):
    """Treat positions as learnable parameters optimized jointly with
    the network via a spatial coherence loss. Solves the chicken-and-egg
    problem by letting gradients flow through both task and spatial losses."""
    positions = nn.Parameter(torch.rand(n_weights, 3))
    # Add to optimizer alongside model parameters
    # Loss = task_loss + lambda_spatial * spatial_coherence_loss
    # spatial_coherence_loss = mean(distance(i,j) * grad_corr(i,j)) for correlated pairs
    return positions.detach().numpy()
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
   - **Spatial coherence score** (do spatially close weights develop similar PCA projections after training?)
   - **Embedding quality over training time** (does the quality score degrade as the network learns?)
4. Run on both MNIST and the topographic sensor task
5. Verify the **three-point validation curve**: adversarial embedding hurts, random is neutral, good embedding helps

### Key Metrics

```
Embedding quality score = correlation(spatial_distance(w_i, w_j), gradient_correlation(w_i, w_j))
```

A good embedding has high negative correlation: spatially close weights have correlated gradients. A bad embedding has zero correlation: spatial proximity is meaningless. An adversarial embedding has positive correlation: correlated weights are deliberately far apart.

```
Spatial coherence score = correlation(spatial_distance(w_i, w_j), PCA_similarity(w_i, w_j))
```

Measures whether training with spatial coupling produces spatially organized weight structure. High score = the mechanism is working (not just regularization).

## Experiment 1.2: Does Embedding Quality Predict Downstream Benefit?

### Implementation

Take the best and worst embeddings from 1.1, plus the adversarial embedding. Apply the same astrocyte-like modulation field (from Step 02) to all. Measure whether the quality of the embedding predicts the benefit of the glial mechanism.

### The Three-Point Validation Curve

The critical test from Critical Review 3: if spatial coupling helps even with a random embedding, the benefit is from regularization (the weak claim). If it only helps with a good embedding AND hurts with an adversarial embedding, spatial structure genuinely matters (the strong claim).

Expected curve:
- Adversarial embedding + coupling → performance WORSE than baseline
- Random embedding + coupling → performance NEUTRAL (or slight regularization benefit)
- Good embedding + coupling → performance BETTER than baseline

This three-point curve is the single strongest piece of evidence for or against the spatial structure hypothesis.

### Hypothesis

If embedding quality (gradient-distance correlation) is high, glial mechanisms will improve performance. If embedding quality is low, glial mechanisms will hurt performance (because locality bias is misleading).

### Expected Outcome

This experiment establishes the **boundary condition** identified in the critical reviews: spatial mechanisms help when the embedding matches functional structure, and hurt otherwise. This is the most important result to establish early because it determines whether the entire framework is viable or requires solving the embedding problem first.

## Experiment 1.3: Self-Organizing Embedding

### Implementation

Implement both the developmental embedding (method F) and the differentiable embedding (method H). Compare their convergence properties and final quality.

**Developmental (force-based)**: Iteratively adjust positions based on gradient correlation forces. Has the chicken-and-egg problem — needs a warmup period.

**Differentiable (loss-based)**: Positions are PyTorch parameters with a spatial coherence loss term. Gradients flow through both task and spatial objectives simultaneously. Cleaner solution to the chicken-and-egg problem.

Key questions:

- How many training steps are needed before the embedding stabilizes?
- Does the embedding quality improve monotonically or oscillate?
- Does the differentiable approach converge faster than the force-based approach?
- Is there still a chicken-and-egg problem with differentiable positions, or does joint optimization resolve it?

### Protocol

1. Initialize with random embedding
2. Every 100 training steps, update embedding positions based on recent gradient correlations
3. Track embedding quality score over time
4. Compare final performance to fixed-embedding baselines

## Experiment 1.4: Embedding Quality Over Training Time

### The Question (from Critical Review 3)

Fixed embeddings (spectral, correlation) capture the network's structure at one point in time. Does this structure remain valid as the network learns? If the spectral embedding becomes meaningless after 5 epochs, all downstream experiments using it are compromised.

### Protocol

1. Compute embedding quality score at regular intervals (every 2 epochs) during training
2. Track for all fixed embedding methods (linear, spectral, correlation, layered-clustered)
3. Compare against the differentiable embedding (which co-adapts)
4. Flag any embedding whose quality drops by >50% from initial value

### Expected Outcomes

- Linear embedding: quality likely stable (structure is fixed by architecture)
- Spectral embedding: may degrade as weight magnitudes change the effective graph
- Correlation embedding: likely degrades (computed from initial activations)
- Differentiable embedding: should maintain or improve quality (co-adapts)

### Implications

If fixed embeddings degrade significantly, this motivates either:
- Periodic recomputation of the embedding during training
- Using the differentiable embedding as the default
- Accepting that the embedding is a "good enough" initialization

## Experiment 1.5: Spatial Coherence — Mechanism vs. Regularization

### The Question (from Critical Review 3)

If spatial coupling improves performance, is it because spatial structure is genuinely informative (the strong claim) or because spatial smoothing of learning rates is just a good regularizer (the weak claim)?

### Protocol

1. Train with spatial coupling using a good embedding → measure spatial coherence
2. Train without spatial coupling → measure spatial coherence
3. Compare: does spatial coupling produce more spatially organized weight structure?

### Metric

```python
# After training, compute top-k PCA components of weight matrix
pcs = PCA(n_components=10).fit_transform(weights)

# For sampled pairs, compute:
# - Spatial distance (from embedding)
# - PC similarity (dot product of PC projections)

spatial_coherence = pearsonr(spatial_distances, pc_similarities)
```

### Interpretation

- If coupled training has significantly higher spatial coherence than uncoupled → the mechanism is producing spatially organized representations (strong claim supported)
- If both have similar spatial coherence → the benefit is from regularization, not spatial organization (weak claim)

## Success Criteria

- At least one embedding method produces a quality score significantly above random
- The developmental or differentiable embedding converges to a stable, high-quality embedding
- Embedding quality predicts downstream benefit of spatial coupling (correlation > 0.5)
- The three-point validation curve is monotonic: adversarial < random < good
- Spatial coherence is higher for coupled training than uncoupled training
- The topographic task shows larger benefit from spatial coupling than MNIST

## Deliverables

- `src/embeddings.py`: All 8 embedding methods implemented (linear, random, spectral, correlation, layered-clustered, developmental, adversarial, differentiable)
- `src/spatial_coupling.py`: Simple nearest-neighbor LR coupling
- `src/quality.py`: Embedding quality score and spatial coherence measurement
- `src/temporal_tracking.py`: Quality-over-time tracking
- `src/topographic_task.py`: Spatially-structured benchmark task
- `experiments/embedding_comparison.py`: Full comparison experiment (10 conditions × 2 tasks)
- `experiments/three_point_validation.py`: Adversarial → random → good curve
- `experiments/temporal_quality.py`: Quality degradation tracking
- `experiments/spatial_coherence.py`: Mechanism vs. regularization test
- `results/embedding_quality.csv`: Quality scores for all methods
- `results/embedding_vs_performance.png`: Scatter plot of embedding quality vs. downstream benefit
- `results/three_point_curve.png`: The validation curve
- `results/temporal_trajectories.png`: Quality over training time
- `results/spatial_coherence.png`: Coupled vs. uncoupled coherence

## Estimated Timeline

3-4 weeks for implementation and initial results (expanded from 2-3 weeks due to additional experiments from Critical Review 3).

## Risk Assessment

**High risk**: The developmental embedding may not converge, or may converge too slowly to be practical. If this happens, we need a principled heuristic for embedding assignment.

**Mitigation**: The spectral embedding (method C) provides a reasonable fallback that requires no training data and captures topological structure. The differentiable embedding (method H) provides an alternative self-organizing approach that avoids the force-based convergence issues.

**Medium risk**: Fixed embeddings may degrade during training (identified by Critical Review 3). If spectral/correlation embeddings become meaningless after a few epochs, downstream experiments using them are compromised.

**Mitigation**: Temporal quality tracking (Experiment 1.4) will detect this early. The differentiable embedding co-adapts and should maintain quality. Periodic recomputation is a fallback.

**Medium risk**: MNIST may not show meaningful differences between embeddings because it's too easy and permutation-invariant.

**Mitigation**: The topographic task provides a benchmark where spatial structure should matter. If MNIST shows no differences but the topographic task does, the framework is validated for structured tasks.
