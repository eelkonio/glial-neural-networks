# Step 14: Volume Transmission as the Teaching/Error Broadcast Signal

```
SIMULATION FIDELITY: Level 1-2 (Transitional)
SIGNAL MODEL: Instantaneous (error computed instantly) but diffusion is temporal
NETWORK STATE DURING INFERENCE: Evolving (error field diffuses over time)
GLIAL INTERACTION WITH SIGNALS: Learning-only (error field guides weight updates)
NOTE: The diffusion of the error signal is inherently temporal — it takes time to
      propagate spatially. At Level 1, we simulate this diffusion but the neural
      forward pass is still instantaneous. At Level 2+, the error signal diffuses
      WHILE the network is processing the next input, creating temporal overlap
      between inference and learning that is biologically realistic.
```

## The Claim Being Tested

Volume transmission (diffusion-based spatial broadcast) can serve as the mechanism that delivers a teaching signal to local synapses, replacing backpropagation's role in distributing error information. The spatial properties of diffusion (distance-dependent attenuation, finite propagation speed, regional broadcast) create a teaching signal that is biologically plausible and computationally sufficient.

## Why This Matters

The biggest weakness of local learning rules is the absence of a global error signal. Backprop provides this by propagating error backward through the network. Biology doesn't do this — but it DOES have volume transmission: broadcast chemical signals that reach all synapses within a spatial radius. This step tests whether volume-transmitted error can replace backpropagated error.

## Experiment 14.1: Error Signal via Volume Transmission

### The Concept

Instead of backpropagating error through the network graph, broadcast an error signal through the spatial embedding via diffusion:

```python
class VolumeTransmittedError:
    """Broadcasts error signal through spatial diffusion."""
    
    def __init__(self, weight_positions, output_positions, D=0.1, decay=0.05):
        self.positions = weight_positions
        self.output_positions = output_positions  # Where error originates
        self.D = D
        self.decay = decay
        self.error_field = np.zeros(len(weight_positions))
        
    def broadcast_error(self, output_error):
        """Release error signal at output layer positions, let it diffuse."""
        # Error originates at output layer
        for i, pos in enumerate(self.output_positions):
            # Compute distance from this output to all weights
            distances = np.linalg.norm(self.positions - pos, axis=1)
            # Error contribution falls off with distance
            contribution = output_error[i] * np.exp(-distances**2 / (2 * self.sigma**2))
            self.error_field += contribution
        
        # Diffuse existing field (spread to neighbors)
        self.diffuse_step()
        
        # Decay
        self.error_field *= (1 - self.decay)
        
    def get_teaching_signal(self):
        """Each weight reads its local error field value as teaching signal."""
        return self.error_field
    
    def diffuse_step(self):
        """One step of spatial diffusion on the weight graph."""
        new_field = np.copy(self.error_field)
        for i in range(len(self.positions)):
            neighbors = self.get_neighbors(i)
            laplacian = np.mean(self.error_field[neighbors]) - self.error_field[i]
            new_field[i] += self.D * laplacian
        self.error_field = new_field
```

### The Learning Rule

```python
def volume_taught_update(synapse, pre, post, local_error_field_value):
    """
    Local Hebbian rule modulated by volume-transmitted error.
    
    This is biologically plausible:
    - pre/post are local (available at the synapse)
    - error_field_value is from volume transmission (available in extracellular space)
    - No backward pass through the network graph
    """
    # Eligibility: Hebbian coincidence
    eligibility = pre * post
    
    # Teaching signal: volume-transmitted error at this spatial location
    teaching = local_error_field_value
    
    # Weight update: eligibility * teaching signal
    delta_w = LEARNING_RATE * eligibility * teaching
    
    return delta_w
```

## Experiment 14.2: Properties of Diffused Error

### The Question

How does the diffused error signal compare to the backpropagated gradient?

### Analysis

1. Compute true backprop gradients for each weight
2. Compute volume-transmitted error field value for each weight
3. Measure correlation between the two
4. Analyze: where does the correlation break down? (deep layers? specific architectures?)

### Expected Properties of Diffused Error

| Property | Backprop Gradient | Diffused Error |
|----------|------------------|----------------|
| Precision | Exact per-weight | Approximate, spatially smoothed |
| Availability | Requires backward pass | Available in "extracellular space" |
| Spatial structure | None (per-weight) | Smooth field (nearby weights get similar signal) |
| Temporal dynamics | Instantaneous | Propagates at finite speed |
| Depth dependence | Attenuates (vanishing gradient) | Attenuates with distance (diffusion) |
| Biological plausibility | None | High (volume transmission is real) |

### Key Insight

The diffused error is NOT the same as the backprop gradient. It is a **spatially smoothed, temporally delayed approximation**. The question is whether this approximation is good enough for learning.

## Experiment 14.3: Diffusion Parameters and Learning Quality

### Protocol

Sweep diffusion parameters and measure learning quality:

- **Diffusion coefficient D**: Controls how far error spreads
  - D too low: error stays near output, deep layers get no signal
  - D too high: error becomes uniform everywhere (no spatial information)
  
- **Decay rate**: Controls how long error persists
  - Decay too fast: error disappears before reaching deep layers
  - Decay too slow: old errors interfere with new ones

- **Sigma (release width)**: Controls initial spread of error release
  - Sigma too small: error is too localized
  - Sigma too large: error is too diffuse from the start

### Measurement

For each parameter combination:
- Final accuracy on MNIST/CIFAR-10
- Correlation between diffused error and true gradient (per layer)
- Learning speed (steps to convergence)
- Stability (does learning diverge?)

## Experiment 14.4: Spatial Embedding Determines Error Routing

### The Critical Insight

In backprop, error flows along the network graph (backward through connections). In volume transmission, error flows through physical space (diffusion in the spatial embedding). This means **the spatial embedding determines how error reaches each weight**.

### Protocol

Test different spatial embeddings (from Step 01) and measure how well error reaches deep layers:

1. **Layer-separated embedding**: Layers are spatially distant → error must diffuse far to reach early layers
2. **Compact embedding**: All layers are spatially close → error reaches everywhere quickly
3. **Hierarchical embedding**: Output near middle, layers arranged around it → error reaches all layers equally

### Expected Result

The spatial embedding that places output neurons centrally (so error diffuses equally to all layers) should produce the best learning. This is a design constraint that the embedding must satisfy.

## Experiment 14.5: Multi-Source Error Broadcasting

### The Concept

Instead of only broadcasting from the output layer, broadcast error from multiple points:

```python
class MultiSourceErrorBroadcast:
    """Error is broadcast from multiple locations, not just output."""
    
    def __init__(self, weight_positions):
        self.positions = weight_positions
        self.error_sources = []  # List of (position, error_value) pairs
        
    def add_local_error_source(self, layer_idx, local_loss):
        """Each layer can compute a local loss and broadcast it."""
        # Local loss: e.g., prediction error in predictive coding
        # or goodness difference in forward-forward
        source_position = self.get_layer_center(layer_idx)
        self.error_sources.append((source_position, local_loss))
    
    def compute_field(self):
        """Superposition of all error sources."""
        field = np.zeros(len(self.positions))
        for pos, error in self.error_sources:
            distances = np.linalg.norm(self.positions - pos, axis=1)
            field += error * np.exp(-distances**2 / (2 * self.sigma**2))
        return field
```

### The Question

Does multi-source broadcasting (where each layer contributes a local error signal) work better than single-source (output-only) broadcasting?

### Connection to Predictive Coding

In predictive coding, each layer computes a local prediction error. If these local errors are broadcast via volume transmission, we get a system where:
- Each layer has its own error source (local prediction error)
- Error diffuses to nearby weights (volume transmission)
- Weights update based on local Hebbian rule * local error field

This is a biologically plausible implementation of predictive coding where the "error neurons" are replaced by volume-transmitted chemical signals.

## Experiment 14.6: Comparison to Other Biologically Plausible Error Delivery

### Baselines

1. **Backprop** (upper bound, not biologically plausible)
2. **Random feedback alignment**: Random backward weights (Lillicrap et al.)
3. **Direct feedback alignment**: Error projected directly to each layer
4. **Forward-forward**: No backward pass, local goodness optimization
5. **Volume-transmitted error**: This experiment
6. **Volume-transmitted error + astrocyte gating**: Combined with Step 13

### Expected Ranking

```
Backprop > Predictive coding ≈ Volume+astrocyte > Forward-forward > 
Direct feedback > Random feedback > Volume alone > Pure Hebbian
```

The key question: where does volume-transmitted error + astrocyte gating land relative to other biologically plausible methods?

## Success Criteria

- Volume-transmitted error enables learning (accuracy > random chance) on MNIST and CIFAR-10
- Correlation between diffused error and true gradient is > 0.3 for at least the last 2 layers
- Optimal diffusion parameters exist in an intermediate range
- Multi-source broadcasting outperforms single-source
- Volume error + astrocyte gating is competitive with forward-forward or predictive coding

## Deliverables

- `src/volume_error.py`: VolumeTransmittedError implementation
- `src/volume_taught_learning.py`: Learning rule using volume-transmitted teaching signal
- `experiments/error_correlation.py`: Correlation between diffused error and true gradient
- `experiments/diffusion_parameter_sweep.py`: Parameter optimization
- `experiments/embedding_for_error.py`: Which spatial embedding best routes error?
- `experiments/bioplausible_comparison.py`: Comparison to other bio-plausible methods
- `results/error_field_evolution.mp4`: Animation of error diffusing through network
- `results/method_comparison.csv`: Accuracy comparison table

## Estimated Timeline

5-6 weeks. Novel territory with many unknowns.
