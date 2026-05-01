# Step 15: Astrocyte-Mediated Heterosynaptic Plasticity

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static
GLIAL INTERACTION WITH SIGNALS: Learning-only (heterosynaptic effects modify weight updates)
NOTE: Heterosynaptic plasticity is primarily a learning phenomenon (lateral competition
      during weight updates) and works at Level 1. At Level 2+, it would also create
      real-time lateral inhibition during signal propagation (signals at one synapse
      suppress transmission at neighboring synapses via astrocyte-mediated depression).
```

## The Claim Being Tested

Astrocytes mediate lateral interactions between synapses within their domain: when one synapse is potentiated, nearby synapses can be depressed (or vice versa). This heterosynaptic plasticity creates competition and decorrelation between co-located synapses, improving representation quality without requiring any global signal.

## Why This Matters

Heterosynaptic plasticity is something backpropagation cannot do and that no standard optimizer implements. It is a purely spatial, purely local mechanism that creates useful computational structure (decorrelated representations, competitive learning) from nothing more than physical proximity. If it works, it validates the spatial geometry argument in its strongest form.

## Background: Biological Heterosynaptic Plasticity

In biology, when synapse A is potentiated (strengthened):
- Nearby synapses B, C, D (within the same astrocyte domain) are depressed
- The astrocyte mediates this: calcium rise from synapse A's activity triggers gliotransmitter release that depresses neighboring synapses
- This creates a **winner-take-all** dynamic within each astrocyte domain
- It also prevents runaway potentiation (homeostatic function)

The mechanism:
```
Synapse A active + post-synaptic neuron active
    -> Synapse A potentiated (Hebbian LTP)
    -> Glutamate spillover detected by astrocyte
    -> Astrocyte calcium rises
    -> Astrocyte releases ATP/adenosine at neighboring synapses
    -> Neighboring synapses B, C, D depressed (heterosynaptic LTD)
```

## Experiment 15.1: Implement Heterosynaptic Plasticity

### Implementation

```python
class HeterosynapticAstrocyte:
    """Astrocyte that mediates lateral competition between synapses in its domain."""
    
    def __init__(self, domain_synapse_indices, positions, radius):
        self.domain = domain_synapse_indices
        self.positions = positions[domain_synapse_indices]
        self.radius = radius
        self.ca = 0.1
        
        # Precompute pairwise distances within domain
        self.internal_distances = cdist(self.positions, self.positions)
        
    def compute_heterosynaptic_signal(self, potentiated_synapses, potentiation_amounts):
        """
        When some synapses are potentiated, compute depression signal for neighbors.
        
        potentiated_synapses: indices (within domain) of synapses that just strengthened
        potentiation_amounts: how much each was strengthened
        """
        depression_signal = np.zeros(len(self.domain))
        
        for syn_idx, amount in zip(potentiated_synapses, potentiation_amounts):
            # Depression falls off with distance from potentiated synapse
            distances = self.internal_distances[syn_idx]
            
            # Nearby synapses get depressed; the potentiated one is spared
            depression = amount * np.exp(-distances**2 / (2 * (self.radius * 0.3)**2))
            depression[syn_idx] = 0  # Don't depress the one that was just potentiated
            
            depression_signal += depression
        
        return depression_signal
    
    def apply_heterosynaptic_plasticity(self, weights, weight_updates):
        """
        After Hebbian updates are computed, apply heterosynaptic modulation.
        
        weights: current weight values for domain synapses
        weight_updates: proposed Hebbian updates (before heterosynaptic modification)
        """
        # Identify which synapses were potentiated
        potentiated = np.where(weight_updates > 0)[0]
        amounts = weight_updates[potentiated]
        
        if len(potentiated) == 0:
            return weight_updates  # No potentiation, no heterosynaptic effect
        
        # Compute depression signal
        depression = self.compute_heterosynaptic_signal(potentiated, amounts)
        
        # Modify updates: add depression to non-potentiated synapses
        modified_updates = weight_updates - depression * HETEROSYNAPTIC_STRENGTH
        
        return modified_updates
```

### Integration with Three-Factor Rule

```python
def three_factor_with_heterosynaptic(network, astrocytes, pre, post, gate):
    """Full learning step with heterosynaptic plasticity."""
    
    # Step 1: Compute standard three-factor updates
    raw_updates = np.zeros(network.n_weights)
    for i in range(network.n_weights):
        eligibility = pre[network.pre_idx[i]] * post[network.post_idx[i]]
        raw_updates[i] = eligibility * gate[i] * LR
    
    # Step 2: Apply heterosynaptic modulation within each astrocyte domain
    final_updates = np.copy(raw_updates)
    for astrocyte in astrocytes:
        domain_updates = raw_updates[astrocyte.domain]
        modified = astrocyte.apply_heterosynaptic_plasticity(
            network.weights[astrocyte.domain], domain_updates
        )
        final_updates[astrocyte.domain] = modified
    
    # Step 3: Apply updates
    network.weights += final_updates
```

## Experiment 15.2: Does Heterosynaptic Plasticity Improve Representations?

### The Question

Does lateral competition between synapses produce better (more decorrelated, more informative) hidden representations?

### Protocol

Train networks with and without heterosynaptic plasticity. Measure representation quality:

1. **Decorrelation**: Compute pairwise correlation between hidden unit activations. Lower correlation = more diverse representations.
2. **Linear separability**: Train linear probe on hidden representations. Higher accuracy = more informative representations.
3. **Redundancy**: Compute effective dimensionality of hidden layer activations. Higher = less redundant.
4. **Sparsity**: Measure activation sparsity. Sparser = more selective features.

### Expected Result

Heterosynaptic plasticity should:
- Decrease correlation between hidden units (competition forces diversity)
- Increase linear separability (more informative features)
- Increase effective dimensionality (less redundancy)
- Increase sparsity (winner-take-all creates selective responses)

## Experiment 15.3: Heterosynaptic Plasticity as Implicit Regularization

### The Question

Does heterosynaptic plasticity prevent overfitting by creating competition that limits co-adaptation?

### Protocol

1. Train on small dataset (1000 MNIST examples) — prone to overfitting
2. Compare generalization gap (train acc - test acc) with and without heterosynaptic plasticity
3. Compare to explicit regularization (dropout, weight decay)

### Expected Result

Heterosynaptic plasticity should reduce overfitting because:
- It prevents multiple synapses from learning the same feature (reduces redundancy)
- It creates implicit competition that acts like dropout (but spatially structured)
- It maintains weight diversity (prevents collapse to similar values)

## Experiment 15.4: Domain Size and Competition Range

### The Question

How far should the heterosynaptic depression extend? This is determined by the astrocyte domain size.

### Protocol

Sweep the competition radius (fraction of domain radius):
- Very local (10% of domain): only immediate neighbors compete
- Medium (30% of domain): moderate competition range
- Domain-wide (100%): all synapses in domain compete with each other
- Cross-domain (>100%): competition extends beyond single astrocyte

### Expected Result

There should be an optimal competition range:
- Too local: insufficient decorrelation
- Too wide: excessive depression prevents learning
- Optimal: enough competition for diversity without suppressing all learning

## Experiment 15.5: Heterosynaptic Plasticity + Backprop Comparison

### The Question

Does heterosynaptic plasticity provide benefit under backprop too, or only under local rules?

### Protocol

```
Condition A: Backprop, no heterosynaptic
Condition B: Backprop + heterosynaptic plasticity
Condition C: Local rule, no heterosynaptic
Condition D: Local rule + heterosynaptic plasticity

Benefit under backprop = B - A
Benefit under local = D - C
```

### Expected Result

Heterosynaptic plasticity should help more under local rules because:
- Backprop already decorrelates representations through its gradient signal
- Local rules have no mechanism for decorrelation — heterosynaptic plasticity provides it
- Under backprop, heterosynaptic plasticity might even hurt (conflicting with gradient signal)

## Success Criteria

- Heterosynaptic plasticity measurably decorrelates hidden representations (correlation reduction > 20%)
- Representation quality improves (linear probe accuracy increase > 3%)
- Generalization improves on small datasets (gap reduction > 5%)
- Optimal competition range exists in intermediate regime
- Benefit is greater under local rules than under backprop

## Deliverables

- `src/heterosynaptic.py`: HeterosynapticAstrocyte implementation
- `src/competitive_learning.py`: Integration with three-factor rule
- `experiments/representation_quality.py`: Decorrelation and probe measurements
- `experiments/regularization_comparison.py`: Comparison to dropout/weight decay
- `experiments/competition_range_sweep.py`: Domain size optimization
- `results/correlation_matrices.png`: Before/after correlation heatmaps
- `results/representation_dimensionality.png`: Effective dimensionality over training

## Estimated Timeline

3-4 weeks. Builds on Step 13 infrastructure.
