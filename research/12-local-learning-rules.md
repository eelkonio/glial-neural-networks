# Step 12: Implement Local Learning Rules

```
SIMULATION FIDELITY: Level 1-2 (Transitional)
SIGNAL MODEL: Instantaneous for rate-based rules; Temporal for STDP
NETWORK STATE DURING INFERENCE: Static for rate-based; Evolving for spiking
GLIAL INTERACTION WITH SIGNALS: Learning-only at this step (glia added in Steps 13-15)
NOTE: STDP fundamentally requires temporal dynamics (Level 2) because it depends on
      spike timing. Rate-based Hebbian rules can work at Level 1. This step implements
      both, establishing which rules need Level 2 and which work at Level 1.
      The spiking implementation here is the foundation for Level 2 simulation.
```

## The Claim Being Tested

Local learning rules (STDP, three-factor rules, forward-forward) can function as the neural weight update mechanism in our spatially-embedded network, providing a biologically faithful substrate on which glial mechanisms can operate in their natural role.

## Why This Comes First in Phase 2

Before we can test whether glia improve local learning, we need local learning that works at all on our benchmarks. This step establishes baselines for local rules WITHOUT glia, so that Phase 2 experiments can measure the glial contribution cleanly.

## Background: Local Learning Rules

### What Makes a Rule "Local"?

A learning rule is local if the weight update for synapse w_ij depends only on information available at that synapse's physical location:
- Pre-synaptic activity (neuron i's output)
- Post-synaptic activity (neuron j's output)
- Possibly a broadcast signal (but NOT a per-weight error gradient computed by backprop)

### Rules to Implement

**A. Rate-Based Hebbian (simplest)**
```python
def hebbian_update(pre, post, w, lr=0.01):
    """Basic Hebbian: cells that fire together wire together."""
    delta_w = lr * pre * post
    # With decay to prevent unbounded growth
    delta_w -= lr * 0.1 * w
    return delta_w
```

**B. Oja's Rule (normalized Hebbian)**
```python
def oja_update(pre, post, w, lr=0.01):
    """Oja's rule: Hebbian with multiplicative normalization."""
    delta_w = lr * post * (pre - post * w)
    return delta_w
```

**C. STDP (for spiking networks)**
```python
def stdp_update(pre_spike_time, post_spike_time, w, A_plus=0.01, A_minus=0.012, 
                tau_plus=20, tau_minus=20):
    """Spike-timing-dependent plasticity."""
    dt = post_spike_time - pre_spike_time
    if dt > 0:  # Pre before post: potentiate
        delta_w = A_plus * np.exp(-dt / tau_plus)
    else:  # Post before pre: depress
        delta_w = -A_minus * np.exp(dt / tau_minus)
    return delta_w
```

**D. Three-Factor Rule (STDP + eligibility trace + third factor)**
```python
class ThreeFactorSynapse:
    def __init__(self):
        self.eligibility = 0.0  # Decaying trace of Hebbian coincidence
        self.tau_eligibility = 100  # Trace decay time constant
        
    def update_eligibility(self, pre, post):
        """Factor 1 & 2: set eligibility based on pre/post correlation."""
        hebbian_signal = pre * post
        self.eligibility += hebbian_signal
        self.eligibility *= (1.0 - 1.0 / self.tau_eligibility)  # Decay
    
    def apply_third_factor(self, third_factor_signal):
        """Factor 3: convert eligibility to actual weight change."""
        delta_w = self.eligibility * third_factor_signal
        return delta_w
```

**E. Forward-Forward Algorithm (Hinton, 2022)**
```python
def forward_forward_update(x_pos, x_neg, layer, lr=0.01):
    """Forward-forward: maximize goodness for positive, minimize for negative."""
    # Positive pass
    h_pos = layer(x_pos)
    goodness_pos = (h_pos ** 2).sum()
    
    # Negative pass
    h_neg = layer(x_neg)
    goodness_neg = (h_neg ** 2).sum()
    
    # Update to increase goodness_pos and decrease goodness_neg
    # This is local: each layer optimizes its own goodness
    loss = -torch.log(torch.sigmoid(goodness_pos - threshold)) \
           -torch.log(torch.sigmoid(threshold - goodness_neg))
    
    # Gradient is local to this layer
    loss.backward()  # Only propagates within the layer
```

**F. Predictive Coding**
```python
class PredictiveCodingLayer:
    """Each layer predicts the layer below; learning minimizes prediction error."""
    
    def __init__(self, input_dim, output_dim):
        self.W_predict = nn.Linear(output_dim, input_dim)  # Top-down prediction
        self.W_update = nn.Linear(input_dim, output_dim)   # Bottom-up update
        
    def compute_error(self, input_from_below, prediction_from_above):
        """Prediction error is local."""
        return input_from_below - prediction_from_above
    
    def update_weights(self, error, representation, lr=0.01):
        """Weight update uses only local error and local activity."""
        # Hebbian-like: correlate error with representation
        delta_W = lr * torch.outer(error, representation)
        return delta_W
```

## Experiment 12.1: Baseline Performance of Local Rules

### Protocol

Train the same MLP architecture (from Phase 1) on MNIST and CIFAR-10 using each local rule. No glial mechanisms — just the raw local rule.

### Measurements

For each rule:
- Final test accuracy (how good can it get?)
- Convergence speed (how many steps to plateau?)
- Stability (does it diverge? oscillate?)
- Representation quality (linear probe on hidden layers)

### Expected Results

| Rule | MNIST Expected | CIFAR-10 Expected | Notes |
|------|---------------|-------------------|-------|
| Backprop (reference) | ~98% | ~93% | Upper bound |
| Hebbian | ~85% | ~50% | Unstable without normalization |
| Oja's | ~90% | ~55% | Better but limited |
| Forward-Forward | ~95% | ~80% | Competitive but slower |
| Predictive Coding | ~96% | ~85% | Closest to backprop |
| Three-Factor (random 3rd) | ~88% | ~55% | Third factor is random noise |
| Three-Factor (reward) | ~92% | ~65% | Third factor is global reward |

The gap between local rules and backprop is the space where glial mechanisms might help.

## Experiment 12.2: What Local Rules Lack

### Analysis

For each local rule, characterize what goes wrong:
- **Credit assignment**: Can the rule assign credit to early layers? (Measure gradient-like signal strength at layer 1)
- **Stability**: Does the rule produce bounded weight growth? (Track weight norms)
- **Decorrelation**: Do hidden representations become redundant? (Measure representation similarity)
- **Coordination**: Do layers learn compatible representations? (Measure inter-layer alignment)

### Purpose

Identify specifically what local rules need help with. This tells us where glial mechanisms should provide the most benefit in Steps 13-15.

## Experiment 12.3: Spatial Embedding Under Local Rules

### The Question

Does the spatial embedding from Step 01 still produce meaningful structure when learning is local rather than backprop-based?

### Protocol

1. Train with local rules
2. Compute gradient-correlation-based embedding quality (but using local "gradients" — the weight update signals from the local rule)
3. Compare: does the developmental embedding (Step 01, method F) still converge to a useful embedding?

### Expected Result

The embedding should still be meaningful because spatial proximity of weights still correlates with functional similarity — this is a property of the network architecture, not the learning rule.

## Experiment 12.4: Spiking Network Implementation

### Implementation

For STDP and three-factor rules, implement a spiking neural network:

```python
class SpikingLayer:
    """Leaky integrate-and-fire neurons with STDP."""
    
    def __init__(self, n_neurons, n_inputs, threshold=1.0, tau_mem=20.0):
        self.n = n_neurons
        self.W = torch.randn(n_neurons, n_inputs) * 0.1
        self.threshold = threshold
        self.tau_mem = tau_mem
        self.membrane = torch.zeros(n_neurons)
        self.spike_times = torch.full((n_neurons,), -1000.0)
        
    def forward(self, input_spikes, t):
        """Process one timestep."""
        # Membrane dynamics
        self.membrane *= (1.0 - 1.0 / self.tau_mem)
        self.membrane += self.W @ input_spikes
        
        # Spike generation
        spikes = (self.membrane >= self.threshold).float()
        self.membrane[spikes > 0] = 0.0  # Reset
        
        # Record spike times
        self.spike_times[spikes > 0] = t
        
        return spikes
```

### Purpose

STDP requires spike timing, which requires a spiking network. This implementation will be the substrate for Phase 2 experiments with astrocyte gating.

## Success Criteria

- At least one local rule achieves >90% on MNIST without glia
- Performance gap between local rules and backprop is clearly characterized
- Specific deficiencies of local rules are identified (credit assignment, stability, coordination)
- Spiking network implementation is functional and trainable
- Spatial embedding remains meaningful under local learning

## Deliverables

- `src/local_rules/hebbian.py`: Hebbian and Oja's rule
- `src/local_rules/stdp.py`: STDP implementation
- `src/local_rules/three_factor.py`: Three-factor rule with eligibility traces
- `src/local_rules/forward_forward.py`: Forward-forward algorithm
- `src/local_rules/predictive_coding.py`: Predictive coding layers
- `src/spiking/lif_network.py`: Leaky integrate-and-fire spiking network
- `experiments/local_rule_baselines.py`: Baseline comparison
- `results/local_rule_performance.csv`: Performance table
- `results/local_rule_deficiencies.md`: Analysis of what each rule lacks

## Estimated Timeline

4-5 weeks. Multiple rule implementations plus thorough benchmarking.
