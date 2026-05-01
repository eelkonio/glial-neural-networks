# Step 03: Astrocyte Units with Calcium Dynamics and Domain Coupling

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static
GLIAL INTERACTION WITH SIGNALS: Learning-only (astrocytes modulate weight updates, not signals in transit)
NOTE: At Level 2+, astrocyte calcium state would also gate synaptic transmission in real-time
      and D-serine release would affect signal processing, not just plasticity.
```

## The Claim Being Tested

Discrete astrocyte units with biologically-inspired calcium dynamics, each governing a spatial domain of weights and coupled to neighbors via gap junctions, provide better learning modulation than a continuous field alone. The calcium dynamics (nonlinear, oscillatory, history-dependent) add computational value beyond simple spatial smoothing.

## Why This Matters

Step 02 tests whether spatial coupling helps at all. This step tests whether the *biological specifics* of astrocyte calcium dynamics add value over a generic diffusion field. If they do, it validates the biological inspiration. If they don't, a simpler mathematical formulation suffices.

## Experiment 3.1: Implement Astrocyte Units

### Calcium Dynamics Model (Simplified Li-Rinzel)

```python
class AstrocyteUnit:
    """Single astrocyte with calcium dynamics governing a spatial domain."""
    
    def __init__(self, center, radius, tau_ca=50, tau_ip3=20):
        self.center = center          # (x, y, z) position
        self.radius = radius          # Domain radius
        self.tau_ca = tau_ca          # Calcium time constant (in training steps)
        self.tau_ip3 = tau_ip3        # IP3 time constant
        
        # State variables
        self.ca = 0.1                 # Cytoplasmic calcium [0, 2]
        self.ip3 = 0.0               # IP3 concentration [0, 2]
        self.er_ca = 1.0             # ER calcium store [0, 2]
        
        # Coupling state
        self.neighbor_ca = []         # Calcium from coupled neighbors
        
    def sense(self, domain_activations, domain_gradients):
        """Integrate neural activity from domain into IP3 signal."""
        # Activity drives IP3 production (neurotransmitter spillover analog)
        activity = np.mean(np.abs(domain_activations))
        gradient_signal = np.mean(np.abs(domain_gradients))
        
        # IP3 production proportional to activity
        self.ip3 += (activity + 0.5 * gradient_signal) / self.tau_ip3
        # IP3 decay
        self.ip3 *= (1.0 - 1.0 / self.tau_ip3)
        self.ip3 = np.clip(self.ip3, 0, 2.0)
        
    def update_calcium(self, dt=1.0, coupling_strength=0.1):
        """Li-Rinzel-inspired calcium dynamics."""
        # IP3-dependent release from ER (Hill function)
        h_ip3 = self.ip3**2 / (self.ip3**2 + 0.3**2)
        release = h_ip3 * self.er_ca * 0.5
        
        # Calcium-induced calcium release (CICR, positive feedback)
        h_cicr = self.ca**2 / (self.ca**2 + 0.5**2)
        cicr = h_cicr * release
        
        # SERCA pump (returns Ca to ER)
        pump = 0.4 * self.ca**2 / (self.ca**2 + 0.2**2)
        
        # Leak from ER
        leak = 0.02 * self.er_ca
        
        # Gap junction coupling (diffusion from neighbors)
        if self.neighbor_ca:
            mean_neighbor = np.mean(self.neighbor_ca)
            coupling = coupling_strength * (mean_neighbor - self.ca)
        else:
            coupling = 0.0
        
        # Update
        d_ca = (cicr + leak - pump + coupling) / self.tau_ca
        d_er = (pump - cicr - leak) / self.tau_ca
        
        self.ca += d_ca * dt
        self.er_ca += d_er * dt
        
        # Clamp to physiological range
        self.ca = np.clip(self.ca, 0.0, 2.0)
        self.er_ca = np.clip(self.er_ca, 0.0, 2.0)
        
    def output_modulation(self):
        """Convert calcium state to learning rate and gain modulation."""
        # Learning rate: sigmoid of calcium (high Ca = high plasticity)
        lr_mod = 0.5 + 1.5 * sigmoid(3.0 * (self.ca - 0.5))
        
        # Gain modulation: mild scaling of weight effective values
        gain_mod = 0.8 + 0.4 * sigmoid(2.0 * (self.ca - 0.3))
        
        return lr_mod, gain_mod
```

### Astrocyte Network (Collection of Coupled Units)

```python
class AstrocyteNetwork:
    """Network of coupled astrocyte units overlaid on neural network."""
    
    def __init__(self, n_astrocytes, weight_positions, domain_radius, coupling_k=4):
        # Place astrocytes (options: grid, random, k-means on weight positions)
        self.centers = self.place_astrocytes(n_astrocytes, weight_positions)
        
        # Assign weights to domains (each weight belongs to nearest astrocyte)
        self.assignments = self.assign_domains(weight_positions)
        
        # Build coupling graph (k nearest astrocyte neighbors)
        self.coupling_graph = build_knn(self.centers, k=coupling_k)
        
        # Create astrocyte units
        self.units = [AstrocyteUnit(c, domain_radius) for c in self.centers]
        
    def step(self, activations, gradients):
        """One astrocyte network update cycle."""
        # 1. Each astrocyte senses its domain
        for i, unit in enumerate(self.units):
            domain_mask = self.assignments == i
            unit.sense(activations[domain_mask], gradients[domain_mask])
        
        # 2. Exchange calcium with neighbors (gap junctions)
        for i, unit in enumerate(self.units):
            neighbor_indices = self.coupling_graph[i]
            unit.neighbor_ca = [self.units[j].ca for j in neighbor_indices]
        
        # 3. Update calcium dynamics
        for unit in self.units:
            unit.update_calcium()
        
        # 4. Compute output modulation
        lr_mods = np.ones(len(activations))
        gain_mods = np.ones(len(activations))
        for i, unit in enumerate(self.units):
            domain_mask = self.assignments == i
            lr, gain = unit.output_modulation()
            lr_mods[domain_mask] = lr
            gain_mods[domain_mask] = gain
            
        return lr_mods, gain_mods
```

## Experiment 3.2: Astrocyte Dynamics vs. Simple Field

### Comparison

Train the same network (MLP on MNIST/CIFAR-10) with:

1. **Baseline**: Adam (no spatial anything)
2. **Simple field**: Modulation field from Step 02 (PDE, no calcium dynamics)
3. **Astrocyte network (no coupling)**: Individual astrocyte units, no gap junctions
4. **Astrocyte network (coupled)**: Full system with gap junction coupling
5. **Astrocyte network (coupled + gain modulation)**: LR modulation AND weight gain modulation

### Measurements

- Test accuracy and convergence speed
- Calcium oscillation patterns (do they emerge? what frequency?)
- Domain specialization (do different astrocytes develop different calcium profiles?)
- Coupling effect (does information propagate between domains?)

## Experiment 3.3: Domain Size and Count

### The Question

How many astrocytes per network, and how large should their domains be?

### Protocol

For a 256-256 MLP (total ~130K weights):
- Sweep n_astrocytes: [4, 8, 16, 32, 64, 128, 256, 512]
- This gives domain sizes from ~32K weights/domain down to ~250 weights/domain
- Measure performance, overhead, and domain specialization at each scale

### Expected Result

Too few astrocytes = domains too large = modulation too coarse (approaches global LR)
Too many astrocytes = domains too small = modulation too fine (approaches per-weight, loses spatial benefit)
Optimal should be somewhere in between.

## Experiment 3.4: Calcium Oscillations and Their Function

### The Question

Biological astrocytes exhibit calcium oscillations (periodic rises and falls). Do these oscillations serve a computational purpose, or are they an artifact of the dynamics?

### Protocol

1. Run astrocyte network and record calcium traces for all units
2. Analyze: do oscillations emerge? At what frequency relative to training steps?
3. Compare: force calcium to be non-oscillatory (overdamped dynamics) vs. allow oscillations
4. Measure: does oscillatory calcium improve performance?

### Hypothesis

Oscillations create periodic windows of high and low plasticity. This may function as a natural form of cyclical learning rate scheduling — but spatially localized and activity-dependent rather than global and time-based.

## Experiment 3.5: Spontaneous Domain Formation

### The Question (from Critical Review 2)

Can the system spontaneously differentiate into specialized domains from uniform initial conditions? This would be a Turing-like instability in the astrocyte network.

### Protocol

1. Initialize all astrocytes identically (same calcium, same IP3)
2. Train the network with coupled astrocytes
3. Monitor: do astrocytes diverge into distinct states?
4. Characterize: what drives the divergence? (Different input statistics to different domains)
5. Compare: does spontaneous specialization correlate with improved performance?

### Connection to Turing Instability (Step 04)

If spontaneous domain formation occurs, Step 04 will characterize the parameter regime where it happens and whether it's beneficial or pathological.

## Success Criteria

- Coupled astrocyte network outperforms simple modulation field by measurable margin
- Calcium dynamics produce emergent oscillations with functional significance
- Domain coupling (gap junctions) provides measurable benefit over isolated units
- Spontaneous domain specialization emerges from uniform initialization

## Deliverables

- `src/astrocyte.py`: AstrocyteUnit and AstrocyteNetwork classes
- `src/astrocyte_optimizer.py`: PyTorch optimizer wrapper with astrocyte modulation
- `experiments/astrocyte_vs_field.py`: Comparison experiment
- `experiments/domain_size_sweep.py`: Domain count optimization
- `results/calcium_traces.png`: Calcium dynamics visualization
- `results/domain_specialization.png`: Heatmap of domain states over training

## Estimated Timeline

3-4 weeks. The calcium dynamics implementation is the core work; experiments are variations.
