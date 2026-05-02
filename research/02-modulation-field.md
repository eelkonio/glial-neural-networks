# Step 02: The Reaction-Diffusion Modulation Field

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static
GLIAL INTERACTION WITH SIGNALS: Learning-only (field modulates learning rates, not signal propagation)
NOTE: At Level 2+, this same field would also modulate signal gain during propagation.
      Level 1 tests whether the field helps learning; Level 2+ tests whether it also helps inference.
```

## The Claim Being Tested

Coupling gradient descent to a reaction-diffusion PDE over the spatial embedding creates qualitatively different (and better) learning dynamics than flat per-weight optimization. The modulation field provides spatially smooth, temporally persistent learning rate adaptation that standard optimizers cannot replicate.

## Why This Matters

The first critical review identified the PDE-ODE coupling as the document's strongest theoretical contribution. This experiment tests whether that theoretical distinction translates into practical benefit.

## Experiment 2.1: Implement the Modulation Field PDE

### The Equation

```
dM/dt = D * laplacian(M) + S(x,y,z,t) - decay * M

Where:
  M(x,y,z,t) = modulation field value (learning rate multiplier)
  D = diffusion coefficient
  S = source term (driven by neural activity / gradient magnitudes)
  decay = natural decay rate
  laplacian = spatial Laplacian (nabla squared)
```

### Implementation

```python
class ModulationField:
    """Reaction-diffusion field over the network's spatial embedding."""
    
    def __init__(self, positions, D=0.1, decay=0.01, dt=0.1):
        """
        positions: (N, 3) array of weight spatial coordinates
        D: diffusion coefficient
        decay: field decay rate
        dt: PDE timestep
        """
        self.positions = positions
        self.D = D
        self.decay = decay
        self.dt = dt
        self.N = len(positions)
        
        # Build spatial neighbor graph (k-nearest neighbors)
        self.neighbors, self.distances = build_knn_graph(positions, k=10)
        
        # Field state: one scalar per weight position
        self.M = np.ones(self.N)  # Initialize at 1.0 (no modulation)
        
    def compute_laplacian(self):
        """Discrete Laplacian on the spatial graph."""
        laplacian = np.zeros(self.N)
        for i in range(self.N):
            for j, d in zip(self.neighbors[i], self.distances[i]):
                laplacian[i] += (self.M[j] - self.M[i]) / (d**2 + 1e-8)
        return laplacian
    
    def step(self, source_term):
        """Advance the PDE by one timestep.
        source_term: (N,) array driven by gradient magnitudes.
        """
        laplacian = self.compute_laplacian()
        dM = self.D * laplacian + source_term - self.decay * (self.M - 1.0)
        self.M += self.dt * dM
        self.M = np.clip(self.M, 0.1, 5.0)  # Prevent extreme values
        
    def get_lr_multipliers(self):
        """Return current field values as learning rate multipliers."""
        return self.M
```

### Source Term Design

The source term S drives the field based on neural network state. Options to test:

**A. Gradient magnitude source**
```python
source = gradient_magnitudes / gradient_magnitudes.mean() - 1.0
# Positive where gradients are large (increase LR there)
# Negative where gradients are small (decrease LR there)
```

**B. Loss contribution source**
```python
source = per_weight_loss_contribution - mean_contribution
# Positive where weight contributes to high loss
```

**C. Gradient consistency source**
```python
source = sign_agreement_over_last_k_steps - 0.5
# Positive where gradient direction is consistent (confident learning)
# Negative where gradient oscillates (uncertain)
```

### Measurement

Train the same MLP (from Step 01) with:
1. **Baseline**: Adam optimizer (flat geometry)
2. **Modulation field + SGD**: SGD with learning rate = base_lr * M(position)
3. **Modulation field + Adam**: Adam with per-weight LR scaled by M(position)
4. **Oracle**: Adam with per-weight LR tuned by grid search (upper bound)
5. **Permuted embedding control**: Modulation field + Adam, but with randomly shuffled spatial positions (tests whether benefit comes from spatial structure or from smoothing-as-regularization)
6. **KFAC baseline**: Kronecker-factored approximate curvature preconditioner (tests whether the modulation field is approximating known structured preconditioning)

Measure:
- Test accuracy over training steps
- Convergence speed
- Generalization gap (train acc - test acc)
- Spatial structure of final M field (is it smooth? does it have meaningful patterns?)
- **Spatial coherence score** (does the field produce spatially organized weight structure?)
- **Three-point check**: does the field help with good embedding, hurt with adversarial embedding?

## Experiment 2.2: Diffusion Coefficient Sweep

### The Question

How does the diffusion coefficient D affect learning? D controls how quickly modulation spreads spatially:
- D = 0: No diffusion, field is purely local (equivalent to per-weight adaptive LR)
- D small: Slow diffusion, local domains form
- D large: Fast diffusion, field becomes spatially uniform (equivalent to global LR)

### Protocol

Sweep D over logarithmic range [0.001, 0.01, 0.1, 1.0, 10.0] and measure:
- Final performance
- Spatial correlation length of M field (how large are the domains?)
- Training stability (variance of loss over last 100 steps)

### Expected Result

There should be an optimal D that is neither zero (no spatial coupling) nor infinite (no spatial structure). This optimal D defines the natural "domain size" of the modulation field.

## Experiment 2.3: PDE vs. Heuristic Spatial Coupling

### The Question

Is the full PDE necessary, or does a simpler heuristic (e.g., averaging LR with k-nearest neighbors every N steps) achieve the same benefit?

### Implementations to Compare

**A. Full PDE** (as above)

**B. Neighbor averaging**
```python
# Every 100 steps:
for i in range(N):
    lr[i] = 0.5 * lr[i] + 0.5 * mean(lr[neighbors[i]])
```

**C. Gaussian smoothing**
```python
# Every 100 steps:
lr = gaussian_filter_on_graph(lr, sigma=domain_size)
```

**D. No spatial coupling** (baseline Adam)

### Expected Result

If the PDE dynamics (temporal persistence, wave-like propagation, source-driven adaptation) provide value beyond simple spatial smoothing, then A should outperform B and C. If simple smoothing is sufficient, the PDE is unnecessary overhead.

## Experiment 2.4: Visualizing Field Dynamics

### Implementation

Create visualizations of the modulation field evolving during training:
- 2D projection of the 3D field (using PCA of positions)
- Color-coded by M value (blue = low LR, red = high LR)
- Animated over training steps
- Overlay with gradient magnitude heatmap

### Purpose

Qualitative understanding of whether the field develops meaningful spatial structure or remains noisy. Look for:
- Domain formation (regions of similar M value)
- Wave-like propagation (spreading activation)
- Correlation with network functional structure

## Experiment 2.5: Glial Field as Implicit Meta-Learner (from Critical Review 3)

### The Question

Instead of the field simply multiplying the learning rate, can the field state serve as an *input* to a learned update function? This reframes the glial system as an implicit meta-optimizer.

### Implementation

```python
# Standard modulation (current design):
delta_w = -lr * field(position) * gradient

# Meta-learner variant:
context = field(position)  # Field state encodes "local learning context"
update_scale = meta_network(context)  # Small learned function (MLP: 1→8→1)
delta_w = -lr * update_scale * gradient
```

The meta-network is tiny (input: field state scalar, hidden: 8 units, output: scalar) and shared across all weights. It learns "given that my local glial environment looks like X, what scale of update is appropriate?"

### Properties

- Spatially local (each weight reads only its local field)
- Temporally smooth (field state changes slowly via PDE dynamics)
- Context-aware (the field encodes recent activity history)
- Learnable (the meta-network adapts during training)

### Comparison

1. **Direct multiplication** (current): delta_w = -lr * M(pos) * grad
2. **Meta-learner**: delta_w = -lr * meta_net(M(pos)) * grad
3. **Learned per-weight LR** (no spatial structure): delta_w = -lr * learned_scale[i] * grad

### Expected Result

If the meta-learner outperforms direct multiplication, it suggests the field state carries richer information than a simple scalar multiplier — the nonlinear transformation extracts useful signal. If it matches direct multiplication, the simpler approach suffices.

### Connection to Meta-Learning Literature

This connects to MAML (Model-Agnostic Meta-Learning) and learned optimizers. The difference: MAML learns initialization, while this learns a spatially-varying, temporally-smooth update rule. The glial field provides the "context" that makes the meta-learner spatially aware.

## Success Criteria

- Modulation field improves convergence speed by >10% over Adam on at least one task
- Optimal D is in an intermediate range (not 0, not infinity)
- Field develops visually meaningful spatial structure
- Full PDE provides measurable benefit over simple smoothing (otherwise, simplify)
- **Permuted embedding control shows reduced or no benefit** (confirming spatial structure matters, not just smoothing)
- **Modulation field provides benefit beyond KFAC** (confirming PDE dynamics add value over static preconditioning)

## Deliverables

- `src/modulation_field.py`: PDE solver on spatial graph
- `src/field_optimizer.py`: PyTorch optimizer wrapper that uses field multipliers
- `experiments/field_vs_adam.py`: Main comparison experiment
- `experiments/diffusion_sweep.py`: D parameter sweep
- `results/field_dynamics.mp4`: Animation of field evolution
- `results/pde_vs_heuristic.csv`: Comparison data

## Estimated Timeline

3-4 weeks. PDE implementation is straightforward; the sweep and visualization take time.

## Connection to Critical Reviews

The first review noted: "Whether the computational benefits of the glial system would pay for its simulation cost relative to simply making the standard network larger" is an open question. This experiment begins to answer it by measuring the overhead of the PDE solve vs. the benefit in convergence speed.

The second review noted: "The claim connects to natural gradient methods, information geometry, and structured preconditioners." We compare the modulation field to KFAC as an explicit baseline (condition 6 in the measurement protocol) to determine whether the PDE dynamics add value beyond known structured preconditioning.

The third review noted: "Add a permuted-embedding baseline to every experiment." Condition 5 (permuted embedding control) tests whether the modulation field's benefit comes from spatial structure or from smoothing-as-regularization. The review also reframes the modulation field as a spatially-structured preconditioner — if it approximately implements KFAC with a spatial constraint, understanding this equivalence tells us whether the PDE dynamics add anything over static KFAC and whether the spatial constraint is better or worse than Kronecker factoring.
