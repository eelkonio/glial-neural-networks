# Step 04: Characterizing Turing Instability Regimes

```
SIMULATION FIDELITY: Level-independent (analytical/mathematical)
SIGNAL MODEL: N/A (stability analysis of the PDE itself)
NETWORK STATE DURING INFERENCE: N/A
GLIAL INTERACTION WITH SIGNALS: N/A
NOTE: This analysis applies equally to Level 1, 2, and 3 implementations.
      The PDE parameters that are safe/unsafe don't depend on signal model.
```

## The Claim Being Tested

The second critical review identified that the reaction-diffusion dynamics of the modulation field can undergo Turing instabilities — spontaneous symmetry-breaking from uniform states to spatially patterned states. This can be beneficial (self-organizing domains) or pathological (runaway spatial collapse). This step maps the parameter space to identify safe operating regimes.

## Why This Matters

If we cannot characterize where the system is stable vs. unstable, we cannot deploy it reliably. This is the safety analysis that must precede any scaling.

## Background: Turing Instability Conditions

A reaction-diffusion system with activator u and inhibitor v:
```
du/dt = D_u * laplacian(u) + f(u, v)
dv/dt = D_v * laplacian(v) + g(u, v)
```

undergoes Turing instability when:
1. The homogeneous steady state is stable without diffusion
2. The inhibitor diffuses faster than the activator (D_v > D_u)
3. The activator is self-activating and the inhibitor is cross-inhibiting

In our system:
- **Activator**: Calcium signal that promotes plasticity (local, slow diffusion via IP3)
- **Inhibitor**: Adenosine/decay signal that suppresses plasticity (faster diffusion via ATP release)

## Experiment 4.1: Linear Stability Analysis

### Implementation

Linearize the astrocyte calcium dynamics around the homogeneous steady state. Compute the dispersion relation (growth rate as a function of spatial wavenumber).

```python
def linear_stability_analysis(params):
    """
    Compute dispersion relation for the astrocyte field.
    
    params: dict with D_ca, D_ip3, decay, coupling_strength, etc.
    
    Returns: growth_rate(k) for spatial wavenumber k
    """
    # Find homogeneous steady state
    ca_ss, ip3_ss = find_steady_state(params)
    
    # Compute Jacobian at steady state
    J = compute_jacobian(ca_ss, ip3_ss, params)
    
    # For each wavenumber k, compute eigenvalues of J - D*k^2
    wavenumbers = np.linspace(0, 10, 100)
    growth_rates = []
    
    for k in wavenumbers:
        D_matrix = np.diag([params['D_ca'], params['D_ip3']])
        M = J - D_matrix * k**2
        eigenvalues = np.linalg.eigvals(M)
        growth_rates.append(np.max(np.real(eigenvalues)))
    
    return wavenumbers, growth_rates
```

### What to Look For

- If max growth rate > 0 for some k > 0: **Turing unstable** (patterns will form)
- If max growth rate < 0 for all k: **Turing stable** (uniform state persists)
- The wavenumber with maximum growth rate determines the **characteristic domain size**

## Experiment 4.2: Parameter Space Mapping

### Protocol

Sweep the key parameters and classify each point as stable or unstable:

```python
parameter_grid = {
    'D_ca': [0.01, 0.05, 0.1, 0.5, 1.0],        # Calcium diffusion
    'D_ip3': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],   # IP3 diffusion  
    'coupling_strength': [0.01, 0.05, 0.1, 0.5],   # Gap junction strength
    'decay': [0.001, 0.01, 0.05, 0.1],             # Field decay rate
    'source_gain': [0.1, 0.5, 1.0, 2.0, 5.0],     # How strongly activity drives field
}
```

For each parameter combination:
1. Run linear stability analysis
2. If unstable: run full nonlinear simulation to see what pattern forms
3. Classify outcome: {stable_uniform, beneficial_domains, pathological_collapse, oscillatory}

### Visualization

Create a phase diagram showing:
- Axes: two most important parameters (likely D ratio and source_gain)
- Color: outcome category
- Boundary: the critical surface separating stable from unstable regimes

## Experiment 4.3: Nonlinear Simulation of Instabilities

### Implementation

For parameter points classified as Turing-unstable, run the full nonlinear astrocyte network and observe what happens:

```python
def simulate_instability(astrocyte_network, neural_network, data, n_steps=10000):
    """Run full system and track spatial pattern formation."""
    
    calcium_history = []
    performance_history = []
    
    for step in range(n_steps):
        # Neural network forward/backward
        loss = train_step(neural_network, data)
        
        # Astrocyte network update
        astrocyte_network.step(get_activations(), get_gradients())
        
        # Record
        calcium_history.append([u.ca for u in astrocyte_network.units])
        performance_history.append(loss)
    
    return calcium_history, performance_history
```

### Outcomes to Classify

**Beneficial domain formation:**
- Calcium field differentiates into distinct high/low regions
- High-calcium regions correspond to actively-learning parts of network
- Low-calcium regions correspond to stable, well-learned parts
- Network performance improves relative to uniform baseline

**Pathological collapse:**
- All calcium concentrates in one small region
- Most of the network becomes frozen (near-zero LR)
- Network performance degrades
- System does not recover

**Oscillatory instability:**
- Calcium oscillates globally with growing amplitude
- Learning becomes unstable (loss oscillates or diverges)
- No stable spatial pattern forms

**Traveling waves:**
- Calcium waves propagate continuously through the network
- May be beneficial (periodic plasticity sweeps) or harmful (disruption)

## Experiment 4.4: Designing Safe Operating Regimes

### The Goal

Identify parameter constraints that guarantee beneficial behavior:

```python
def is_safe_regime(params):
    """Return True if parameters are in the safe operating regime."""
    # Condition 1: Not Turing unstable (or only weakly unstable)
    wavenumbers, growth_rates = linear_stability_analysis(params)
    max_growth = np.max(growth_rates)
    
    if max_growth > PATHOLOGICAL_THRESHOLD:
        return False
    
    # Condition 2: Bounded modulation (field cannot exceed limits)
    if params['source_gain'] / params['decay'] > MAX_FIELD_RATIO:
        return False
    
    # Condition 3: Coupling not too strong (prevents global synchronization)
    if params['coupling_strength'] > SYNC_THRESHOLD:
        return False
    
    return True
```

### Deliverable

A set of **safe parameter constraints** that can be enforced in all subsequent experiments. These constraints define the "operating envelope" of the glial system.

## Experiment 4.5: Controlled Instability as a Feature

### The Question

Can we deliberately operate near (but not in) the Turing-unstable regime to get the benefits of self-organization without the risks of collapse?

### Protocol

1. Find the critical parameter surface (boundary between stable and unstable)
2. Set parameters just inside the stable regime (subcritical)
3. Compare performance to:
   - Deep in stable regime (far from instability)
   - Just inside unstable regime (supercritical)
   - Deep in unstable regime (strongly unstable)

### Hypothesis

The best performance may be at the "edge of instability" — where the system is maximally responsive to perturbations (critical dynamics) without actually destabilizing. This would be analogous to "edge of chaos" dynamics in other complex systems.

## Success Criteria

- Phase diagram clearly separates safe from unsafe parameter regions
- At least one safe regime produces beneficial domain formation
- Pathological collapse is reliably avoidable with identified constraints
- Edge-of-instability operation provides measurable benefit over deep-stable operation

## Deliverables

- `src/stability_analysis.py`: Linear stability analysis tools
- `src/phase_diagram.py`: Parameter sweep and classification
- `experiments/instability_simulation.py`: Full nonlinear simulations
- `results/phase_diagram.png`: 2D phase diagram with regime boundaries
- `results/safe_parameters.json`: Validated safe parameter ranges
- `results/instability_outcomes.mp4`: Animations of different instability types

## Estimated Timeline

2-3 weeks. Mostly computational (many simulations to run).

## Risk Assessment

**Risk**: The safe regime may be too narrow to be practical (tiny parameter window).
**Mitigation**: Add explicit clamping and damping mechanisms that widen the safe regime at the cost of reduced self-organization capability. Trade off between safety and emergent behavior.

## Go/No-Go Gate: Modulation Field Viability (from Critical Review 3)

**This step serves as a go/no-go gate for the entire modulation field approach.** If stable operating regions cannot be reliably identified, the reaction-diffusion modulation field (Steps 02-03) must be redesigned, not just parameter-tuned.

### Gate Criteria

**PASS** (proceed with modulation field as designed):
- Safe parameter region occupies at least 20% of the tested parameter space
- At least one safe regime produces measurable benefit over no-field baseline
- Pathological collapse is avoidable with simple, enforceable constraints

**CONDITIONAL PASS** (proceed with modifications):
- Safe region exists but is narrow (<20% of parameter space)
- Action: Add explicit clamping/damping mechanisms to widen the safe region
- Action: Reduce the PDE to a simpler spatial smoothing if full dynamics are too unstable

**FAIL** (redesign required):
- No safe parameter region produces benefit over baseline
- OR pathological collapse occurs unpredictably even within "safe" parameters
- OR the safe region is so narrow that practical use requires impractical precision in parameter setting

### If the Gate Fails

Options:
1. **Simplify to static spatial smoothing**: Replace the PDE with periodic Gaussian smoothing on the spatial graph (no dynamics, no instability risk). This loses temporal adaptation but retains spatial structure.
2. **Add hard constraints**: Enforce M ∈ [0.5, 2.0] with hard clamping, add momentum damping to prevent oscillation, use exponential moving average instead of PDE integration.
3. **Abandon continuous field, keep discrete domains**: Use only the astrocyte domain approach (Step 03) without the underlying continuous field. Domains provide spatial structure without the instability risk of a continuous PDE.
4. **Skip to Phase 2**: If spatial modulation under backprop is fundamentally unstable, proceed directly to local learning rules where glia play a constitutive role (the biological argument is stronger there anyway).
