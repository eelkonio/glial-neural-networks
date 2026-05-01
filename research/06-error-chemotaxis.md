# Step 06: Microglia Clustering at High-Error Regions

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static
GLIAL INTERACTION WITH SIGNALS: Learning-only
NOTE: Error is computed instantaneously via backprop. At Level 2+, error would be
      a temporally-extended signal that microglia detect over time, making chemotaxis
      more biologically faithful but also slower to respond.
```

## The Claim Being Tested

When microglia agents migrate toward regions of high error via chemotaxis-like dynamics, they cluster where the network is struggling most. This clustering concentrates pruning and remodeling effort where it's needed, reducing chaotic learning behavior by removing redundant or conflicting connections in problematic regions.

## Why This Matters

This is the core "self-healing" claim: the network automatically directs maintenance resources to its weakest points. If this works, it means the network can self-diagnose and self-repair without external intervention — a property no current ANN possesses.

## Experiment 6.1: Error Landscape Mapping

### Implementation

Before testing chemotaxis, we need to measure the spatial distribution of error in the network.

```python
class ErrorLandscape:
    """Maps per-weight error contribution to the spatial embedding."""
    
    def __init__(self, model, weight_positions):
        self.model = model
        self.positions = weight_positions
        self.error_map = np.zeros(len(weight_positions))
        self.gradient_variance_map = np.zeros(len(weight_positions))
        self.conflict_map = np.zeros(len(weight_positions))
    
    def update(self, data_batch, n_samples=10):
        """Compute per-weight error contribution over multiple mini-batches."""
        gradient_history = []
        
        for _ in range(n_samples):
            batch = next(data_batch)
            loss = self.model.compute_loss(batch)
            grads = self.model.compute_per_weight_gradients(loss)
            gradient_history.append(grads)
        
        gradient_history = np.stack(gradient_history)
        
        # Error contribution: mean absolute gradient (how much this weight needs to change)
        self.error_map = np.mean(np.abs(gradient_history), axis=0)
        
        # Gradient variance: how inconsistent the gradient direction is across batches
        # High variance = conflicting signals = chaotic learning
        self.gradient_variance_map = np.var(gradient_history, axis=0)
        
        # Conflict score: fraction of sign disagreements across batches
        signs = np.sign(gradient_history)
        self.conflict_map = 1.0 - np.abs(np.mean(signs, axis=0))
        # conflict_map near 0 = consistent direction
        # conflict_map near 1 = gradient flips sign every batch (chaos)
    
    def get_attraction_field(self):
        """Combine signals into a single attraction field for microglia."""
        return (
            0.3 * self.error_map / (self.error_map.max() + 1e-8) +
            0.4 * self.gradient_variance_map / (self.gradient_variance_map.max() + 1e-8) +
            0.3 * self.conflict_map
        )
```

### Visualization

Plot the error landscape on the spatial embedding:
- 3D scatter plot of weight positions
- Color = error contribution (red = high error, blue = low)
- Size = gradient variance (large = chaotic)
- Overlay agent positions as markers

## Experiment 6.2: Chemotaxis Dynamics

### Implementation

Agents migrate up the gradient of the attraction field:

```python
def chemotaxis_step(agent, attraction_field, weight_positions, dt=0.01):
    """Move agent toward high-attraction regions."""
    
    # Compute local gradient of attraction field at agent position
    # (finite difference using nearby weight positions)
    territory = agent.get_territory(weight_positions)
    
    if len(territory) < 3:
        return  # Not enough points to estimate gradient
    
    local_positions = weight_positions[territory]
    local_values = attraction_field[territory]
    
    # Weighted direction toward high-attraction weights
    diffs = local_positions - agent.position
    distances = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8
    directions = diffs / distances
    
    # Weight by attraction value and inverse distance
    weights = local_values / (distances.squeeze() + 0.01)
    gradient_estimate = np.average(directions, axis=0, weights=weights)
    
    # Apply chemotaxis
    agent.velocity += dt * gradient_estimate * CHEMOTAXIS_STRENGTH
    agent.position += agent.velocity * dt
    
    # Damping
    agent.velocity *= 0.95
```

### Measurement: Clustering Behavior

Track agent density over time:
```python
def measure_clustering(agents, weight_positions, attraction_field, n_bins=20):
    """Measure correlation between agent density and error density."""
    # Bin the spatial embedding
    # Count agents per bin
    # Count error per bin
    # Compute correlation
    agent_density = spatial_histogram(agent_positions, n_bins)
    error_density = spatial_histogram_weighted(weight_positions, attraction_field, n_bins)
    
    return pearsonr(agent_density.flatten(), error_density.flatten())
```

## Experiment 6.3: Does Clustering Reduce Chaotic Learning?

### The Core Experiment

Train a network on a task with known "difficult" regions (e.g., a multi-task setup where some tasks conflict). Measure whether microglia clustering at conflict regions reduces gradient variance and improves convergence.

### Setup

**Task**: Multi-head MLP where different output heads have partially conflicting gradients for shared weights. This creates natural "conflict zones" in the network.

**Protocol**:
1. Train with no microglia (baseline) — measure gradient variance over time
2. Train with microglia (no chemotaxis) — agents at random positions
3. Train with microglia (with chemotaxis) — agents migrate to high-conflict regions
4. Train with microglia (chemotaxis + pruning) — agents cluster AND prune conflicting weights

**Measurements**:
- Gradient variance in conflict regions over time
- Loss convergence curve
- Final accuracy on each task
- Number of weights pruned in conflict vs. non-conflict regions

### Expected Result

When agents cluster at conflict regions and prune redundant/conflicting connections:
- Gradient variance should decrease (less chaos)
- Convergence should improve (cleaner gradient signal)
- The network should develop cleaner task separation (less interference)

## Experiment 6.4: Clustering Dynamics and Timescales

### The Question

How quickly do agents cluster? Does clustering track changes in the error landscape as training progresses?

### Protocol

1. Record agent positions and error landscape every 100 training steps
2. Compute agent-error correlation at each checkpoint
3. Introduce a sudden distribution shift at step 5000 (new error landscape)
4. Measure: how quickly do agents re-cluster to the new error regions?

### Expected Result

- Agents should cluster within ~500-1000 steps (depending on migration speed)
- After distribution shift, agents should migrate to new error regions
- Migration speed determines adaptation speed to new problems

## Experiment 6.5: Preventing Over-Clustering

### The Problem

If all agents cluster in one region, the rest of the network is unmonitored. Need territorial repulsion to maintain coverage.

### Protocol

Sweep repulsion strength:
- repulsion = 0: agents all cluster at single highest-error point (pathological)
- repulsion = low: agents cluster loosely, some coverage gaps
- repulsion = medium: agents cluster at error regions but maintain spacing
- repulsion = high: agents spread uniformly (no clustering benefit)

### Measurement

- Coverage: fraction of network within at least one agent's territory
- Clustering quality: correlation between agent density and error density
- Pruning quality: accuracy at target sparsity

### Expected Result

Optimal repulsion balances clustering (directed effort) with coverage (no blind spots).

## Experiment 6.6: Emergent Division of Labor

### The Question

Do agents that cluster together develop specialized roles? (e.g., some focus on pruning, others on monitoring)

### Implementation

Give agents internal state that can differentiate:
```python
class SpecializableMicrogliaAgent(MicrogliaAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specialization = 0.5  # 0 = pure monitor, 1 = pure pruner
        
    def update_specialization(self, local_agent_density):
        """If many agents nearby, some should specialize in monitoring
        while others specialize in pruning."""
        if local_agent_density > CROWDING_THRESHOLD:
            # Differentiate: become more extreme in current tendency
            if self.specialization > 0.5:
                self.specialization += 0.01  # Become more pruner-like
            else:
                self.specialization -= 0.01  # Become more monitor-like
```

### Measurement

Track whether agents in clusters develop different specializations and whether this improves outcomes vs. uniform agents.

## Success Criteria

- Agent-error density correlation > 0.6 after clustering stabilizes
- Chemotaxis-guided pruning reduces gradient variance in conflict regions by >30%
- Convergence speed improves by >15% compared to random-position agents
- Agents successfully re-cluster after distribution shift within reasonable time
- Territorial repulsion maintains >80% network coverage while allowing clustering

## Deliverables

- `src/error_landscape.py`: Per-weight error and conflict mapping
- `src/chemotaxis.py`: Chemotaxis migration dynamics
- `experiments/clustering_dynamics.py`: Agent clustering measurement
- `experiments/chaos_reduction.py`: Gradient variance reduction experiment
- `experiments/distribution_shift.py`: Re-clustering after shift
- `results/clustering_animation.mp4`: Agent migration and clustering over training
- `results/chaos_reduction.png`: Gradient variance before/after clustering

## Estimated Timeline

3-4 weeks. Builds directly on Step 05 infrastructure.

## Connection to Critical Reviews

The first review noted that the "emerges from glial remodeling" claims are premature. This experiment provides the first concrete test: does directed pruning at error regions actually produce better network structure than undirected pruning? If yes, it's evidence for self-organizing architecture. If no, the claim needs revision.
