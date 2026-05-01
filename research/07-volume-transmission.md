# Step 07: Broadcast Modulation via Diffusion Fields

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static
GLIAL INTERACTION WITH SIGNALS: Learning-only (diffusion field modulates plasticity, not signals)
NOTE: At Level 2+, volume-transmitted chemicals would also modulate signal GAIN during
      propagation (e.g., adenosine suppressing synaptic transmission in real-time).
      At Level 3, signals passing through high-concentration regions would be attenuated
      or amplified during transit. This is the mechanism most transformed by Level 3.
```

## The Claim Being Tested

Volume transmission — releasing a modulatory signal that diffuses outward through space affecting all weights within range regardless of network topology — provides a communication mechanism that is computationally distinct from and complementary to edge-based (connection-based) information flow. It enables O(1) broadcast to spatially local regions without requiring explicit connections.

## Why This Matters

This tests whether topology-independent spatial communication adds value. If it does, it means the spatial embedding provides a communication channel that the network graph alone cannot replicate efficiently.

## Experiment 7.1: Implement Volume Transmission

### Core Mechanism

```python
class VolumeTransmitter:
    """Simulates release and diffusion of modulatory signals in the spatial embedding."""
    
    def __init__(self, weight_positions, D=0.05, decay=0.02, dt=1.0):
        self.positions = weight_positions
        self.N = len(weight_positions)
        self.D = D
        self.decay = decay
        self.dt = dt
        
        # Precompute distance matrix (or use sparse approximation for large N)
        self.dist_matrix = cdist(weight_positions, weight_positions)
        
        # Field state: concentration at each weight position
        self.field = np.zeros(self.N)
        
        # Active release events
        self.releases = []  # list of (position, amount, time_remaining)
    
    def release(self, source_position, amount, duration=10):
        """Trigger a release event at a spatial position."""
        self.releases.append({
            'position': source_position,
            'amount': amount,
            'remaining': duration
        })
    
    def step(self):
        """Advance diffusion by one timestep."""
        # Add contributions from active releases
        for release in self.releases:
            distances = np.linalg.norm(self.positions - release['position'], axis=1)
            # Gaussian source (concentrated near release point)
            contribution = release['amount'] * np.exp(-distances**2 / (2 * 0.05**2))
            self.field += contribution * self.dt
            release['remaining'] -= 1
        
        # Remove expired releases
        self.releases = [r for r in self.releases if r['remaining'] > 0]
        
        # Diffusion (discrete Laplacian on k-nearest neighbor graph)
        new_field = np.copy(self.field)
        for i in range(self.N):
            # Average of neighbors minus self (discrete Laplacian)
            neighbors = self.get_neighbors(i)
            if len(neighbors) > 0:
                laplacian_i = np.mean(self.field[neighbors]) - self.field[i]
                new_field[i] += self.D * laplacian_i * self.dt
        
        # Decay
        new_field -= self.decay * new_field * self.dt
        
        self.field = np.clip(new_field, -2.0, 2.0)
        
    def get_modulation(self):
        """Return current field as modulation signal."""
        return self.field
```

### ATP-like Concentric Ring Signaling

```python
class ATPSignaling(VolumeTransmitter):
    """Models ATP release with enzymatic conversion creating distance-dependent effects."""
    
    def get_modulation(self):
        """ATP near source (excitatory), adenosine far from source (inhibitory)."""
        modulation = np.zeros(self.N)
        
        for release in self.releases:
            distances = np.linalg.norm(self.positions - release['position'], axis=1)
            
            # Inner ring: ATP dominant (excitatory, increases LR)
            inner_mask = distances < 0.03
            modulation[inner_mask] += 0.5 * release['amount']
            
            # Middle ring: transitional
            middle_mask = (distances >= 0.03) & (distances < 0.07)
            modulation[middle_mask] += 0.0  # Neutral
            
            # Outer ring: adenosine dominant (inhibitory, decreases LR)
            outer_mask = (distances >= 0.07) & (distances < 0.12)
            modulation[outer_mask] -= 0.3 * release['amount']
        
        return modulation
```

## Experiment 7.2: Volume Transmission vs. Explicit Broadcast Connections

### The Question

Can volume transmission achieve something that would require O(n) explicit connections in a standard network?

### Setup

Create a scenario where a "broadcast" signal is needed: when one region of the network detects an anomaly, all nearby regions should be alerted.

**Task**: Network trained on CIFAR-10 with periodic injection of adversarial examples. The network needs to detect and suppress adversarial influence.

**Implementations**:
1. **No broadcast**: Each weight region handles adversarial examples independently
2. **Explicit broadcast**: Add explicit connections from anomaly detector to all weights (O(n) connections)
3. **Volume transmission**: Anomaly detection triggers a release event that diffuses to nearby weights

**Measurement**:
- Robustness to adversarial examples
- Computational cost of broadcast mechanism
- Speed of response (how quickly does the alert reach distant weights?)

### Expected Result

Volume transmission should achieve similar robustness to explicit broadcast but with O(1) release events instead of O(n) connections. The tradeoff is speed: diffusion is slower than direct connections.

## Experiment 7.3: Trigger Conditions for Release Events

### The Question

What should trigger a volume transmission release? Options:

**A. Astrocyte distress**: When an astrocyte's calcium exceeds a threshold, it releases ATP
```python
def check_release_trigger(astrocyte):
    if astrocyte.ca > DISTRESS_THRESHOLD:
        return Release(position=astrocyte.center, amount=astrocyte.ca - DISTRESS_THRESHOLD)
    return None
```

**B. Error spike**: When local loss suddenly increases
```python
def check_release_trigger(local_loss, local_loss_history):
    if local_loss > 2 * np.mean(local_loss_history[-10:]):
        return Release(position=region_center, amount=local_loss - baseline)
    return None
```

**C. Pruning event**: When a microglia agent prunes a weight, it releases a signal
```python
def on_prune(agent, pruned_weight_position):
    # Alert nearby weights that topology has changed
    return Release(position=pruned_weight_position, amount=0.5)
```

**D. Periodic heartbeat**: Regular releases that maintain baseline field
```python
def periodic_release(astrocytes, step, period=100):
    if step % period == 0:
        for a in astrocytes:
            return Release(position=a.center, amount=0.1)
```

### Protocol

Test each trigger condition independently and in combination. Measure which triggers produce useful modulation patterns vs. noise.

## Experiment 7.4: Information Content of the Diffusion Field

### The Question

Does the spatial pattern of the diffusion field encode useful information about network state? Can a simple readout of the field predict network behavior?

### Protocol

1. Train network with volume transmission active
2. At each checkpoint, record the field state M(x,y,z)
3. Train a simple linear probe: field_state → {loss_next_100_steps, accuracy_next_100_steps}
4. If the probe has high accuracy, the field encodes predictive information

### Expected Result

If the field encodes meaningful information, it validates the "spatially distributed memory" claim — the field state is a form of working memory about recent network activity.

## Experiment 7.5: Multi-Channel Volume Transmission

### The Question

Biological glia use multiple chemical channels simultaneously (ATP, glutamate, D-serine, cytokines). Does having multiple independent diffusion fields (each with different diffusion rates and effects) provide benefit over a single field?

### Implementation

```python
class MultiChannelVolumeTransmission:
    """Multiple independent diffusion fields with different properties."""
    
    def __init__(self, weight_positions):
        # Fast channel: rapid diffusion, quick decay (alert signal)
        self.fast = VolumeTransmitter(weight_positions, D=0.5, decay=0.1)
        
        # Slow channel: slow diffusion, persistent (context signal)
        self.slow = VolumeTransmitter(weight_positions, D=0.01, decay=0.005)
        
        # Inhibitory channel: medium diffusion, suppressive effect
        self.inhibitory = VolumeTransmitter(weight_positions, D=0.1, decay=0.05)
    
    def get_modulation(self):
        """Combine channels into net modulation."""
        lr_mod = 1.0 + self.fast.get_modulation() + 0.5 * self.slow.get_modulation()
        lr_mod -= self.inhibitory.get_modulation()
        return np.clip(lr_mod, 0.1, 5.0)
```

### Measurement

Compare single-channel vs. multi-channel volume transmission on:
- Adaptation speed (fast channel should help)
- Context persistence (slow channel should help)
- Preventing runaway activation (inhibitory channel should help)

## Success Criteria

- Volume transmission achieves broadcast-like coordination with O(1) cost per event
- Field state encodes predictive information about network behavior (probe accuracy > 60%)
- Multi-channel transmission outperforms single-channel
- At least one trigger condition produces clearly beneficial modulation patterns
- Volume transmission + astrocytes outperforms either alone

## Deliverables

- `src/volume_transmission.py`: VolumeTransmitter and ATPSignaling classes
- `src/multi_channel.py`: Multi-channel implementation
- `experiments/broadcast_comparison.py`: Volume vs. explicit broadcast
- `experiments/field_information.py`: Linear probe on field state
- `results/diffusion_patterns.mp4`: Visualization of field evolution
- `results/broadcast_efficiency.csv`: Cost and effectiveness comparison

## Estimated Timeline

3 weeks. Builds on spatial infrastructure from earlier steps.
