# Step 17: Temporal Simulation Engine with Propagation Delays

```
SIMULATION FIDELITY: Level 2 (Temporal)
SIGNAL MODEL: Discrete delays — signals sent at time t arrive at time t + d(i,j)
NETWORK STATE DURING INFERENCE: Evolving (glial state updates during signal propagation)
GLIAL INTERACTION WITH SIGNALS: At endpoints (signals are modulated at source and destination, not in transit)
NOTE: This is the first step where the network operates in simulated TIME.
      A "forward pass" is no longer instantaneous — it is a temporal simulation
      where signals propagate, arrive at different times, and interact with
      a glial environment that is itself evolving.
```

## The Claim Being Tested

Adding propagation delays to the network — where each connection has a myelination-dependent delay — changes which glial mechanisms are beneficial and reveals timing-dependent phenomena invisible at Level 1.

## Why This Matters

This is the transition from "spatial geometry affects learning" to "spatial geometry affects computation." At Level 1, the spatial embedding only influenced how weights were updated. At Level 2, it determines WHEN signals arrive, which fundamentally changes what the network computes.

## Experiment 17.1: Build the Temporal Simulation Engine

### Core Architecture

```python
class TemporalNetwork:
    """Neural network with per-connection propagation delays."""
    
    def __init__(self, neurons, connections, positions, dt=0.1):
        """
        neurons: list of neuron objects with membrane dynamics
        connections: list of (source, target, weight, delay) tuples
        positions: (N_neurons, 3) spatial coordinates
        dt: simulation timestep (ms)
        """
        self.neurons = neurons
        self.connections = connections
        self.positions = positions
        self.dt = dt
        
        # Delay buffer: stores signals in transit
        # For each connection, a queue of (arrival_time, signal_value)
        self.delay_buffers = {conn_id: [] for conn_id in range(len(connections))}
        
        # Current simulation time
        self.t = 0.0
        
    def compute_delays_from_myelination(self, myelination_levels):
        """
        Convert myelination state to propagation delays.
        
        Unmyelinated: ~1 m/s conduction velocity
        Fully myelinated: ~100 m/s conduction velocity
        
        delay = distance / velocity
        """
        for i, conn in enumerate(self.connections):
            source_pos = self.positions[conn.source]
            target_pos = self.positions[conn.target]
            distance = np.linalg.norm(target_pos - source_pos)
            
            # Velocity depends on myelination (linear interpolation)
            velocity = 1.0 + 99.0 * myelination_levels[i]  # 1 to 100 m/s
            
            # Convert to timesteps
            conn.delay = int(np.ceil(distance / (velocity * self.dt)))
            conn.delay = max(1, conn.delay)  # Minimum 1 timestep
    
    def step(self):
        """Advance simulation by one timestep."""
        self.t += self.dt
        
        # 1. Deliver signals that have arrived at this timestep
        for conn_id, buffer in self.delay_buffers.items():
            conn = self.connections[conn_id]
            while buffer and buffer[0][0] <= self.t:
                arrival_time, signal = buffer.pop(0)
                # Deliver to target neuron
                self.neurons[conn.target].receive_input(signal * conn.weight)
        
        # 2. Update all neuron states (membrane dynamics)
        new_signals = []
        for neuron in self.neurons:
            output = neuron.update(self.dt)
            if output is not None:  # Neuron fired or produced output
                new_signals.append((neuron.id, output))
        
        # 3. Queue new signals with appropriate delays
        for source_id, signal in new_signals:
            for conn_id, conn in enumerate(self.connections):
                if conn.source == source_id:
                    arrival_time = self.t + conn.delay * self.dt
                    self.delay_buffers[conn_id].append((arrival_time, signal))
        
        # 4. Update glial state (at appropriate rate)
        if int(self.t / self.dt) % self.glial_update_interval == 0:
            self.update_glial_state()
        
        return new_signals
    
    def simulate(self, input_signal, duration_ms):
        """Run simulation for a given duration."""
        n_steps = int(duration_ms / self.dt)
        
        # Inject input
        self.inject_input(input_signal)
        
        # Run
        output_history = []
        for _ in range(n_steps):
            signals = self.step()
            output_history.append(self.read_output_neurons())
        
        return output_history
```

### Neuron Model (Leaky Integrate-and-Fire)

```python
class LIFNeuron:
    """Leaky integrate-and-fire neuron for temporal simulation."""
    
    def __init__(self, neuron_id, tau_mem=20.0, threshold=1.0, reset=0.0):
        self.id = neuron_id
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.reset = reset
        self.membrane = 0.0
        self.input_current = 0.0
        self.last_spike_time = -1000.0
        
    def receive_input(self, current):
        """Accumulate input current (from arriving signals)."""
        self.input_current += current
    
    def update(self, dt):
        """Update membrane potential, check for spike."""
        # Leak
        self.membrane += (-self.membrane / self.tau_mem + self.input_current) * dt
        self.input_current = 0.0  # Reset input accumulator
        
        # Spike check
        if self.membrane >= self.threshold:
            self.membrane = self.reset
            self.last_spike_time = self.t
            return 1.0  # Spike
        
        return None  # No spike
```

## Experiment 17.2: Re-validate Phase 1 Results with Delays

### The Critical Question

Do the mechanisms that helped at Level 1 (instantaneous signals) still help at Level 2 (delayed signals)?

### Protocol

Take the best-performing configurations from Phase 1 (Steps 02-10) and re-implement them in the temporal simulation engine. Compare:

1. Level 1 result: benefit of mechanism X under instantaneous signals
2. Level 2 result: benefit of mechanism X under delayed signals

For each mechanism:
- Astrocyte modulation field
- Microglia pruning
- Volume transmission
- Multi-timescale dynamics
- Full ecosystem

### Possible Outcomes

- **Results transfer**: All Level 1 benefits persist at Level 2 (delays are orthogonal)
- **Some transfer**: Certain mechanisms help more/less with delays (interaction effects)
- **New phenomena**: Level 2 reveals benefits invisible at Level 1 (timing-dependent)
- **Results reverse**: Something that helped at Level 1 hurts at Level 2 (conflict with timing)

## Experiment 17.3: Delay-Dependent Computation

### The Question

Can the network perform computations that are IMPOSSIBLE without delays? (Temporal pattern recognition, coincidence detection, sequence processing)

### Tasks That Require Timing

- **Temporal XOR**: Output depends on the ORDER of two inputs, not just their values
- **Coincidence detection**: Output fires only when two signals arrive simultaneously
- **Sequence recognition**: Distinguish ABC from CBA (same elements, different order)
- **Rhythm detection**: Respond to periodic input at a specific frequency

### Protocol

Test each task with:
1. Level 1 network (no delays) — should FAIL on timing-dependent tasks
2. Level 2 network (fixed delays) — should succeed if delays are appropriate
3. Level 2 network (adaptive delays via myelination) — should learn optimal delays

## Experiment 17.4: Myelination as Learnable Delay

### Implementation

```python
class OligodendrocyteSystem:
    """Adapts connection delays based on activity patterns."""
    
    def __init__(self, connections, learning_rate=0.001):
        self.connections = connections
        self.myelin_lr = learning_rate
        self.myelination = np.zeros(len(connections))  # 0 to 1
        self.activity_history = np.zeros(len(connections))
        
    def update(self, connection_activity, target_synchrony):
        """
        Increase myelination (reduce delay) for connections whose signals
        need to arrive more synchronously at their targets.
        """
        for i, conn in enumerate(self.connections):
            # Track activity
            self.activity_history[i] = 0.99 * self.activity_history[i] + 0.01 * connection_activity[i]
            
            # Myelination drive: active connections get myelinated
            # (biological: activity-dependent myelination)
            drive = self.activity_history[i]
            
            # Update myelination (slow, very slow timescale)
            self.myelination[i] += self.myelin_lr * (drive - self.myelination[i])
            self.myelination[i] = np.clip(self.myelination[i], 0, 1)
```

### The Question

Does learnable myelination (adaptive delays) improve performance on temporal tasks compared to fixed delays?

## Success Criteria

- Temporal simulation engine runs correctly (signals arrive at correct times)
- At least 2 of 5 Phase 1 mechanisms still provide benefit at Level 2
- Network can solve timing-dependent tasks that Level 1 cannot
- Adaptive myelination improves temporal task performance over fixed delays
- Glial mechanisms interact with timing (not just orthogonal)

## Deliverables

- `src/temporal_engine.py`: Full temporal simulation engine
- `src/lif_neuron.py`: Leaky integrate-and-fire neuron model
- `src/delay_buffer.py`: Efficient delay queue implementation
- `src/oligodendrocyte.py`: Myelination-delay coupling system
- `experiments/level1_vs_level2.py`: Re-validation of Phase 1 results
- `experiments/temporal_tasks.py`: Timing-dependent task benchmarks
- `results/level_transfer.csv`: Which mechanisms transfer from Level 1 to Level 2
- `results/temporal_computation.png`: Tasks solvable at Level 2 but not Level 1

## Estimated Timeline

8-10 weeks. Building the temporal engine is substantial engineering work.

## Hardware Requirements

Level 2 simulation is 10-100x more expensive than Level 1. Expect:
- Small networks (1000 neurons): single GPU, minutes per experiment
- Medium networks (10000 neurons): single GPU, hours per experiment
- Large networks (100000 neurons): multi-GPU, days per experiment
