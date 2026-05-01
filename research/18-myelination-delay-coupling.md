# Step 18: Myelination-Delay Coupling and Temporal Synchronization

```
SIMULATION FIDELITY: Level 2 (Temporal)
SIGNAL MODEL: Discrete delays — adaptive, controlled by oligodendrocyte system
NETWORK STATE DURING INFERENCE: Evolving (myelination changes slowly; delays change with it)
GLIAL INTERACTION WITH SIGNALS: At endpoints + delay modification
NOTE: This is where oligodendrocytes become computationally active. They don't just
      affect learning — they change the TIMING of signal arrival, which changes what
      the network computes during inference. This is the first step where a glial
      mechanism directly modifies the inference computation (not just learning dynamics).
```

## The Claim Being Tested

Oligodendrocyte-controlled adaptive myelination — which adjusts signal propagation delays based on activity patterns — enables temporal synchronization that improves network computation. The network learns not just WHAT to compute but WHEN signals should arrive.

## Why This Matters

At Level 1, myelination was just another way to modulate learning rates. At Level 2, it becomes something fundamentally different: a mechanism that controls the temporal structure of computation. Two signals that need to arrive simultaneously at a target neuron can be synchronized by differentially myelinating their axons. This is a computational capability that has NO analog in standard ANNs.

## Experiment 18.1: Synchronization Through Adaptive Myelination

### The Problem

Two source neurons (A and B) send signals to a target neuron (C). A is far from C; B is close to C. Without myelination, A's signal arrives much later than B's. With appropriate myelination, both signals can arrive simultaneously.

```
Without myelination:          With adaptive myelination:

A ----long path----> C        A ====fast path=====> C
     (slow, late)                  (myelinated, on time)

B --short--> C                B --short--> C
   (fast, early)                 (unmyelinated, on time)

Signal arrival at C:          Signal arrival at C:
B arrives first               A and B arrive together
A arrives later               -> Coincidence detection possible
-> No coincidence             -> Stronger activation of C
```

### Implementation

```python
class SynchronizationLearning:
    """Learn myelination patterns that synchronize signal arrival."""
    
    def __init__(self, network, target_neurons):
        self.network = network
        self.targets = target_neurons
        
    def compute_synchrony_error(self, target_id):
        """
        For a target neuron, measure how synchronous its inputs are.
        Perfect synchrony = all inputs arrive within a small time window.
        """
        arrival_times = self.network.get_arrival_times(target_id)
        if len(arrival_times) < 2:
            return 0.0
        
        # Synchrony error = variance of arrival times
        # (lower = more synchronous)
        return np.var(arrival_times)
    
    def update_myelination(self, learning_rate=0.0001):
        """
        Adjust myelination to reduce synchrony error.
        
        Rule: If a signal arrives TOO LATE, increase myelination (speed it up).
              If a signal arrives TOO EARLY, decrease myelination (slow it down).
        """
        for target_id in self.targets:
            arrival_times = self.network.get_arrival_times(target_id)
            mean_arrival = np.mean(arrival_times)
            
            for conn_id, arrival in zip(self.network.get_input_connections(target_id), arrival_times):
                # Error: how far from mean arrival time
                timing_error = arrival - mean_arrival
                
                # If arrives late (positive error): increase myelination (reduce delay)
                # If arrives early (negative error): decrease myelination (increase delay)
                myelin_update = -learning_rate * timing_error
                
                self.network.oligodendrocytes.myelination[conn_id] += myelin_update
                self.network.oligodendrocytes.myelination[conn_id] = np.clip(
                    self.network.oligodendrocytes.myelination[conn_id], 0, 1
                )
```

## Experiment 18.2: Myelination Interacts with Astrocyte Domains

### The Question

Do astrocyte domains and myelination patterns co-evolve? Does the astrocyte system influence which pathways get myelinated?

### Biological Basis

In biology:
- Astrocytes provide metabolic support to oligodendrocytes
- Astrocyte signals can promote or inhibit myelination
- High-activity regions (high astrocyte calcium) tend to get myelinated more

### Implementation

```python
def astrocyte_myelination_coupling(astrocyte_network, oligodendrocyte_system):
    """Astrocyte activity drives myelination of connections in active domains."""
    for astrocyte in astrocyte_network.units:
        if astrocyte.ca > MYELINATION_PROMOTION_THRESHOLD:
            # Connections in this domain get a myelination boost
            domain_connections = get_connections_in_domain(astrocyte.domain)
            for conn_id in domain_connections:
                oligodendrocyte_system.myelination[conn_id] += ASTROCYTE_MYELIN_DRIVE
```

### Measurement

- Do myelination patterns correlate with astrocyte domain activity?
- Does astrocyte-driven myelination improve synchronization faster than activity-only myelination?
- Does the combined system (astrocyte + oligodendrocyte) outperform either alone?

## Experiment 18.3: Temporal Computation Benchmarks

### Tasks That Require Precise Timing

**1. Temporal pattern classification**
- Input: sequences of spikes with specific timing patterns
- Task: classify which pattern is presented
- Requires: precise delay matching to detect temporal features

**2. Sound localization analog**
- Input: same signal arriving at two input groups with a time difference
- Task: determine which side the signal came from (interaural time difference)
- Requires: delay lines that compensate for the time difference

**3. Motor sequence generation**
- Output: precisely timed sequence of activations
- Task: generate a specific temporal pattern
- Requires: delay chains that produce the correct output timing

**4. Temporal binding**
- Input: features arriving at different times from different processing streams
- Task: bind features that belong to the same object (arrived from same source)
- Requires: delay equalization so related features arrive simultaneously at binding neurons

### Protocol

For each task, compare:
1. Fixed random delays (no myelination learning)
2. Fixed optimal delays (hand-tuned, upper bound)
3. Adaptive myelination (oligodendrocyte learning)
4. Adaptive myelination + astrocyte coupling

## Experiment 18.4: Myelination and Continual Learning

### The Question

Does myelination provide a DIFFERENT kind of memory protection than weight freezing?

### Insight

A myelinated pathway is:
- Fast (signals arrive quickly)
- Stable (myelination changes slowly)
- Temporally precise (timing is locked in)

This means old knowledge encoded in myelinated pathways is protected not just structurally (the connections exist) but TEMPORALLY (the timing relationships are preserved). New learning that changes delays elsewhere won't disrupt the temporal patterns of old knowledge.

### Protocol

1. Learn task A until myelination stabilizes
2. Learn task B (which requires different timing patterns)
3. Measure: does task A performance degrade?
4. Compare to: weight-only protection (EWC), topology-only protection (pruning)

### Expected Result

Myelination-based protection should be especially effective for tasks that depend on timing (temporal patterns, sequences, rhythms) — more so than weight or topology protection alone.

## Success Criteria

- Adaptive myelination achieves synchronization (arrival time variance reduced by >50%)
- Astrocyte-myelination coupling improves synchronization speed
- Network solves temporal tasks that fixed-delay networks cannot
- Myelination provides temporal memory protection distinct from weight protection

## Deliverables

- `src/synchronization_learning.py`: Myelination-based synchronization
- `src/astrocyte_myelin_coupling.py`: Astrocyte-oligodendrocyte interaction
- `experiments/synchronization.py`: Arrival time synchronization experiments
- `experiments/temporal_benchmarks.py`: Temporal computation tasks
- `experiments/temporal_memory.py`: Myelination as temporal memory protection
- `results/synchronization_curves.png`: Arrival time variance over training
- `results/myelination_patterns.png`: Spatial distribution of myelination

## Estimated Timeline

6-8 weeks. Builds on Step 17's temporal engine.
