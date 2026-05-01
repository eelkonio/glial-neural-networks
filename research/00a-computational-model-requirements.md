# Computational Model Requirements: From Null-Space to Full Temporal-Spatial Simulation

## Purpose

This document defines the fundamental computational model differences between standard artificial neural networks and the glia-augmented spatial-geometric network we are building. It establishes three simulation fidelity levels, explains what each level captures and what it sacrifices, and maps the research steps to a progression from the simplest model toward a full temporal-spatial simulation.

---

## The Three Fundamental Departures from Standard ANNs

### Departure 1: Signal Propagation Is Not Instantaneous

**Standard ANN (null-space model)**:
```
layer_2 = activation(W_12 @ layer_1 + bias_2)
```
This computation is instantaneous. The signal from every neuron in layer 1 arrives at every neuron in layer 2 at the same logical moment. There is no concept of "this connection is faster" or "this signal hasn't arrived yet." The network exists outside of time during inference — it is a pure mathematical function evaluation.

**Biological reality**:
- A signal from neuron A takes time to reach neuron B
- That time depends on axon length and myelination state
- Heavily myelinated axons conduct at ~100 m/s; unmyelinated at ~1 m/s
- This means signals from different sources arrive at a target neuron at DIFFERENT TIMES
- The relative timing of arrivals determines the computation (STDP, coincidence detection)
- Oligodendrocytes actively adjust myelination, changing these delays over days/weeks

**What this means for our system**:
Every connection in the network has a **delay parameter** d(i,j) that determines when a signal sent from neuron i at time t arrives at neuron j. This delay is:
- Determined by the spatial distance between i and j in the embedding
- Modified by the myelination state of that connection (oligodendrocyte control)
- Variable over time (myelination changes with learning)
- Computationally significant (arrival timing affects downstream computation)

A "forward pass" is no longer a single matrix multiply. It is a **temporal simulation** where signals propagate through the network over many timesteps, arriving at different neurons at different times.

### Departure 2: Signals Traverse Glial Regions In Transit

**Standard ANN (null-space model)**:
```
signal_at_B = W_AB * activation_A
```
The signal teleports from A to B. Nothing happens to it in between. There is no "between."

**Biological reality**:
- An axon from neuron A to neuron B physically passes through tissue
- That tissue contains astrocyte domains, microglial territories, chemical fields
- As the signal (action potential) travels along the axon, it:
  - Passes through different astrocyte domains (which may modulate it)
  - Triggers neurotransmitter release at en-passant synapses along the way
  - Is affected by the local ionic environment (maintained by astrocytes)
  - Can be detected by microglia surveilling that region
- The signal's properties can change during transit based on the glial environment it traverses

**What this means for our system**:
A signal from spatial position A to spatial position N doesn't just get multiplied by a weight. It **traverses intermediate space**, and that space has properties:

```
Signal path from A to N:

  A ──[astrocyte domain 1]──[astrocyte domain 2]──[domain 3]── N
       |                      |                     |
       modulation_1           modulation_2          modulation_3
       (gain, delay)          (gain, delay)         (gain, delay)

Effective signal at N = signal_A * product(modulations along path) 
                        arriving at time = t_send + sum(delays along path)
```

Each segment of the path passes through a glial region that can:
- Attenuate or amplify the signal (astrocyte gain control)
- Add additional delay (local ionic conditions)
- Block the signal entirely (reactive astrogliosis, glial scarring)
- Trigger glial responses (the signal's passage is itself an event that glia detect)

### Departure 3: Multi-Timescale Glial State Modifies Signals Continuously

**Standard ANN (null-space model)**:
The network state (weights) is fixed during the forward pass. It only changes during the backward pass / optimizer step. Inference is stateless.

**Biological reality**:
The network state is NEVER static. At every moment, multiple processes are running simultaneously:

```
During a single "forward pass" (which takes real time):

FAST (milliseconds):
  - Neurons fire, signals propagate
  - Astrocyte processes detect neurotransmitter at synapses
  - Local ionic concentrations shift

MEDIUM (seconds):
  - Astrocyte calcium rises in response to detected activity
  - Calcium-dependent D-serine release gates plasticity
  - Gliotransmitter release modulates nearby synapses
  - Volume-transmitted signals begin diffusing outward

SLOW (minutes-hours):
  - Astrocyte calcium waves propagate across domains
  - Microglia detect sustained activity patterns
  - Microglia begin migrating toward active regions
  - Gap junction conductance adjusts

VERY SLOW (hours-days):
  - Microglia execute pruning decisions
  - Myelination state changes
  - New synapses form in cleared regions
  - Domain boundaries shift
```

All of these are happening SIMULTANEOUSLY and INTERACTING with the signals propagating through the network. A signal that takes 10ms to propagate from A to N encounters a glial environment that is itself evolving during those 10ms.

**What this means for our system**:
The simulation must maintain and update multiple state variables at different rates, all of which can interact with signals in transit:

```
State variables (updated at different rates):

Per-connection:
  - weight value (changes with learning rule)
  - delay value (changes with myelination)
  - myelination level (changes very slowly)
  - pruning mask (binary, changes with microglia decisions)

Per-astrocyte-domain:
  - calcium concentration (changes every ~100 neural timesteps)
  - IP3 concentration (changes every ~50 neural timesteps)
  - D-serine output level (changes every ~200 neural timesteps)
  - gap junction conductance to neighbors (changes every ~10000 timesteps)

Per-microglia-agent:
  - position (changes every ~1000 neural timesteps)
  - state (surveilling/activated/pruning)
  - evidence accumulators (updated every ~500 timesteps)
  - velocity (changes every ~1000 timesteps)

Global fields:
  - volume-transmitted chemical concentrations (PDE, updated every ~100 timesteps)
  - extracellular potassium map (updated every ~10 timesteps)
```

---

## Three Simulation Fidelity Levels

Given these requirements, we define three levels of simulation fidelity. The research plan progresses through these levels.

### Level 1: Null-Space with Spatial Learning Dynamics

**What it is**: Standard instantaneous forward/backward pass. The spatial geometry affects only the LEARNING process (which weights change and how fast), not the INFERENCE process (how signals propagate).

**What it captures**:
- Spatial correlation of learning rates (astrocyte domains)
- Spatially-informed pruning (microglia)
- Multi-timescale learning dynamics
- Volume-transmitted modulation of plasticity

**What it sacrifices**:
- Signal propagation delays (all signals are instantaneous)
- In-transit glial interaction (signals teleport, no intermediate space)
- Timing-dependent computation (no STDP possible without timing)
- Myelination effects on inference (only affects learning rate, not signal speed)

**Implementation**: Standard PyTorch. Forward pass is `y = f(W @ x)`. Glial system runs as a sidecar that modifies W, learning rates, and masks between training steps.

**When to use**: Phase 1 (Steps 01-11). Establishes whether spatial geometry helps learning at all, before investing in full temporal simulation.

---

### Level 2: Temporal Simulation with Discrete Delays

**What it is**: The network operates in discrete timesteps. Each connection has a delay (integer number of timesteps). Signals are queued and delivered at the correct time. Glial state evolves in parallel with signal propagation.

**What it captures**:
- Signal propagation delays (myelination-dependent)
- Timing-dependent computation (STDP, coincidence detection)
- Signals arriving at different times from different sources
- Glial state evolving during signal propagation
- Myelination affecting both learning AND inference

**What it sacrifices**:
- Continuous in-transit modulation (signals are "in flight" but not modified mid-flight)
- Sub-timestep precision (delays are quantized to timestep resolution)
- Spatial continuity of signal path (signal jumps from source to destination after delay, doesn't traverse intermediate space)

**Implementation**: Spiking neural network simulator with delay queues (Brian2CUDA, custom engine). Each timestep: (1) deliver queued signals, (2) update neuron states, (3) generate new signals with delays, (4) update glial state at appropriate rate.

**When to use**: Phase 2 (Steps 12-16) and transition steps. Tests whether temporal dynamics and STDP interact with glial mechanisms as biology predicts.

---

### Level 3: Full Temporal-Spatial Simulation

**What it is**: Signals propagate through a continuous spatial medium. As they traverse space, they interact with the glial environment at each point along their path. The glial environment is itself a dynamic, spatially continuous system (reaction-diffusion fields). Everything interacts with everything in real time.

**What it captures**:
- Everything from Level 2, plus:
- In-transit signal modulation (signals are modified as they pass through glial regions)
- Continuous spatial propagation (not just source-to-destination, but through intermediate space)
- Bidirectional interaction: signals affect glia they pass through, glia affect signals passing through them
- Emergent phenomena from spatial continuity (wave interference, resonance, spatial filtering)

**What it sacrifices**:
- Computational efficiency (this is expensive)
- Simplicity (many interacting continuous systems)
- Analytical tractability (too complex for closed-form analysis)

**Implementation**: Custom CUDA simulation engine. The network is a spatial volume. Signals propagate as wavefronts through this volume. The volume has spatially varying properties (determined by glial state). The glial state is itself a set of coupled PDEs evolving on the same spatial grid.

**When to use**: Phase 3 (Steps 17-20, defined below). The culminating implementation that tests whether full spatial-temporal fidelity produces qualitatively different behavior from the simplified models.

---

## Comparison Table

| Property | Level 1 (Null-Space) | Level 2 (Temporal) | Level 3 (Full Spatial-Temporal) |
|----------|---------------------|-------------------|-------------------------------|
| Forward pass | Instantaneous matrix multiply | Timestep simulation with delays | Continuous spatial propagation |
| Signal speed | Infinite (all same) | Finite, per-connection | Finite, spatially varying |
| Myelination effect | Learning rate only | Signal delay | Signal delay + in-transit properties |
| Glial interaction with signals | Between passes only | At source and destination | Continuously along path |
| STDP possible | No | Yes | Yes |
| Spatial traversal | No (teleportation) | No (delayed teleportation) | Yes (continuous propagation) |
| Computational cost | 1x (baseline) | 10-100x | 100-1000x |
| Implementation | PyTorch | SNN simulator + custom | Custom CUDA engine |
| Biological fidelity | Low | Medium | High |

---

## CUDA Library Support by Level

### Level 1: Fully Supported by Existing Libraries

| Operation | Library | Notes |
|-----------|---------|-------|
| Forward/backward pass | PyTorch / cuDNN | Standard, no modification |
| Per-weight LR modulation | PyTorch optimizer | Element-wise multiply |
| Weight masking | PyTorch | Binary mask tensor |
| PDE field solver | Custom (sparse matmul) | Uses torch.sparse or cuSPARSE |
| Astrocyte dynamics | Custom (batched ODE) | Vectorized tensor ops |
| Agent simulation | CPU-side | Not GPU-friendly |

### Level 2: Partially Supported

| Operation | Library | Notes |
|-----------|---------|-------|
| Neuron state updates | Brian2CUDA / Norse / custom | Parallel over neurons |
| Delay queue management | Brian2CUDA / custom | Per-synapse delay buffers |
| Spike delivery | Brian2CUDA / custom | Scatter operations |
| STDP computation | Brian2CUDA / custom | Local to each synapse |
| Glial state updates | Custom | Batched, runs at slower rate |
| Myelination-delay coupling | Custom | Updates delay values |

**Key gap**: No existing library integrates SNN simulation with glial dynamics. Brian2CUDA handles neurons and synapses with delays; the glial layer must be added as custom code.

### Level 3: Requires Custom Engine

| Operation | Possible approach | Notes |
|-----------|------------------|-------|
| Spatial signal propagation | Lattice Boltzmann or finite difference | Wave equation on 3D grid |
| Spatially varying medium properties | Per-voxel parameter arrays | Updated by glial state |
| Glial field evolution | Reaction-diffusion PDE solver | Coupled to signal field |
| Signal-glia interaction | Per-voxel coupling terms | Bidirectional |
| Agent simulation in continuous space | Custom | Positions, forces, decisions |
| Multi-timescale coordination | Custom scheduler | Different update rates |

**No existing library does this as an integrated system.** The closest analogs are:
- Computational neuroscience simulators (NEURON, Brian2) — handle biophysics but not learning at scale
- Physics engines (CUDA-based CFD solvers) — handle spatial PDEs but not neural computation
- Neuromorphic hardware (Loihi 2) — handles delays and local learning but not continuous spatial fields

A Level 3 implementation would likely combine:
- CUDA kernels for parallel field updates (borrowed from CFD/PDE solver patterns)
- Event-driven spike scheduling (borrowed from SNN simulators)
- Custom multi-clock scheduler (novel)
- Agent logic on CPU with GPU-accelerated spatial queries

---

## Progression Through the Research Plan

### Current Research Steps Mapped to Fidelity Levels

| Step | Current Level | What It Tests | What It Misses |
|------|--------------|---------------|----------------|
| 01 (embedding) | Level 1 | Spatial coordinate assignment | — (foundational, level-independent) |
| 02 (modulation field) | Level 1 | PDE-coupled learning rate | Field doesn't affect signals, only learning |
| 03 (astrocyte domains) | Level 1 | Calcium dynamics for LR modulation | Astrocytes don't interact with signals in transit |
| 04 (Turing stability) | Level 1 | Parameter safety analysis | — (analytical, level-independent) |
| 05 (microglia agents) | Level 1 | Spatially-informed pruning | Pruning is topology-only, no timing effects |
| 06 (error chemotaxis) | Level 1 | Agent clustering at error regions | Error is computed instantaneously |
| 07 (volume transmission) | Level 1 | Broadcast modulation | Broadcast affects learning, not signal propagation |
| 08 (multi-timescale) | Level 1 | Timescale separation | Only learning timescales, not signal timescales |
| 09 (continual learning) | Level 1 | Structural memory protection | — |
| 10 (full ecosystem) | Level 1 | Integration of all mechanisms | All mechanisms affect learning only |
| 11 (scaling/cost) | Level 1 | Computational overhead | Measures Level 1 cost only |
| 12 (local rules) | Level 1-2 | STDP and local learning | STDP needs timing (Level 2) to be meaningful |
| 13 (astrocyte gate) | Level 1-2 | D-serine gating | Gating is more meaningful with temporal dynamics |
| 14 (volume teaching) | Level 1-2 | Error broadcast via diffusion | Diffusion speed matters more at Level 2+ |
| 15 (heterosynaptic) | Level 1 | Lateral competition | — |
| 16 (comparative) | Level 1-2 | Benefit comparison | Comparison is level-dependent |

### New Steps for Level 2 and Level 3 Transition

The following steps should be added to the research plan to progress toward full temporal-spatial simulation:

| Step | Level | Focus | Key Question |
|------|-------|-------|--------------|
| 17 | 2 | Temporal simulation engine with delays | Does adding propagation delays change which mechanisms help? |
| 18 | 2 | Myelination-delay coupling | Does adaptive delay (oligodendrocyte control) improve temporal computation? |
| 19 | 2-3 | In-transit signal-glia interaction | Does modulating signals during propagation add computational value? |
| 20 | 3 | Full spatial-temporal simulation | Does the complete system exhibit qualitatively new behaviors? |

---

## Implications for Each Existing Research Step

### What Each Step Should Explicitly State

Every research step document should declare:

```
SIMULATION FIDELITY: Level [1/2/3]
SIGNAL MODEL: [Instantaneous / Delayed / Spatially-propagated]
NETWORK STATE DURING INFERENCE: [Static / Evolving]
GLIAL INTERACTION WITH SIGNALS: [Learning-only / At endpoints / In-transit]
```

### What Changes at Each Level Transition

**Level 1 → Level 2 transition** (after Step 16, before Step 17):
- Forward pass becomes a temporal simulation loop
- Each connection gets a delay parameter
- STDP becomes meaningful (timing matters)
- Myelination now affects inference, not just learning
- Computational cost increases ~10-100x
- Need to re-validate all Phase 1 results: do they still hold with temporal dynamics?

**Level 2 → Level 3 transition** (after Step 18, before Step 19):
- Signals propagate through continuous space (not just delayed teleportation)
- Glial fields interact with signals in transit
- Bidirectional coupling: signals affect glia, glia affect signals
- Emergent spatial phenomena become possible (interference, resonance)
- Computational cost increases another ~10x
- Need to determine: does Level 3 produce qualitatively different results from Level 2?

---

## The Critical Question at Each Transition

**Level 1 → Level 2**: "Do the mechanisms that helped under instantaneous signals STILL help when signals have delays? Or do delays change which mechanisms matter?"

Possible outcomes:
- All Level 1 results transfer to Level 2 (delays are orthogonal to glial benefits)
- Some results transfer, others don't (delays interact with specific mechanisms)
- Level 2 reveals NEW benefits invisible at Level 1 (timing-dependent phenomena)

**Level 2 → Level 3**: "Does in-transit signal-glia interaction produce qualitatively new computation? Or is delayed-teleportation (Level 2) sufficient?"

Possible outcomes:
- Level 3 produces same results as Level 2 (in-transit interaction is negligible)
- Level 3 produces quantitatively better results (in-transit interaction helps but isn't essential)
- Level 3 produces qualitatively NEW behaviors (spatial continuity enables phenomena impossible at Level 2)

If Level 3 produces the same results as Level 2, we save enormous computational cost by staying at Level 2. If Level 3 is qualitatively different, the full simulation is necessary and the cost is justified.

---

## Recommended Progression Strategy

```
Phase 1 (Level 1): Validate spatial geometry helps learning
  Steps 01-11
  Cost: Low (standard PyTorch)
  Duration: 40-50 weeks
  
  Key deliverable: "Spatial geometry improves learning dynamics"
  Go/no-go: If no benefit at Level 1, reconsider entire approach

Phase 2 (Level 1-2): Validate local learning rules + glial gating  
  Steps 12-16
  Cost: Low-Medium (PyTorch + simple temporal simulation for STDP)
  Duration: 25-35 weeks
  
  Key deliverable: "Glia are more beneficial under local rules"
  Go/no-go: If no additional benefit from local rules, stay at Level 1

Phase 3 (Level 2): Full temporal simulation with delays
  Steps 17-18
  Cost: Medium-High (SNN simulator + custom glial integration)
  Duration: 20-30 weeks
  
  Key deliverable: "Propagation delays interact meaningfully with glial mechanisms"
  Go/no-go: If delays don't change results, Level 2 is unnecessary overhead

Phase 4 (Level 3): Full spatial-temporal simulation
  Steps 19-20
  Cost: High (custom CUDA engine)
  Duration: 30-40 weeks
  
  Key deliverable: "In-transit signal-glia interaction produces novel computation"
  Go/no-go: If same results as Level 2, stay at Level 2 for efficiency
```

Each phase validates whether the additional fidelity is worth the computational cost before committing to the next level. This prevents building an expensive Level 3 simulator only to discover that Level 1 captures all the important effects.

---

## Hardware Implications

| Level | Hardware Requirement | Estimated Cost |
|-------|---------------------|---------------|
| Level 1 | Single GPU (RTX 4090 or A100) | Existing hardware |
| Level 2 | Multi-GPU or GPU cluster | 2-8 GPUs |
| Level 3 | GPU cluster or neuromorphic hardware | 8+ GPUs or Loihi 2 |

The progression from Level 1 to Level 3 is also a progression in hardware requirements. Level 1 experiments can run on a single consumer GPU. Level 3 may require dedicated compute infrastructure.

---

## Summary

The research plan progresses from a simplified model (Level 1: spatial geometry affects learning only) through an intermediate model (Level 2: temporal dynamics with delays) to a full simulation (Level 3: continuous spatial-temporal propagation with in-transit glial interaction). Each level adds fidelity and cost. Each transition is gated by a go/no-go decision: does the additional fidelity produce meaningfully different results?

This staged approach ensures we don't build an expensive simulation engine for phenomena that can be captured by a simpler model, while also ensuring we don't miss qualitatively important effects by staying at too low a fidelity level.
