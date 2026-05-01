# Experimental Research Plan: Glia-Augmented Neural Networks

## Purpose

This research plan lays out a sequence of implementable experiments designed to test the core claims of the spatial-geometric glia-augmented neural network framework. Each step produces a concrete artifact (code, trained model, measurements) that either supports or refutes a specific claim.

The plan is informed by two critical reviews that identified:
- The strongest theoretical contribution: coupling gradient descent (ODE) to a reaction-diffusion PDE over a spatial embedding creates qualitatively different learning dynamics
- The most important open problem: the spatial coordinate assignment problem (how to embed weights in space meaningfully)
- The key risk: Turing instabilities in the modulation field can self-organize beneficially OR collapse pathologically
- The boundary condition: spatial locality bias helps only when the task's computational structure respects locality

---

## The Learning Rule Question

A fundamental design decision underlies every experiment in this plan: **how do the weights actually update?**

### How Biological Synaptic Learning Works

In biology, there is no backward pass. Synaptic strength changes through **local rules** — the synapse only has access to information available at its own physical location:

**Spike-Timing-Dependent Plasticity (STDP)**: If the pre-synaptic neuron fires and the post-synaptic neuron fires shortly after, the connection strengthens. If the timing is reversed, it weakens. This is a purely local, Hebbian rule.

**Three-factor learning rules**: The modern biological consensus is that STDP alone is insufficient. The actual rule is:

```
Delta_w = eligibility_trace(pre, post) x third_factor_signal

Where:
  Factor 1 (pre-synaptic activity):  Was the pre-synaptic neuron active?
  Factor 2 (post-synaptic activity): Was the post-synaptic neuron active?
  Factor 3 (gating signal):          Is there a "go ahead" from a third source?
```

The first two factors set an **eligibility trace** — a flag saying "this synapse is a candidate for change." The third factor determines whether the change actually happens. This third factor can be a neuromodulatory signal (dopamine, norepinephrine) or — critically — a **glial signal**.

### The Role of Each Glial Type in Learning

**Astrocytes — The Gatekeepers of Plasticity**

Astrocytes are not merely modulators of a learning process that works without them. They are *constitutive components* of the learning rule itself:

- **D-serine release**: NMDA receptors (the molecular coincidence detectors that implement Hebbian learning) require both glutamate AND a co-agonist (D-serine or glycine) to open. Astrocytes are the primary source of D-serine at many synapses. Without astrocytic D-serine, NMDA receptors don't fully activate, and LTP (long-term potentiation) fails. The astrocyte literally gates whether Hebbian learning can occur at a given synapse.

- **Astrocyte-mediated STDP**: Research has shown that astrocytes mediate spike-timing-dependent LTD (long-term depression) in developing cortex. The astrocyte detects correlated pre/post activity, releases D-serine, and this enables the timing-dependent plasticity window. ([PMC7654831](https://pmc.ncbi.nlm.nih.gov/articles/PMC7654831/))

- **Astrocyte-gated multi-timescale plasticity (AGMP)**: A 2025 paper implements exactly this — an astrocyte-mediated gating mechanism that augments eligibility traces with a broadcast teaching signal. This is a working implementation of three-factor learning with astrocytes as the third factor. ([Frontiers in Neuroscience, 2025](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1768235/full))

- **Heterosynaptic plasticity**: When one synapse is potentiated, nearby synapses (within the same astrocyte domain) can be depressed. The astrocyte mediates this cross-synapse interaction. This is a lateral interaction governed by spatial proximity that backpropagation cannot capture.

- **Learning-associated astrocyte ensembles**: A 2024 Nature paper showed that ensembles of astrocytes are activated by learning events and control memory recall — astrocytes form their own "engrams" parallel to neuronal engrams. ([PubMed 39506118](https://pubmed.ncbi.nlm.nih.gov/39506118/))

**Microglia — Structural Learning (Not Weight Updates)**

Microglia don't change synaptic weights. They change whether synapses *exist*:

- They eliminate weak/unused synapses (pruning)
- They facilitate new synapse formation (by clearing debris, releasing trophic factors)
- They selectively prune inhibitory vs. excitatory synapses to shift the E/I balance
- Selective activation of microglia facilitates synaptic strength ([PMC4363384](https://pmc.ncbi.nlm.nih.gov/articles/PMC4363384/))
- This is **structural plasticity** — changing the network topology, not the connection strengths

**Oligodendrocytes — Timing-Based Learning**

Oligodendrocytes don't change weights either. They change signal *timing*:

- Activity-dependent myelination adjusts conduction velocity
- This changes when signals arrive at downstream neurons
- Since STDP depends on precise timing, changing arrival times changes *which* synapses get potentiated or depressed
- A 2014 Science paper showed that new oligodendrocyte generation is *required* for motor skill learning — blocking it prevents learning even though synapses are intact
- Myelin plasticity plays a crucial role in learning and memory at a level beyond synaptic plasticity alone ([PMC8018611](https://pmc.ncbi.nlm.nih.gov/articles/PMC8018611/))

### The Biological Learning Stack

```
Layer 1 (fastest, ms):       STDP at individual synapses
                              Local Hebbian rule: pre before post = strengthen
                              Gated by: astrocyte D-serine (Factor 3)

Layer 2 (fast, seconds):     Eligibility traces
                              Synapse "remembers" recent co-activation
                              Actual weight change waits for gating signal

Layer 3 (medium, sec-min):   Astrocyte modulation
                              Calcium dynamics integrate activity
                              Release D-serine to enable/disable plasticity
                              Heterosynaptic effects across domain

Layer 4 (slow, hours-days):  Microglial remodeling
                              Remove unused synapses, enable new ones
                              Structural topology changes

Layer 5 (slowest, days-wks): Myelination
                              Optimize timing of established pathways
                              Required for motor skill consolidation
```

### Three Approaches for This Research

Given this biological reality, we have three options for how to implement learning in our experiments:

---

**Approach A: Backpropagation + Glial Modulation**

```
Weight update: Delta_w = -glial_modulation(position) * learning_rate * dL/dw
```

- Neural learning: standard backpropagation computes gradients
- Glial role: modulate learning rates, prune connections, gate plasticity regions
- The glial system operates as a meta-controller over a backprop-trained network

| Pros | Cons |
|------|------|
| Easy to implement and benchmark | Not biologically faithful |
| Isolates glial spatial contribution | Backprop's global error signal may mask glial effects |
| Comparable to existing SOTA methods | Glia are reduced to "fancy learning rate scheduler" |
| Clear baselines (Adam, SAM, etc.) | Misses glia's role as constitutive part of learning |

---

**Approach B: Local Hebbian/STDP + Glial Gating (Biologically Faithful)**

```
Weight update: Delta_w = STDP(pre, post) * astrocyte_gate(D_serine) * global_broadcast
```

- Neural learning: STDP or three-factor local rules (only local information used)
- Glial role: provide the third factor (D-serine gating), heterosynaptic modulation, structural pruning
- Astrocytes are the *reason* learning happens, not just a modulator of it

| Pros | Cons |
|------|------|
| Biologically coherent | Local rules alone perform poorly on standard benchmarks |
| Glia play their natural role as plasticity gatekeepers | Harder to compare to SOTA |
| Tests the strongest biological claim | Requires spiking or rate-coded STDP implementation |
| May reveal benefits invisible under backprop | Slower convergence expected |

---

**Approach C: Hybrid — Local Rules + Glial Broadcast as Teaching Signal**

```
Weight update: Delta_w = eligibility(pre, post) * volume_transmitted_error_signal(position)
```

- Neural learning: local Hebbian rule sets eligibility traces
- Glial role: the global teaching/error signal IS the glial broadcast (volume transmission provides the "third factor" that makes local rules work at scale)
- This bridges biology and performance: glia provide the missing piece that makes local rules competitive

| Pros | Cons |
|------|------|
| Bridges biology and performance | More complex implementation |
| Glia provide the "missing piece" for local learning | Less well-understood theoretically |
| Testable prediction: glia MORE beneficial here than under backprop | Novel territory — fewer existing baselines |
| Maps onto predictive coding / forward-forward frameworks | May require careful tuning |
| Volume transmission naturally implements broadcast error | |

---

### The Key Insight

In biology, **glia are not just modulating a learning rule that works fine without them**. D-serine from astrocytes is *required* for NMDA-dependent LTP at many synapses. Without it, Hebbian learning simply doesn't happen. This is fundamentally different from "backprop but with a variable learning rate."

The most faithful emulation would be:

```
Delta_w_ij = eligibility(pre_i, post_j) * astrocyte_gate(domain_k) * broadcast_signal

Where:
  eligibility    = STDP-like function of pre/post timing (local, fast)
  astrocyte_gate = calcium-dependent D-serine analog (domain-local, medium timescale)
  broadcast      = volume-transmitted teaching signal (regional, slow timescale)
```

This is a **three-factor rule where the astrocyte IS the third factor** — not a modulator of backprop, but a constitutive part of the learning rule.

---

## Research Phases

The research is organized into two phases. Phase 1 uses backpropagation (Approach A) to isolate and validate the spatial/structural contributions of glia in a well-understood setting. Phase 2 replaces backprop with biologically plausible local rules (Approaches B and C) to test whether glia are *more* beneficial — as biology predicts — when the learning rule is local.

### Phase 1: Spatial Geometry and Glial Mechanisms Under Backpropagation

**Goal**: Establish that spatial embedding, glial modulation, mobile pruning, and multi-timescale dynamics provide measurable benefit even when the underlying learning rule is standard backpropagation.

**Rationale**: This isolates the spatial/structural contributions. If glia help even under backprop (which doesn't need them), the spatial geometry argument stands on its own merits.

| Step | File | Focus | Key Question |
|------|------|-------|--------------|
| 01 | [01-spatial-embedding.md](./01-spatial-embedding.md) | Assign spatial coordinates to weights | Does the embedding method matter? |
| 02 | [02-modulation-field.md](./02-modulation-field.md) | Implement the reaction-diffusion modulation field | Does a PDE-coupled learning rate outperform Adam? |
| 03 | [03-astrocyte-domains.md](./03-astrocyte-domains.md) | Astrocyte units with calcium dynamics and domain coupling | Does spatial correlation of learning rates help? |
| 04 | [04-turing-stability.md](./04-turing-stability.md) | Characterize Turing instability regimes | Where are the safe vs. pathological parameter regions? |
| 05 | [05-microglia-agents.md](./05-microglia-agents.md) | Mobile pruning agents with spatial patrol | Does spatially-informed pruning beat magnitude pruning? |
| 06 | [06-error-chemotaxis.md](./06-error-chemotaxis.md) | Microglia clustering at high-error regions | Does agent migration reduce chaotic learning? |
| 07 | [07-volume-transmission.md](./07-volume-transmission.md) | Broadcast modulation via diffusion fields | Does topology-independent communication add value? |
| 08 | [08-multi-timescale.md](./08-multi-timescale.md) | Multi-clock training (fast neural, slow glial) | Does timescale separation improve convergence? |
| 09 | [09-continual-learning.md](./09-continual-learning.md) | Topology-as-memory for catastrophic forgetting | Does structural protection beat weight regularization? |
| 10 | [10-full-ecosystem.md](./10-full-ecosystem.md) | Combined system: astrocytes + microglia + timing | Does the full system exceed the sum of its parts? |
| 11 | [11-scaling-and-cost.md](./11-scaling-and-cost.md) | Computational cost analysis and scaling behavior | Is the overhead justified by improved efficiency? |

**Estimated timeline**: 40-50 weeks (steps 02-08 can be partially parallelized)

---

### Phase 2: Biologically Plausible Learning Rules with Glial Gating

**Goal**: Replace backpropagation with local learning rules and test whether glial mechanisms become *more* beneficial — validating the biological prediction that glia are constitutive to learning, not merely modulatory.

**Rationale**: If glia provide greater benefit under local rules than under backprop, it confirms that the biological architecture is not arbitrary — glia evolved to solve the specific problem of making local learning work at scale.

| Step | File | Focus | Key Question |
|------|------|-------|--------------|
| 12 | [12-local-learning-rules.md](./12-local-learning-rules.md) | Implement STDP, three-factor, and forward-forward rules | Can local rules work at all on our benchmarks? |
| 13 | [13-astrocyte-as-third-factor.md](./13-astrocyte-as-third-factor.md) | Astrocyte D-serine gating as the third factor in three-factor learning | Does astrocyte gating make local rules competitive? |
| 14 | [14-volume-broadcast-teaching.md](./14-volume-broadcast-teaching.md) | Volume transmission as the teaching/error broadcast signal | Can diffusion-based error signals replace backprop? |
| 15 | [15-heterosynaptic-plasticity.md](./15-heterosynaptic-plasticity.md) | Astrocyte-mediated lateral interactions between synapses | Does heterosynaptic modulation improve representations? |
| 16 | [16-comparative-benefit.md](./16-comparative-benefit.md) | Compare glial benefit under backprop vs. local rules | Are glia MORE beneficial under local learning? |

**Estimated timeline**: 25-35 weeks (after Phase 1 infrastructure is established)

---

### Phase 3: Temporal Simulation with Propagation Delays (Level 2)

**Goal**: Transition from instantaneous signals to a temporal simulation where signals have propagation delays controlled by myelination. Test whether temporal dynamics interact with glial mechanisms and reveal phenomena invisible at Level 1.

**Rationale**: Standard ANNs operate in "null-space" — signals teleport instantly. Biology operates in time — signals take measurable time to propagate, and that time is controlled by oligodendrocytes. This phase tests whether adding temporal dynamics changes which mechanisms matter.

| Step | File | Focus | Key Question |
|------|------|-------|--------------|
| 17 | [17-temporal-simulation-engine.md](./17-temporal-simulation-engine.md) | Build temporal simulation with delay queues | Do Phase 1 results still hold with propagation delays? |
| 18 | [18-myelination-delay-coupling.md](./18-myelination-delay-coupling.md) | Adaptive myelination controls signal timing | Does learnable delay improve temporal computation? |

**Estimated timeline**: 14-18 weeks

**Go/no-go for Phase 4**: Do delays interact meaningfully with glial mechanisms? Does Level 2 reveal new phenomena?

---

### Phase 4: Full Spatial-Temporal Simulation (Level 3)

**Goal**: Implement continuous spatial signal propagation where signals interact with glial fields in transit. Determine whether this full-fidelity simulation produces qualitatively different (and better) results than Level 2.

**Rationale**: In biology, signals don't just teleport after a delay — they physically traverse space, passing through multiple glial domains. Each domain can modulate the signal during transit. This phase tests whether that in-transit interaction matters computationally.

| Step | File | Focus | Key Question |
|------|------|-------|--------------|
| 19 | [19-in-transit-signal-glia-interaction.md](./19-in-transit-signal-glia-interaction.md) | Signals interact with glial fields during propagation | Does in-transit modulation add computational value beyond delays? |
| 20 | [20-full-spatial-temporal-simulation.md](./20-full-spatial-temporal-simulation.md) | Optimized CUDA engine for full spatial-temporal simulation | Can this be made practical? Which applications need it? |

**Estimated timeline**: 20-28 weeks

**Go/no-go**: Step 19 is the critical gate. If Level 3 produces the same results as Level 2, stay at Level 2 and save the compute. If Level 3 is qualitatively different, proceed to Step 20 optimization.

---

### Phase Transition Criteria

**Phase 1 → Phase 2** (move when):
1. Spatial embedding problem is solved (Step 01 produces a reliable embedding method)
2. At least 3 of Steps 02-08 show positive results (glial mechanisms provide measurable benefit under backprop)
3. Full ecosystem (Step 10) demonstrates synergistic effects
4. Computational overhead is characterized (Step 11) and manageable

**Phase 2 → Phase 3** (move when):
1. Local learning rules are implemented and baselined (Step 12)
2. Astrocyte gating provides measurable benefit under local rules (Step 13)
3. The comparative benefit experiment (Step 16) is complete
4. STDP implementation requires temporal dynamics to be meaningful

**Phase 3 → Phase 4** (move when):
1. Temporal simulation engine is working (Step 17)
2. Propagation delays interact meaningfully with glial mechanisms (not orthogonal)
3. Adaptive myelination improves temporal computation (Step 18)
4. There is reason to believe in-transit interaction adds value (theoretical or empirical)

**Phase 4 go/no-go** (at Step 19):
- If Level 3 produces qualitatively different results from Level 2 → proceed to Step 20
- If Level 3 produces same results as Level 2 → STOP, stay at Level 2, document why

If Phase 1 shows no benefit, Phase 2 is still worth pursuing — it's possible that glial mechanisms are specifically adapted to local learning and provide little benefit when backprop already solves the credit assignment problem globally.

---

## Guiding Principles

1. **Every step produces runnable code and quantitative measurements**
2. **Each experiment isolates one claim** — no experiment tests everything at once
3. **Baselines are mandatory** — every glia-augmented result is compared to the best non-glial equivalent
4. **Failure is informative** — negative results constrain the framework as usefully as positive ones
5. **Start small, scale deliberately** — prove mechanisms on toy problems before scaling
6. **The learning rule matters** — results under backprop may not transfer to local rules, and vice versa

## Dependencies

### Phase 1

```
01 (embedding) ──────────────────────────────────────────────────────┐
    |                                                                 |
    +---> 02 (modulation field) ---> 04 (Turing stability)           |
    |         |                                                       |
    |         +---> 03 (astrocyte domains) ---+                       |
    |                                         |                       |
    +---> 05 (microglia agents) --------------+                       |
    |         |                               |                       |
    |         +---> 06 (error chemotaxis) ----+                       |
    |                                         |                       |
    +---> 07 (volume transmission) -----------+                       |
    |                                         |                       |
    +---> 08 (multi-timescale) ---------------+                       |
                                              |                       |
                                              v                       |
                                     09 (continual learning)          |
                                              |                       |
                                              v                       |
                                     10 (full ecosystem) <------------+
                                              |
                                              v
                                     11 (scaling and cost)
```

### Phase 2

```
Phase 1 results (especially 03, 07, 10)
    |
    +---> 12 (local learning rules) ----+
    |                                    |
    |                                    v
    +---> 13 (astrocyte as third factor)
    |         |
    |         v
    +---> 14 (volume broadcast teaching)
    |         |
    |         v
    +---> 15 (heterosynaptic plasticity)
              |
              v
         16 (comparative benefit)
```

### Phase 3

```
Phase 2 results (especially 12, 16) + Phase 1 infrastructure
    |
    +---> 17 (temporal simulation engine)
    |         |
    |         v
    +---> 18 (myelination-delay coupling)
              |
              v
         Go/no-go decision for Phase 4
```

### Phase 4

```
Phase 3 results (17, 18) + all previous infrastructure
    |
    +---> 19 (in-transit signal-glia interaction)
              |
              v
         Go/no-go: does Level 3 add value?
              |
              v (if yes)
         20 (full optimized spatial-temporal simulation)
```

### Parallelization and Agent Assignment Analysis

The steps are NOT all independent. Here is a detailed breakdown of what can be given to separate AI agents and what requires shared code or sequential results.

#### Fully Independent Steps (can be assigned to separate agents with no shared code)

- **Step 01 (spatial embedding)**: Foundational. Produces the position array that nearly everything else consumes. Must run first or its output must be specified as a contract.
- **Step 04 (Turing stability)**: Pure math/simulation. Only needs the *concept* of a reaction-diffusion PDE, not code from other steps. An agent could implement this from the description alone.
- **Step 12 (local learning rules)**: Implementing STDP, three-factor rules, forward-forward, etc. are self-contained. No dependency on any spatial embedding or glial code.

#### Step 01 as the Universal Dependency

Everything in Phase 1 (except Step 04) depends on Step 01 producing a spatial coordinate assignment for weights. Steps 02-11 all need to know "where is each weight in 3D space?" — that's the foundational data structure.

An agent doing Step 02 needs:
- A trained (or training) neural network
- A spatial position array of shape `(N_weights, 3)` — the output of Step 01

Step 01 must run first, OR every subsequent agent must be given a fixed embedding method (e.g., "use spectral embedding") as a specification rather than waiting for Step 01's experimental results.

#### Code/Results Dependency Table

| Step | Needs code/results from | Shares code with |
|------|------------------------|-----------------|
| 02 (modulation field) | 01 (positions) | 03, 04, 07 (all use the PDE solver) |
| 03 (astrocyte domains) | 01 (positions), 02 (field concept) | 06, 08, 09, 10 |
| 04 (Turing stability) | 02 or 03 (the PDE to analyze) | — |
| 05 (microglia agents) | 01 (positions) | 06 (extends 05 with chemotaxis) |
| 06 (error chemotaxis) | 05 (agent code) | 10 |
| 07 (volume transmission) | 01 (positions) | 14 (extends 07 for error broadcast) |
| 08 (multi-timescale) | 03 (astrocytes), 05 (microglia) | 10 |
| 09 (continual learning) | 03 + 05 (astrocytes + pruning) | 10 |
| 10 (full ecosystem) | 03 + 05 + 07 + 08 (ALL components) | 11 |
| 11 (scaling/cost) | 10 (full system to measure) | — |

#### Phase 2 Dependencies

| Step | Needs from Phase 1 | Needs from Phase 2 |
|------|-------------------|-------------------|
| 12 (local rules) | Nothing | — |
| 13 (astrocyte gate) | 03 (astrocyte code) | 12 (local rule implementations) |
| 14 (volume teaching) | 07 (volume transmission code) | 12 (local rule implementations) |
| 15 (heterosynaptic) | 03 (astrocyte domains) | 12 or 13 (a local rule to apply it to) |
| 16 (comparative) | 10 (full Phase 1 results) | 12-15 (all Phase 2 results) |

#### Practical Parallelization Strategy

Three clusters that could be run across separate agents simultaneously:

**1. Foundation agent** — Step 01 (spatial embedding). Must finish first. Produces the position array that everyone else consumes.

**2. Parallel cluster** (after Step 01 delivers positions):
- Agent A: Steps 02 → 03 → 04 (modulation field → astrocyte domains → Turing analysis)
- Agent B: Steps 05 → 06 (microglia agents → chemotaxis)
- Agent C: Step 07 (volume transmission)
- Agent D: Step 08 (multi-timescale — but needs astrocyte code from Agent A's Step 03)
- Agent E: Step 12 (local learning rules — fully independent, can start immediately)

**3. Integration agents** (need results from the parallel cluster):
- Steps 09, 10, 11 (need multiple components combined)
- Steps 13, 14, 15, 16 (need Phase 1 components + Step 12)

#### Hard Dependencies (cannot be worked around)

- Step 06 literally extends Step 05's agent code with migration logic
- Step 08 needs both astrocyte (03) and microglia (05) code running together
- Step 10 needs everything from 02-08 integrated into one system
- Steps 13-15 each need both a Phase 1 glial component AND Step 12's local rules
- Step 16 needs everything from both phases

#### Recommendation for Agent-Based Execution

If assigning steps to independent AI agents, either:
- **(a)** Define shared interfaces/APIs upfront (e.g., "a spatial embedding is a numpy array of shape (N, 3)", "an astrocyte unit exposes .sense(), .update_calcium(), .output_modulation()") so agents can code against contracts without seeing each other's implementations
- **(b)** Run them sequentially, with each agent receiving the previous agent's output code as input context
- **(c)** Hybrid: run the independent steps (01, 04, 12) in parallel, then feed their outputs into the dependent steps

---

## Technology Stack (Recommended)

- **Framework (Phase 1)**: PyTorch (for dynamic graph support and custom autograd)
- **Spiking networks (Phase 2-3)**: Norse, snnTorch, or custom implementation
- **Temporal simulation (Phase 3)**: Brian2CUDA as reference; custom engine for integration with glial systems
- **Spatial simulation (Phase 4)**: Custom CUDA kernels (signal propagation, PDE field solvers)
- **Spatial computation**: Custom CUDA kernels or PyTorch sparse ops for PDE solving
- **Agent simulation**: Simple Python classes initially; Numba/JAX for scaling
- **Visualization**: Matplotlib for fields, NetworkX for topology, Weights & Biases for tracking
- **Benchmarks**: MNIST/CIFAR-10 for vision; simple sequence tasks for temporal; Split-MNIST/Permuted-MNIST for continual learning; temporal pattern tasks for Phase 3+

## Simulation Fidelity Reference

See [00a-computational-model-requirements.md](./00a-computational-model-requirements.md) for the full specification of the three simulation fidelity levels (Level 1: Null-Space, Level 2: Temporal, Level 3: Full Spatial-Temporal), what each captures, what each sacrifices, and the CUDA library support analysis for each level.

## Key Prediction

The central testable prediction of this research program:

**Glial mechanisms will provide GREATER benefit under local learning rules (Phase 2) than under backpropagation (Phase 1).**

If this prediction holds, it validates the biological argument: glia evolved specifically to make local learning work at scale, and their computational role is constitutive rather than merely modulatory. If it fails (glia help equally or more under backprop), the framework is still valuable but the biological motivation needs revision — the benefit would be purely geometric/structural rather than learning-rule-dependent.

## Long-Term Vision: Modular Brain Structure Library

Beyond the research steps, the ultimate goal is to develop a **library of composable brain-like structures** — pre-configured glia-neural modules with distinct internal architectures tuned for specific computational roles (sensory processing, memory, motor control, executive function, etc.). These modules can be assembled into larger cognitive systems, much like biological brains are assembled from specialized regions.

See [../glia-neural-network-research/12-modular-brain-structures.md](../glia-neural-network-research/12-modular-brain-structures.md) for the full vision document, including:
- Catalog of proposed structures (visual cortex, hippocampus, cerebellum, amygdala, prefrontal cortex, basal ganglia, thalamus, etc.)
- Each structure's neural architecture, glial configuration, interface specification, and behavioral preset
- Composition rules for connecting structures into systems
- Development roadmap tied to research step milestones
- Design principles for the library

This library becomes buildable once the research plan validates the core mechanisms (Phase 1-2) and the simulation engine (Phase 3-4).
