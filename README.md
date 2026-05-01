# Glial-Neural Networks

## A New Computational Paradigm: Neural Networks with Spatial Geometry and Glial Control Systems

This repository documents the theoretical foundations, critical analysis, and experimental research plan for a fundamentally new approach to artificial neural networks — one that introduces spatial geometry, mobile computational agents, chemical signaling fields, and multi-timescale structural remodeling by emulating the glial cell systems that govern biological neural tissue.

---

## Origin of the Idea

Standard artificial neural networks are mathematical abstractions. Their "connections" are entries in a weight matrix — dimensionless scalars with no physical extent, no spatial relationship to one another, and no concept of proximity. Every weight is equidistant from every other weight in the only sense that matters computationally: they are all equally and independently accessible to the optimizer.

Biological neural networks are nothing like this. They exist in three-dimensional physical space. Signals take time to propagate. Connections pass through tissue populated by glial cells — astrocytes, microglia, oligodendrocytes — that actively modulate, prune, time, and restructure the neural network in real time. These glial cells outnumber neurons in many brain regions and perform computations on fundamentally different timescales, spatial scales, and chemical modalities than neuronal signaling.

This project asks: **what happens when you give an artificial neural network a body?** When weights have positions, when signals take time to travel, when mobile agents patrol the network pruning weak connections, when chemical fields diffuse through the spatial embedding modulating everything they touch, when the network's topology is not a fixed hyperparameter but a living, evolving structure shaped by glial systems operating on timescales from seconds to weeks?

The answer, we hypothesize, is not merely "a better neural network." It is a qualitatively different computational system — one whose learning dynamics are governed by coupled partial differential equations rather than simple gradient descent, one that self-organizes, self-repairs, and self-optimizes through mechanisms that have no analog in current AI architectures.

---

## How This Differs from Existing Neural Network Systems

### 1. Spatial Geometry Is Computationally Active

In standard ANNs, architecture diagrams showing spatial layout are visualization artifacts — they have no computational meaning. In our framework, every weight has a physical position in 3D space, and that position determines:
- Which astrocyte domain governs it (spatially correlated learning)
- Which microglia agents can observe and prune it (spatial patrol)
- What volume-transmitted chemical signals reach it (distance-dependent diffusion)
- How long signals take to traverse it (myelination-dependent delay)
- What other weights are its "neighbors" (shared chemical environment)

### 2. Signals Are Not Instantaneous

Standard ANNs operate in "null-space" — a forward pass is an instantaneous mathematical function evaluation. Our framework introduces propagation delays: signals take time to travel from neuron A to neuron B, and that time is controlled by oligodendrocyte myelination. This means:
- Different inputs arrive at a target neuron at different times
- Relative timing determines computation (spike-timing-dependent plasticity)
- The network can perform temporal computations impossible in instantaneous systems
- Myelination is a learnable parameter that changes what the network computes

### 3. The Network Has Mobile Agents

Standard ANNs have no concept of "maintenance." Our framework includes microglia — mobile computational agents that physically patrol the network's spatial embedding, accumulate evidence about connection health, migrate toward problem regions via chemotaxis, and execute pruning decisions based on local spatial context. This is continuous, spatially-directed neural architecture search at runtime.

### 4. Chemical Fields Modulate Everything

Standard ANNs communicate only along edges (connections). Our framework adds volume transmission — chemical signals released into the spatial embedding that diffuse outward, affecting all weights within range regardless of network topology. This provides O(1) broadcast communication, distance-dependent modulation, and a form of spatially distributed working memory stored in chemical concentrations rather than weight values.

### 5. Learning Is Not Backpropagation

In biology, there is no backward pass. Synaptic strength changes through local rules gated by glial signals. Astrocytes release D-serine that is literally required for NMDA-dependent learning — without it, Hebbian plasticity doesn't occur. Our framework explores learning rules where the astrocyte IS the third factor in three-factor plasticity, where volume-transmitted signals replace backpropagated error, and where the glial system is constitutive to learning rather than merely modulatory.

### 6. The Network Topology Evolves

Standard ANNs have fixed architecture chosen before training. Our framework treats topology as a dynamic, evolving object: microglia prune connections, regrowth signals create new ones, myelination stabilizes important pathways, and astrocyte domain boundaries create functional compartments. The topology at any moment encodes the network's developmental history.

### 7. Multi-Timescale Dynamics

Standard ANNs have one timescale: the training step. Our framework operates on five simultaneous timescales:
- Milliseconds: neural spike propagation
- Seconds: astrocyte calcium dynamics, D-serine gating
- Minutes-hours: microglia migration, volume transmission diffusion
- Hours-days: pruning decisions, structural remodeling
- Days-weeks: myelination changes, developmental maturation

---

## Repository Structure

### `glia-neural-network-research/`

Theoretical foundations — 13 documents covering the biology, computational theory, architectural proposals, and long-term vision.

| Document | Topic |
|----------|-------|
| `00-overview.md` | Index and core thesis |
| `01-biological-foundations.md` | Glial cell types, mobility, chemical signaling, timescales |
| `02-astrocyte-modulation.md` | Tripartite synapse, gain control, meta-learning, attention |
| `03-microglia-pruning.md` | Mobile pruning agents, complement tagging, structural remodeling |
| `04-oligodendrocyte-timing.md` | Adaptive delay lines, synchronization, bandwidth control |
| `05-glia-intercommunication.md` | Calcium waves, gap junctions, the glial syncytium |
| `06-interface-mechanisms.md` | How emulated glia interface with existing ANN architectures |
| `07-architectural-proposals.md` | Five concrete architectures from simple overlay to neuromorphic chip |
| `08-emergent-behaviors.md` | Self-organization, homeostasis, pathologies, feature binding |
| `09-existing-research.md` | Current state of academic and industry research |
| `10-open-questions.md` | Challenges, boundary conditions, future directions |
| `11-spatial-geometries.md` | Why glia force physical dimensionality onto ANNs (the paradigm shift article) |
| `12-modular-brain-structures.md` | Long-term vision: composable brain-like structure library |

### `critical-reviews/`

Two extensive critical reviews of the spatial geometry framework, identifying:
- What the framework gets genuinely right (PDE-ODE coupling, modulation field formalism)
- Where it overstates (binding problem, self-designing architectures)
- The most important open problem (spatial coordinate assignment)
- The key risk (Turing instabilities in the modulation field)
- The deeper insight (this is really about non-flat geometry over parameter space)

### `research/`

Experimental research plan — 22 documents defining a staged progression from simple experiments to a full spatial-temporal simulation engine.

**Overview and requirements:**
| Document | Topic |
|----------|-------|
| `00-research-plan-overview.md` | Master plan with phases, dependencies, parallelization analysis |
| `00a-computational-model-requirements.md` | Three simulation fidelity levels (null-space → temporal → full spatial) |

**Phase 1 — Spatial geometry under backpropagation (Level 1: Null-Space):**
| Step | Focus |
|------|-------|
| 01 | Spatial coordinate assignment for weights |
| 02 | Reaction-diffusion modulation field (PDE-coupled learning rate) |
| 03 | Astrocyte units with calcium dynamics and domain coupling |
| 04 | Turing instability regime characterization |
| 05 | Mobile microglia pruning agents with spatial patrol |
| 06 | Microglia chemotaxis and error-region clustering |
| 07 | Volume transmission (broadcast modulation via diffusion) |
| 08 | Multi-timescale training (fast neural, slow glial clocks) |
| 09 | Topology-as-memory for continual learning |
| 10 | Full glial ecosystem integration |
| 11 | Computational cost analysis and scaling |

**Phase 2 — Biologically plausible local learning rules (Level 1-2: Transitional):**
| Step | Focus |
|------|-------|
| 12 | Implement STDP, three-factor, forward-forward, predictive coding |
| 13 | Astrocyte D-serine gating as the third factor in learning |
| 14 | Volume transmission as teaching/error broadcast signal |
| 15 | Astrocyte-mediated heterosynaptic plasticity |
| 16 | Comparative benefit: are glia more beneficial under local rules? |

**Phase 3 — Temporal simulation with propagation delays (Level 2: Temporal):**
| Step | Focus |
|------|-------|
| 17 | Temporal simulation engine with delay queues |
| 18 | Myelination-delay coupling and temporal synchronization |

**Phase 4 — Full spatial-temporal simulation (Level 3: Full Spatial-Temporal):**
| Step | Focus |
|------|-------|
| 19 | In-transit signal-glia interaction (go/no-go for Level 3) |
| 20 | Optimized CUDA engine for full spatial-temporal simulation |

---

## The Central Prediction

**Glial mechanisms will provide GREATER computational benefit when the underlying learning rule is local (Hebbian/STDP/three-factor) than when it is global (backpropagation).**

If this holds, it validates the biological argument: glia evolved specifically to make local learning work at scale. The spatial geometry framework is not just an optimization of backpropagation — it is a principled alternative architecture where glia are constitutive to learning rather than merely modulatory.

---

## Long-Term Vision

The ultimate goal is a **library of composable brain-like structures** — pre-configured glia-neural modules (visual cortex, hippocampus, cerebellum, amygdala, prefrontal cortex, basal ganglia, etc.) that can be assembled into cognitive architectures. Each module has distinct internal architecture, glial configuration, interface specification, and behavioral preset. See `glia-neural-network-research/12-modular-brain-structures.md` for the full vision.

---

## Status

This repository is currently in the **theoretical and planning phase**. No implementation code exists yet. The documents define what should be built, why, in what order, and what success looks like at each step.

---

## Key References

- Kozachkov et al., "Building Transformers from Neurons and Astrocytes," PNAS, 2023
- IBM Research, "Large Memory Storage Through Astrocyte Computation," 2025
- "Neuromorphic Circuits with Spiking Astrocytes," arXiv:2502.20492, 2025
- "Astrocyte-Gated Multi-Timescale Plasticity for Online Continual Learning," Frontiers in Neuroscience, 2025
- "MA-Net: Rethinking Neural Unit in the Light of Astrocytes," AAAI, 2024
- "Activity-Dependent Myelination: Oscillatory Self-Organization," PNAS, 2020
- "Oligodendrocyte-Mediated Myelin Plasticity and Neural Synchronization," eLife, 2023
