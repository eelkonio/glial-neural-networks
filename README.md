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

This repository is in **active experimental phase**. Steps 01, 01b, 12, 13, and 12b are complete with full results. Step 14 (Predictive Coding + BCM) is in requirements phase.

### Completed Experiments

| Step | Result | Key Finding |
|------|--------|-------------|
| 01/01b | Spatial coupling = weak regularizer under backprop | Spatial structure irrelevant for backprop on FC networks |
| 12 | All local rules at chance (~10%) | Eligibility trace always positive under ReLU — can't weaken synapses |
| 13 | All gate variants at chance (10%) | Multiplicative gates can't provide direction to non-negative eligibility |
| 12b | BCM direction produces signed updates, still 10% | Direction alone is necessary but not sufficient — needs task-relevant information |

### Current Focus: Step 14 — Predictive Coding + BCM

Combining BCM-directed signed updates with inter-layer prediction errors as the missing task-relevant information channel. Operating at the domain level (astrocyte domains as computational units) rather than individual neurons.

---

## Biological Plausibility: How Local Error Signals Work in Cortex

A key question in this research is whether biological networks can carry error information between layers locally — without backpropagation. The answer is yes, but the mechanism is fundamentally different from passing an error tensor backward.

### The Anatomy of Feedback

In cortex, every feedforward projection (area V1 → V2) has a corresponding feedback projection (V2 → V1). These are not the same axons running backward — they are separate populations of neurons with their own dynamics. The feedback connections are:
- **More diffuse** — one feedback neuron projects broadly, not point-to-point
- **Targeting different dendritic compartments** — apical dendrites, not basal
- **Operating on different timescales** — slower, modulatory rather than driving

### Error Is Computed Locally at the Receiving Neuron

A pyramidal neuron in cortical layer N receives:
- **Bottom-up input** on its basal dendrites (from layer N-1, the "actual" signal)
- **Top-down prediction** on its apical dendrites (from layer N+1, the "expected" signal)

The neuron itself computes the mismatch — it doesn't receive a pre-computed "error signal." The error emerges from the difference between what the apical dendrite expects and what the basal dendrite delivers. This is genuinely local: the neuron compares two inputs it already receives.

In our computational model, the prediction error `actual_next - predicted_next` is a simplification of this biological process. The prediction weights P_i are the computational analog of feedback connections. The key biological constraint is satisfied: the error is computed *at* the neuron, not transmitted *to* it.

### The Temporal Constraint

In biology, the feedback signal arrives with a delay (axonal conduction + synaptic transmission ≈ 5-20ms). This means the "prediction" is always slightly stale — it reflects what the higher layer expected based on slightly earlier input. In our rate-based model we currently ignore this delay, but it's worth noting that the biological system works despite this temporal imprecision. This suggests the mechanism is robust to temporal mismatch, which is encouraging for our simplified implementation.

The delay also has a functional consequence: it creates a natural temporal window during which the neuron can compare "what arrived" (bottom-up, fast) with "what was expected" (top-down, delayed). This comparison window may be coordinated by theta oscillations (~4-8 Hz), which create alternating phases of bottom-up processing and top-down prediction. Our model doesn't yet capture this oscillatory structure, but it could be added in future steps.

### Domains as the Unit of Prediction

A single astrocyte domain (~50μm, covering ~100-1000 synapses) shares a chemical environment. The D-serine availability, calcium state, and ATP/adenosine signaling are uniform within a domain. This means the biological "prediction" is not per-synapse — it's per-domain. The astrocyte doesn't predict what each individual synapse should do; it maintains a domain-level expectation of aggregate activity.

This motivates our design choice to operate prediction errors at the domain level rather than the neuron level. Domain-level predictions are:
- More biologically faithful (matches the spatial scale of glial computation)
- Less noisy (averaging over ~16 neurons reduces variance)
- Computationally cheaper (8 domains per layer vs 128 neurons)
- Sufficient for the information we need (which domains are "surprised" vs "satisfied")

---

## Higher-Dimensional Brains: What Happens Beyond 3D?

Our framework embeds neural networks in spatial geometry — currently 2D or 3D, matching biological brains. But since we're building a *computational* framework unconstrained by physical atoms, a natural question arises: **what happens if we embed the network in 4D, 5D, or higher-dimensional space?** Could a higher-dimensional brain be more efficient? Less efficient? Qualitatively different?

This is not merely a theoretical curiosity. The dimensionality of the embedding space fundamentally determines how the glial mechanisms operate — how many neighbors each domain has, how chemical signals diffuse, how sharp the boundaries between domains are, and how much local information is available to each computational unit.

### What Dimensionality Controls

The spatial dimension d determines three critical properties of the glial-neural system:

**1. Neighborhood richness — how many neighbors each domain has.**

In a d-dimensional space, the number of domains within a fixed radius r of any given domain scales as r^d. In 3D, an astrocyte domain might border ~12-15 other domains. In 5D, the same domain (with the same radius of influence) would border ~50-100 other domains. In 8D, potentially hundreds.

This means each domain has access to more "local" information in higher dimensions. For our predictive coding framework, this translates to: each domain can observe more neighboring domains' activities, make richer predictions, and receive more diverse prediction error signals — all without any long-range connections.

**2. Diffusion dynamics — how chemical signals propagate.**

Volume-transmitted signals (D-serine, ATP, calcium waves) diffuse through the spatial embedding. The physics of diffusion changes fundamentally with dimensionality:

- In d=1: Diffusion is slow, signals spread linearly. A point source affects a long thin region.
- In d=2: Concentration from a point source falls off as ~log(1/r). Signals spread broadly but weakly.
- In d=3: Concentration falls off as ~1/r. The biological default — moderate spread, moderate attenuation.
- In d≥3: Concentration falls off as ~1/r^(d-2). Higher dimensions mean *faster attenuation* — signals are more spatially confined.

The implication: in higher dimensions, chemical signals have **sharper boundaries**. A D-serine release in 5D affects a tight hypersphere around the source, with rapid falloff. There's less "bleed" between adjacent domains, less cross-talk, cleaner separation of independent computations. This is potentially very useful for maintaining distinct functional regions.

**3. Surface-to-volume ratio — how domains interact at boundaries.**

In d dimensions, a domain of radius r has volume proportional to r^d but surface area proportional to r^(d-1). More importantly, in high dimensions, the fraction of a domain's neurons that are "near the boundary" (and thus interact with neighboring domains) increases dramatically.

This is the well-known "curse of dimensionality" working *in our favor*: in high-dimensional spaces, almost all the volume of a sphere is concentrated near its surface. For a domain of neurons, this means almost every neuron is close to the domain boundary — almost every neuron has direct access to inter-domain signals. There are no "deep interior" neurons isolated from cross-domain communication.

### The Spectrum from Low to High Dimensionality

**d=2 (Flat sheet)**
- Few neighbors (~6 in a hexagonal packing)
- Slow diffusion, broad signals
- Strong spatial locality — distant domains are truly isolated
- Analogous to a cortical sheet (which is approximately 2D, ~2-4mm thick but spread over ~2,500 cm²)
- Risk: too few neighbors for rich local prediction

**d=3 (Biological default)**
- Moderate neighbors (~12-15)
- Well-understood diffusion (1/r falloff)
- Good balance of locality and connectivity
- The regime where all our biological references apply
- Proven to work for biological brains

**d=4-5 (Enhanced locality)**
- Rich neighborhoods (~30-80 neighbors)
- Sharper signal boundaries (1/r² to 1/r³ falloff)
- Each domain sees a substantial fraction of the network locally
- Prediction errors from many adjacent domains available
- Potentially the computational sweet spot: rich context without losing structure

**d=6-8 (High-dimensional)**
- Very rich neighborhoods (~100-500 neighbors)
- Very sharp signal boundaries (rapid attenuation)
- Almost all neurons are "boundary neurons"
- Local context approaches global context
- Risk: beginning to lose meaningful spatial structure

**d→∞ (Fully connected limit)**
- Every domain is a neighbor of every other domain
- Signals don't attenuate (or attenuate infinitely fast — depending on formulation)
- Spatial structure becomes meaningless
- Equivalent to the fully-connected case where Step 01 showed spatial structure is irrelevant
- This is the regime we want to *avoid*

### The Optimal Dimensionality Hypothesis

There likely exists an optimal dimensionality for our framework — high enough to provide rich local context and sharp chemical boundaries, but low enough to maintain meaningful spatial structure where "nearby" and "distant" are genuinely different.

Our hypothesis: **d=4-8 is the computational sweet spot** for glial-neural networks. The reasoning:

1. **Rich enough for prediction**: With ~50-200 neighboring domains, each domain has access to enough context to make meaningful predictions about its local network state. In 3D with only ~12 neighbors, the prediction context may be too impoverished.

2. **Sharp enough for modularity**: The rapid signal attenuation in d≥4 creates clean boundaries between functional regions. D-serine released in one domain doesn't bleed into distant domains. This supports the formation of independent computational modules.

3. **Structured enough for locality**: Unlike d→∞, there's still a meaningful distinction between "local" and "global." Nearby domains share chemical context; distant domains don't. This is what makes spatial structure computationally useful.

4. **Efficient for wiring**: In higher dimensions, the average shortest path between any two nodes in a lattice decreases. Information can flow from one side of the network to the other through fewer hops. This reduces the need for long-range connections.

### Could a Higher-Dimensional Brain Be More Useful?

**Yes, with caveats.** A brain embedded in 5D would have:

- Each astrocyte domain bordering ~50 other domains (vs ~12 in 3D) → richer local context for prediction
- Sharper chemical boundaries → less cross-talk, cleaner modular computation
- More efficient routing → shorter path lengths between any two neurons
- More boundary neurons → better inter-domain communication
- Faster local consensus → domain-level predictions converge faster with more neighbors

The caveats:

- **Diminishing returns**: Going from 3D to 5D is a big jump in neighborhood size. Going from 5D to 8D is less impactful (you already have rich neighborhoods).
- **Computational cost**: Distance calculations in d dimensions cost O(d) per pair. With d=8, this is ~3× more expensive than d=3. But this is a minor cost compared to the matrix operations in the forward pass.
- **Embedding quality**: Finding a good d-dimensional embedding of the network's weight space becomes harder as d increases (more degrees of freedom, potentially more local optima in the embedding optimization).
- **Interpretability**: 3D embeddings can be visualized. 5D+ cannot. This makes debugging and understanding the spatial structure harder.

### Why Biological Brains Are 3D (And Why We're Not Constrained)

Biological brains are 3D because they're made of atoms in 3D space. But there's a deeper reason: **wiring cost**. The total axon length needed to connect N neurons with random connectivity scales as N^(1+1/d) in d dimensions. In 3D, this is N^(4/3). In 5D, it would be N^(6/5) — more favorable. Biology can't exploit this because it can't build 5D structures. We can.

There's also a metabolic argument: in 3D, the brain already uses ~20% of the body's energy despite being ~2% of its mass. The wiring (white matter) is a major energy cost. A higher-dimensional brain would have shorter wires (less white matter) but the same computational capacity — it would be more energy-efficient.

For our computational framework, we're free to choose any dimensionality. The "wiring cost" analog is the computational cost of maintaining the spatial embedding and computing distances — which is negligible compared to the forward pass. We should choose the dimensionality that maximizes the utility of the glial mechanisms, not the one that minimizes physical wiring.

### Implications for Our Experiments

This analysis suggests a concrete experiment: **test whether increasing the embedding dimensionality from 3D to 5D or 8D improves the effectiveness of domain-level prediction in Step 14.** The prediction is that higher dimensionality will:

1. Provide richer prediction context (more neighboring domains to predict from)
2. Create sharper domain boundaries (less noise in the prediction signal)
3. Improve the speed of prediction convergence (more data points for the small prediction matrices)

This could be tested as a simple ablation within the Step 14 experiment: run the same predictive-BCM rule with the domain assignment computed from a 3D, 5D, and 8D spectral embedding. The network architecture and learning rule stay the same — only the spatial geometry changes.

### Computational Cost of Higher-Dimensional Experiments

A natural concern: are higher-dimensional experiments significantly more expensive to run? The answer is reassuring — **they cost essentially zero additional training time per run.**

The key insight is that the embedding dimensionality only affects the *domain assignment* computation, which happens once at initialization. The training loop — forward pass, BCM direction computation, prediction error calculation, weight updates — operates on *domain activities*, which have the same shape (e.g., 8 domains per layer) regardless of whether those domains were assigned using a 3D or 8D spectral embedding.

| Component | 3D baseline | Cost per additional dimension | Notes |
|-----------|:-----------:|:-----------------------------:|-------|
| Spectral embedding | ~2s | +0.5s per dim | One-time cost. SVD on (128×128) gram matrix, extract d eigenvectors. |
| Domain assignment | ~0.1s | ~0s | Same partitioning algorithm regardless of d. |
| Training loop (per epoch) | ~2s | **~0s** | Domain activities are the same shape regardless of embedding dimension. |
| Full experiment (50 epochs × 3 seeds × 6 conditions) | ~5 hours | **~0 additional** | Training time is independent of embedding dimension. |

The reason is structural: once neurons are assigned to domains (a one-time operation), the learning rule never looks at spatial coordinates again. It operates on domain-level aggregates — mean activations, prediction errors, calcium states — all of which are determined by *which* neurons are in each domain, not by *where* those neurons are in space. The "where" only matters for the initial assignment.

Therefore, the total cost of testing multiple dimensionalities is simply: `n_dimensions × base_experiment_time`. If we test 5 dimensionalities (3D, 4D, 5D, 6D, 8D) and the base experiment takes ~5 hours, that's ~25 hours total. But these are fully parallelizable — each dimensionality is an independent experiment that can run simultaneously on separate cores or machines.

This makes the multi-dimensional ablation one of the cheapest experiments we could run relative to its potential insight. The only real cost is wall-clock time if running sequentially, not computational complexity.

---

## Key References

- Kozachkov et al., "Building Transformers from Neurons and Astrocytes," PNAS, 2023
- IBM Research, "Large Memory Storage Through Astrocyte Computation," 2025
- "Neuromorphic Circuits with Spiking Astrocytes," arXiv:2502.20492, 2025
- "Astrocyte-Gated Multi-Timescale Plasticity for Online Continual Learning," Frontiers in Neuroscience, 2025
- "MA-Net: Rethinking Neural Unit in the Light of Astrocytes," AAAI, 2024
- "Activity-Dependent Myelination: Oscillatory Self-Organization," PNAS, 2020
- "Oligodendrocyte-Mediated Myelin Plasticity and Neural Synchronization," eLife, 2023
