# From Weights to Worlds: How Glial Networks Introduce Spatial Geometry to Artificial Intelligence

## Abstract

Standard artificial neural networks exist in a geometric void. Their connections are entries in a matrix with no physical extent, no spatial relationship to one another, and no concept of proximity. A weight connecting neuron 47 to neuron 312 has no meaningful location relative to the weight connecting neuron 48 to neuron 313, even though in any biological brain, those synapses would be micrometers apart and bathed in the same chemical milieu.

This article argues that the introduction of emulated glial networks forces and rewards the assignment of spatial geometry to artificial neural networks. Once glia are present, weights are no longer abstract numbers in a tensor. They become objects embedded in a simulated physical space, subject to diffusion fields, proximity effects, mobile agents, and volume-transmitted signals. This shift transforms neural network computation from pure linear algebra into something closer to physics: a dynamic, spatially extended system where topology, geometry, and physical law govern computation alongside gradient descent.

---

## 1. The Dimensionality Gap

### 1.1 The Geometric Poverty of Standard ANNs

A conventional neural network is, mathematically, a directed graph with weighted edges. Its computational model is:

```
y = f(Wx + b)
```

In this formulation:

- **W** is a matrix. Its entries have indices (i, j) but no coordinates.
- **Connectivity is binary**: two neurons are either connected (edge exists) or not. There is no concept of "nearby" connections.
- **All weights are equidistant from each other** in the only meaningful sense: they are all equally accessible to the optimizer, equally subject to the same learning rule, equally independent.
- **The graph has no embedding dimension**. Layer 3, neuron 45 does not exist "to the left of" or "above" neuron 46. These spatial metaphors, common in architecture diagrams, are visualization artifacts with no computational meaning.

This is a **1D connectivity graph** in the strongest sense: each connection is a single scalar, and the set of all connections forms an unstructured collection (a vector in parameter space) with no intrinsic geometry.

### 1.2 The Spatial Richness of Biological Neural Tissue

Biological neural tissue is radically different. It exists in three-dimensional physical space, and this spatiality is not incidental but computationally essential:

- **Synapses have physical locations.** Two synapses 5 micrometers apart share chemical environments that synapses 500 micrometers apart do not.
- **Diffusion creates spatial fields.** A molecule released at point A reaches point B with a concentration that falls off as ~1/r (in 3D). This creates continuous gradients, not discrete signals.
- **Cells have physical extent.** An astrocyte's processes span 100-400 micrometers, creating a domain of influence defined by geometry, not by network topology.
- **Agents move through space.** Microglia physically migrate, and their influence is determined by where they currently are, a property that has no analog in standard ANNs.
- **Signals propagate with spatial dynamics.** Calcium waves travel at 15-25 micrometers per second through tissue, creating expanding spheres of influence that respect physical distance.

### 1.3 The Gap, Quantified

| Property | Standard ANN | Biological Neural Tissue |
|----------|-------------|------------------------|
| Dimensionality of connection space | 0D (scalar per weight) | 3D (embedded in physical space) |
| Concept of "nearby" weights | None | Fundamental (shared chemical milieu) |
| Influence propagation | Instantaneous, topology-only | Distance-dependent, diffusive |
| Agent mobility | N/A | Continuous movement through 3D space |
| Field effects | None | Concentration gradients, waves |
| Spatial scale of modulation | Per-weight or per-layer | Continuous domains (50-400 um) |
| Boundary effects | None | Domain boundaries, diffusion barriers |

**The core claim**: This gap is not merely a biological curiosity. It represents missing computational structure. The spatial dimension provides a natural mechanism for correlation, coordination, locality, and hierarchy that standard ANNs must approximate through explicit architectural choices (convolutions, attention, normalization) or cannot achieve at all.

---

## 2. Astrocyte Domains: Spatial Correlation of Learning

### 2.1 From Per-Weight to Per-Region

In a standard ANN with an adaptive optimizer like Adam, each weight maintains its own first and second moment estimates. Weight w_ij's learning rate adaptation is completely independent of weight w_ik's adaptation, even if they share the same pre-synaptic neuron and are "adjacent" in any intuitive sense.

An astrocyte changes this fundamentally. A single biological astrocyte:

- Extends processes that contact **100,000 to 2,000,000 synapses**
- Integrates activity from all of these synapses into a unified internal state (calcium concentration)
- Outputs modulation signals that affect **all synapses in its domain simultaneously**
- Creates a **spatial correlation structure** in learning dynamics that is determined by physical proximity

### 2.2 The Domain as a Computational Primitive

When we assign spatial coordinates to weights in an artificial network and overlay astrocyte units with spatial domains, a new computational primitive emerges: the **modulation field**.

```
Standard ANN parameter space:

    w1  w2  w3  w4  w5  w6  w7  w8  w9  w10  w11  w12
    |   |   |   |   |   |   |   |   |   |    |    |
    Independent learning rate per weight (Adam, RMSProp)
    No spatial relationship between weights


Glia-augmented parameter space (with spatial embedding):

    Astrocyte Domain A              Astrocyte Domain B
    +---------------------+        +---------------------+
    |                     |        |                     |
    |  w1  w2  w3  w4  w5|        |w6  w7  w8  w9  w10 |
    |                     |        |                     |
    |  All share astrocyte|        |  All share astrocyte|
    |  A's calcium state  |        |  B's calcium state  |
    |                     |        |                     |
    |  Learning rates are |        |  Learning rates are |
    |  CORRELATED within  |        |  CORRELATED within  |
    |  this domain        |        |  this domain        |
    |                     |        |                     |
    +---------------------+        +---------------------+
              | gap junction coupling |
              +=======================+
              (domains influence each other
               with strength ~ 1/distance)
```

### 2.3 What Spatial Correlation Buys You

Why is spatially correlated learning rate adaptation better than independent per-weight adaptation?

**1. Implicit regularization through spatial smoothness**

Weights that are spatially close learn at similar rates. This prevents the pathological case where adjacent weights in a representation diverge wildly, one learning fast, its neighbor frozen. The spatial smoothness of astrocyte modulation creates a natural regularizer that standard optimizers lack.

**2. Efficient information sharing**

If one weight in a domain discovers that high plasticity is needed (because gradients are large and consistent), the astrocyte propagates this information to all weights in the domain. Nearby weights don't need to independently discover this; they receive the signal through the shared astrocyte state. This is faster than waiting for each weight's individual moment estimates to converge.

**3. Coordinated feature learning**

Weights within an astrocyte domain tend to learn together or stabilize together. This creates natural "feature groups," sets of weights that collectively represent a feature and are collectively managed. This is analogous to how biological astrocyte domains correspond to functional microcolumns in cortex.

**4. Boundary formation**

Where astrocyte domains meet, there is a natural boundary. Weights on opposite sides of a domain boundary can have very different learning dynamics, even if they are topologically adjacent in the network graph. These boundaries create functional compartments without explicit architectural separation.

### 2.4 The Mathematics of Spatial Modulation

For a weight at spatial position **p** = (x, y, z), the effective learning rate becomes:

```
eta_effective(p) = eta_base * G(p)

Where G(p) is the glial modulation field:

G(p) = sum_over_astrocytes_k [ A_k * K(p, c_k, sigma_k) ]

A_k = calcium-dependent output of astrocyte k
c_k = center position of astrocyte k's domain
sigma_k = spatial extent of astrocyte k's domain
K = spatial kernel (e.g., Gaussian: exp(-|p - c_k|^2 / 2*sigma_k^2))
```

This means:
- Learning rate is a **continuous field** over the network's spatial embedding
- The field is **smooth** (determined by overlapping Gaussian-like kernels)
- The field is **dynamic** (A_k changes with astrocyte calcium state)
- The field has **natural length scales** (sigma_k defines domain size)

This is fundamentally different from Adam's per-weight adaptation, which has no spatial structure whatsoever.

---

## 3. Mobile Computational Agents: Microglia and Spatial Patrol

### 3.1 The Concept of Patrol

In a standard ANN, pruning algorithms examine all weights simultaneously (or in arbitrary order). There is no concept of "visiting" a weight; the algorithm has instant access to all parameters.

Emulated microglia break this assumption. They are **spatially localized agents** that:

- Exist at a specific position in the network's spatial embedding
- Can only observe weights within their **patrol territory** (~50-100 um radius in biological terms, mapping to a subset of weights in the artificial system)
- Must **physically migrate** to observe or act on distant weights
- Make decisions based on **local information** gathered during patrol
- Leave behind a **refractory zone** (recently patrolled areas are temporarily ignored)

### 3.2 Why Spatial Patrol Matters

This spatial constraint on pruning agents introduces several computationally valuable properties:

**Temporal ordering of observations**: A microglia agent observes weights sequentially as it patrols, building up a temporal picture of each weight's behavior. This is richer than a single snapshot; it captures dynamics, trends, and correlations over time.

**Evidence accumulation with spatial context**: When a microglia agent evaluates a weight for pruning, it has already observed the weight's spatial neighbors. It knows whether the entire region is underperforming (suggesting a systemic issue) or just this one weight (suggesting true redundancy). This contextual information is unavailable to standard pruning algorithms that evaluate weights independently.

**Natural load balancing**: Multiple microglia agents with territorial behavior (repelled by each other) naturally distribute themselves across the network. High-error regions attract more agents (chemotaxis), creating adaptive allocation of pruning resources without central coordination.

**Exploration-exploitation tradeoff**: Agents must choose between thoroughly surveying their current territory (exploitation) and migrating to potentially more important regions (exploration). This creates a natural exploration schedule that adapts to network state.

### 3.3 Migration Dynamics

Microglial migration in the spatial embedding follows chemotaxis-like dynamics:

```
Velocity of agent M_i:

v_i = alpha * gradient(attraction_field, position_i)
    + beta * sum_j[ repulsion(position_i, position_j) ]
    + gamma * random_walk_component

Where:
  attraction_field(p) = error_density(p) + astrocyte_distress(p) + novelty(p)
  repulsion(p_i, p_j) = -k / |p_i - p_j|^2  (territorial repulsion)
  random_walk = Brownian motion (ensures exploration)
```

This means:
- Agents **converge on problem areas** (high error attracts them)
- Agents **spread out** (territorial repulsion prevents clustering)
- Agents **explore** (random walk discovers new issues)
- The **density of agents** in a region reflects that region's need for maintenance

### 3.4 The Pruning Decision in Spatial Context

A microglia agent at position **p** evaluating weight **w** considers:

```
Prune score for weight w at position p:

score(w) = intrinsic_weakness(w)           // weight magnitude, gradient, activity
          + neighborhood_redundancy(w, p)   // are spatial neighbors carrying same info?
          + domain_health(p)                // is the astrocyte domain stressed?
          - topological_importance(w)       // would pruning disconnect paths?
          - spatial_isolation(w, p)         // is this the only active weight nearby?

The spatial terms (neighborhood_redundancy, spatial_isolation) are ONLY available
because the agent has a position and has surveyed the local spatial neighborhood.
Standard pruning algorithms cannot compute these terms.
```

---

## 4. Volume Transmission: Spatial Broadcasting Without Wires

### 4.1 The Concept

In standard ANNs, information travels along edges (connections). There is no mechanism for a signal to "broadcast" to all units within a spatial radius. Every communication requires an explicit connection.

Volume transmission is the biological mechanism where signaling molecules are released into extracellular space and diffuse outward, affecting all cells within range. There are no dedicated connections; the signal propagates through physical space according to diffusion laws.

### 4.2 Mechanisms of Spatial Broadcasting

#### Calcium Waves

The most dramatic form of glial spatial signaling:

```
Calcium wave propagation in the astrocyte syncytium:

t=0:    ..........X..........    (initial trigger at center)
t=1:    ........##X##........    (wave expanding)
t=2:    ......####X####......    (wave expanding further)
t=3:    ....######X######....    (100-500 um radius)
t=4:    ..########X########..    (maximum extent)
t=5:    ...#######X#######...    (wave dissipating)

X = trigger point
# = elevated calcium (modulation active)
. = baseline

Properties:
- Speed: 15-25 um/second (MUCH slower than neural signals)
- Range: 100-500 um (covers thousands of synapses)
- Shape: roughly spherical in 3D (circular in 2D embedding)
- Effect: ALL weights within the wave front receive modulation
- Duration: seconds to tens of seconds
```

#### Paracrine Chemical Gradients

Slower, more persistent spatial signals:

```
Chemical gradient from a release point:

Concentration C(r, t) = (Q / (4*pi*D*t)^(3/2)) * exp(-r^2 / (4*D*t))

Where:
  Q = amount released
  D = diffusion coefficient
  r = distance from release point
  t = time since release

This creates a spatial field that:
- Falls off with distance (1/r in steady state)
- Has a characteristic length scale (sqrt(D*t))
- Persists for minutes to hours
- Affects all weights within range regardless of network topology
```

#### ATP/Adenosine Signaling

```
Astrocyte releases ATP at position p_0:

1. ATP diffuses outward: affects all units within ~100 um
2. Ectonucleotidases convert ATP -> ADP -> AMP -> Adenosine
3. Each metabolite has different receptor affinities
4. Creates CONCENTRIC RINGS of different modulation:

   Inner ring (0-30 um):   ATP dominant -> excitatory modulation
   Middle ring (30-70 um): ADP/AMP mix -> transitional
   Outer ring (70-100 um): Adenosine dominant -> inhibitory modulation

   This single release event creates a SPATIAL PATTERN of modulation
   with different effects at different distances, impossible in
   standard ANNs without explicit multi-hop connectivity.
```

### 4.3 Computational Implications of Volume Transmission

**1. Topology-independent communication**

Two weights that share no neural pathway can still influence each other if they are spatially close. This creates "shortcuts" in the information flow that bypass the network's connectivity graph entirely.

**2. Broadcast efficiency**

One release event modulates thousands of weights simultaneously. In a standard ANN, achieving the same effect would require explicit connections from a modulator to every target, O(n) connections for n targets. Volume transmission achieves this with O(1) release events.

**3. Distance-dependent modulation**

The strength of modulation naturally decreases with distance from the source. This creates smooth spatial gradients of influence rather than binary on/off effects. Weights closer to the source are more strongly affected, a natural attention-like mechanism based on spatial proximity.

**4. Temporal dynamics create spatial patterns**

Because diffusion takes time, the spatial pattern of modulation evolves. A wave front sweeps outward, creating a temporal sequence of activation across the spatial embedding. This temporal-spatial coupling encodes information that pure spatial or pure temporal signals cannot.

### 4.4 Implementation: The Modulation Field

Volume transmission is best implemented as a **continuous field** overlaid on the network's spatial embedding:

```
The network exists in a simulated 3D (or 2D) space.
Each weight has coordinates: w_ij -> (x_ij, y_ij, z_ij)

The modulation field M(x, y, z, t) evolves according to:

dM/dt = D * laplacian(M) + S(x,y,z,t) - decay * M

Where:
  D = diffusion coefficient (determines spatial spread rate)
  S = source term (glial release events)
  decay = degradation rate (determines persistence)
  laplacian = spatial second derivative (drives diffusion)

Each weight reads its modulation from the field:
  modulation_ij = M(x_ij, y_ij, z_ij, t)

This is a PARTIAL DIFFERENTIAL EQUATION governing the modulation landscape.
The neural network's learning dynamics are now coupled to a PDE,
a fundamentally different mathematical object than the ODE of gradient descent.
```

---

## 5. Spatially Distributed Memory

### 5.1 Where is Information Stored?

In a standard ANN, information is stored in exactly one place: the weight matrix W. All learned knowledge is encoded as numerical values of weights. There is no other storage medium.

A glia-augmented network stores information in multiple spatially distributed substrates:

| Storage Medium | What It Encodes | Timescale | Spatial Character |
|---------------|----------------|-----------|-------------------|
| Weight values | Learned representations | Long-term | Point (per-weight) |
| Astrocyte calcium state | Recent activity history | Seconds | Domain (100s of weights) |
| Glial field concentrations | Regional context | Minutes | Continuous field |
| Microglial positions | Where attention is needed | Hours | Point (per-agent) |
| Myelination pattern | Pathway importance | Days | Per-connection |
| Network topology (post-pruning) | Structural knowledge | Permanent | Global |
| Domain boundaries | Functional compartments | Hours-days | Spatial boundaries |
| Gap junction conductance | Communication topology | Minutes-hours | Per-junction |

### 5.2 Memory in the Spatial Configuration

The most radical form of spatially distributed memory: information encoded not in any numerical value, but in the **geometric arrangement** of the network itself.

After glial remodeling, the network's spatial topology encodes knowledge:

```
Before learning task A:          After learning task A:
(uniform topology)               (remodeled topology)

o o o o o o o o o               o=o=o o o o=o=o=o
o o o o o o o o o               o=o=o o o o=o=o=o
o o o o o o o o o               o o o o o o o o o
o o o o o o o o o               o o o=o=o=o o o o
o o o o o o o o o               o o o=o=o=o o o o

= means myelinated (fast, stable) connections
o means neurons
(spaces between o with no = means pruned connections)

The PATTERN of which connections were pruned and which were
myelinated IS the memory of task A. Even if you reset all
weight values, the topology retains structural knowledge.
```

### 5.3 The Spatial Pattern as Working Memory

Astrocyte calcium concentrations form a spatial pattern that persists for seconds after the neural activity that created it has ceased. This pattern functions as a form of working memory:

```
Neural activity at t=0:
  Neurons in region A fire strongly
  Neurons in region B are silent

Astrocyte state at t=5 (5 seconds later, neural activity has changed):
  Region A astrocytes: elevated calcium (memory of recent activity)
  Region B astrocytes: baseline calcium

  This spatial pattern of calcium IS the working memory.
  It influences current neural processing even though the
  original neural activity is gone.

Neural activity at t=5:
  New input arrives
  Processing is BIASED by the astrocyte spatial pattern
  Region A weights have enhanced plasticity (calcium-dependent)
  Region B weights have baseline plasticity

  The network "remembers" what happened 5 seconds ago
  through the spatial distribution of glial state.
```

### 5.4 Multi-Scale Spatial Memory Hierarchy

The spatial character of glial memory creates a natural hierarchy based on spatial scale:

```
Finest scale (single synapse, ~1 um):
  Weight value encodes specific learned association

Local scale (astrocyte process, ~10 um):
  Process calcium encodes recent activity at this synapse cluster

Domain scale (astrocyte soma, ~100 um):
  Soma calcium integrates across entire domain
  Represents "what this region has been doing"

Regional scale (calcium wave, ~500 um):
  Wave propagation encodes coordinated regional events
  "Something important happened in this area"

Global scale (network-wide):
  Pattern of domain states encodes overall network context
  "The network is in state X" (learning, consolidating, stressed)
```

Each scale has its own timescale, its own spatial resolution, and its own type of information. Together they form a **spatial memory hierarchy** that has no equivalent in standard ANNs.

---

## 6. The Paradigm Shift: From Weight Updates to Topology Management

### 6.1 What Changes

The introduction of spatial geometry through glial networks shifts the fundamental operation of a neural network:

**Old paradigm (standard ANN):**
```
Repeat:
  1. Forward pass: compute output
  2. Compute loss
  3. Backward pass: compute gradients
  4. Update weights: W <- W - eta * grad(L)

The network IS its weights. Training IS weight updates.
Nothing else changes. Topology is fixed. Space does not exist.
```

**New paradigm (glia-augmented network):**
```
Repeat (fast clock, neural):
  1. Forward pass through spatially-embedded network
  2. Compute loss
  3. Backward pass (gradients modulated by glial field)
  4. Update weights: W <- W - G_field(position) * eta * grad(L)

Repeat (medium clock, astrocyte):
  5. Astrocytes sense regional activity statistics
  6. Update calcium dynamics (reaction-diffusion PDE)
  7. Propagate state through gap junctions (spatial diffusion)
  8. Update modulation field

Repeat (slow clock, microglia):
  9. Agents survey local territory
  10. Accumulate evidence for pruning decisions
  11. Execute structural modifications (prune/regrow)
  12. Migrate to new positions based on spatial gradients

Repeat (very slow clock, oligodendrocyte):
  13. Assess pathway utilization
  14. Adjust connection properties (speed/bandwidth)
  15. Stabilize or destabilize pathways

The network is its weights AND its topology AND its spatial
configuration AND its glial state AND its agent positions.
Training involves ALL of these, on different timescales.
```

### 6.2 Topology as a First-Class Computational Object

In the new paradigm, the network's topology is not a fixed hyperparameter chosen before training. It is a **dynamic, evolving object** shaped by glial agents:

- **Microglia prune connections** -> topology becomes sparser
- **Regrowth signals create new connections** -> topology becomes denser in specific regions
- **Oligodendrocytes stabilize pathways** -> parts of topology become frozen
- **Astrocyte domain boundaries** -> create functional partitions in topology
- **Gap junction modulation** -> glial communication topology itself changes

The topology at any moment encodes the network's developmental history: which tasks it has learned, which regions have been stressed, where resources have been allocated.

### 6.3 Physical Law as Inductive Bias

By embedding the network in simulated physical space and subjecting it to diffusion-like dynamics, we introduce **physical law as inductive bias**:

- **Locality**: nearby things interact more strongly (diffusion falls off with distance)
- **Continuity**: fields are smooth (no discontinuous jumps in modulation)
- **Conservation**: signals don't appear from nowhere (sources and sinks are explicit)
- **Causality**: effects propagate at finite speed (calcium waves have velocity)
- **Symmetry**: isotropic diffusion treats all directions equally (unless barriers exist)

These physical constraints are powerful inductive biases that standard ANNs must learn from data (if they learn them at all). A glia-augmented network gets them for free from its spatial embedding.

### 6.4 Implications for Network Design

This paradigm shift changes how we think about network architecture:

| Design Question | Old Answer | New Answer |
|----------------|-----------|-----------|
| How many layers? | Hyperparameter search | Emerges from glial remodeling |
| How wide per layer? | Hyperparameter search | Starts wide, microglia prune to optimal |
| Which connections? | All-to-all or predefined pattern | Evolves through pruning and regrowth |
| Learning rate schedule? | Cosine decay, warmup, etc. | Emerges from astrocyte dynamics |
| Regularization? | Dropout, weight decay, etc. | Emerges from spatial smoothness of glial field |
| When to stop training? | Early stopping on validation | Network self-stabilizes (myelination) |
| How to handle new tasks? | Fine-tuning, adapters | Glial system allocates new spatial regions |

---

## 7. Spatial Geometry as the Missing Dimension of Intelligence

### 7.1 Why Space Matters for Computation

The argument for spatial geometry in neural networks is not merely aesthetic or biological. It addresses fundamental computational limitations:

**The binding problem**: How do you associate features processed in different parts of the network? In standard ANNs, this requires explicit architectural choices (concatenation, attention). With spatial geometry, features processed in spatially proximate regions are naturally bound through shared glial modulation. Calcium wave synchronization creates temporal correlation between spatially close representations.

**The resource allocation problem**: How do you allocate computational resources to where they are needed? Standard ANNs allocate uniformly (every weight gets the same compute). Spatial geometry enables non-uniform allocation through microglial density: more agents in important regions means more maintenance and optimization there.

**The stability-plasticity dilemma**: How do you learn new things without forgetting old ones? Spatial geometry provides a natural solution: old knowledge occupies myelinated, stable spatial regions while new learning occurs in plastic, unmyelinated regions. The spatial separation prevents interference.

**The communication efficiency problem**: How do you coordinate distant parts of a large network? Standard ANNs require explicit skip connections or attention. Spatial geometry provides volume transmission: broadcast signals that reach all units within a spatial radius without requiring point-to-point connections.

### 7.2 The Network as a Living Tissue

The ultimate implication of introducing spatial geometry through glial networks: the artificial neural network stops being a mathematical abstraction and starts behaving like a **living tissue**.

It has:
- **Anatomy** (spatial arrangement of components)
- **Physiology** (dynamic processes operating on that anatomy)
- **Development** (progressive structural refinement)
- **Homeostasis** (self-regulating stability mechanisms)
- **Pathology** (failure modes with spatial character)
- **Healing** (self-repair through mobile agents)

This is not a metaphor. These are literal computational properties that emerge from embedding a neural network in simulated physical space and coupling it to a glial system that respects spatial law.

The transition from "neural network as linear algebra" to "neural network as simulated tissue" may represent the most significant architectural paradigm shift since the introduction of backpropagation itself. Not because it makes networks bigger or faster, but because it makes them **spatial**, and space, it turns out, is where computation becomes self-organizing.

---

## References

- Kozachkov et al., "Building Transformers from Neurons and Astrocytes," PNAS, 2023. [Link](https://www.pnas.org/doi/10.1073/pnas.2219150120)
- IBM Research, "Large Memory Storage Through Astrocyte Computation," 2025. [Link](https://research.ibm.com/blog/astrocytes-cognition-ai-architectures)
- "Neuromorphic Circuits with Spiking Astrocytes for Increased Energy Efficiency, Fault Tolerance, and Memory Capacitance," arXiv:2502.20492, 2025.
- "Astrocytes as a Mechanism for Contextually-Guided Network Dynamics and Function," PLoS Computational Biology, 2024.
- "Nonlinear Gap Junctions Enable Long-Distance Propagation of Pulsating Calcium Waves in Astrocyte Networks," PLoS Computational Biology, 2010.
- "Activity-Dependent Myelination: A Glial Mechanism of Oscillatory Self-Organization," PNAS, 2020.
- "Oligodendrocyte-Mediated Myelin Plasticity and Its Role in Neural Synchronization," eLife, 2023.
- "MA-Net: Rethinking Neural Unit in the Light of Astrocytes," AAAI, 2024.

*Content was rephrased for compliance with licensing restrictions. Inline links provided to original sources.*
