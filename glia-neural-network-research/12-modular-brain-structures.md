# Modular Brain Structures: A Library of Composable Glia-Neural Components

## Vision

The long-term goal of this research program is not just to build a single glia-augmented neural network, but to develop a **library of pre-configured, composable brain-like structures** — each with distinct internal architecture (layer types, glial densities, connectivity patterns, timescale profiles) tuned for specific computational roles. These structures can be assembled into larger systems, much like biological brains are assembled from distinct regions with specialized functions.

This is analogous to how:
- The biological brain has a hippocampus (memory formation), a cerebellum (motor coordination), a visual cortex (hierarchical feature extraction), an amygdala (threat detection), a prefrontal cortex (planning and inhibition), etc.
- Each region has distinct cytoarchitecture: different layer counts, different neuron types, different glia-to-neuron ratios, different connectivity patterns, different dominant timescales
- Regions connect to each other through defined pathways (white matter tracts) with specific delay and bandwidth characteristics
- The whole system functions as an integrated organism because the regions are designed to interface with each other

We aim to create the artificial equivalent: a catalog of **structural primitives** that can be instantiated, configured, and connected to build task-specific intelligent systems.

---

## What a "Brain Part" Is

A brain part (module, structure, component) in our framework is a self-contained unit that includes:

### 1. Neural Architecture
- Number and arrangement of neurons
- Internal connectivity pattern (feedforward, recurrent, lateral, columnar)
- Neuron types (excitatory/inhibitory ratio, time constants, thresholds)
- Layer structure (how many layers, how they connect internally)

### 2. Glial Configuration
- Astrocyte density and domain size
- Astrocyte coupling topology (gap junction pattern)
- Microglia population and patrol policy
- Myelination profile (which internal pathways are fast/slow)
- Volume transmission parameters (diffusion rates, chemical types)
- Timescale profile (how fast each glial subsystem operates)

### 3. Interface Specification
- Input ports: what signals it accepts, in what format, at what rate
- Output ports: what signals it produces, in what format, at what rate
- Modulation ports: what external signals can influence its behavior
- Spatial footprint: how much volume it occupies in the spatial embedding

### 4. Behavioral Preset
- What computational function it performs
- What learning rule it uses internally
- What its default parameters are
- How it responds to standard input patterns
- What its failure modes are

---

## Proposed Catalog of Structures

### Sensory Processing Structures

#### Visual Cortex Analog
```
Purpose: Hierarchical feature extraction from spatial input
Architecture:
  - 6 layers (mimicking V1-V4 hierarchy)
  - Columnar organization (groups of neurons processing same spatial location)
  - Lateral inhibition within layers (competition between features)
  - Feedforward + feedback connections between layers
Glial profile:
  - High astrocyte density (fine-grained modulation)
  - Small astrocyte domains (per-column modulation)
  - Strong heterosynaptic plasticity (feature competition)
  - Moderate myelination (fast feedforward, slower feedback)
  - Low microglia activity (stable structure after development)
Learning: Local Hebbian with astrocyte gating (feature learning)
Input: 2D spatial array (pixel-like)
Output: Feature maps at multiple abstraction levels
```

#### Auditory Cortex Analog
```
Purpose: Temporal pattern extraction from sequential input
Architecture:
  - Tonotopic organization (frequency-to-position mapping)
  - Strong recurrent connections (temporal integration)
  - Delay lines (for temporal pattern matching)
  - Coincidence detectors (for binaural processing)
Glial profile:
  - Heavy myelination (precise timing is critical)
  - Oligodendrocyte-dominated (timing control)
  - Medium astrocyte density
  - Large astrocyte domains (broad temporal integration)
Learning: STDP (timing-dependent, requires Level 2+)
Input: Frequency-decomposed temporal signal
Output: Temporal pattern classifications, rhythm detection
```

#### Somatosensory Analog
```
Purpose: Spatial-temporal integration of touch/proprioception
Architecture:
  - Somatotopic map (body-surface-to-network mapping)
  - Multi-scale receptive fields (fine detail + broad context)
  - Lateral inhibition (spatial sharpening)
Glial profile:
  - Variable astrocyte domain sizes (matching receptive field sizes)
  - Activity-dependent myelination (frequently used pathways speed up)
  - Moderate microglia (ongoing refinement of receptive fields)
Learning: Hebbian with spatial correlation (map formation)
Input: Spatially organized sensor array
Output: Object recognition, texture classification, spatial localization
```

### Memory Structures

#### Hippocampus Analog
```
Purpose: Rapid one-shot memory formation, pattern completion, spatial navigation
Architecture:
  - Trisynaptic loop (DG → CA3 → CA1 equivalent)
  - Sparse coding in DG-equivalent (pattern separation)
  - Recurrent connections in CA3-equivalent (pattern completion/autoassociation)
  - Comparator in CA1-equivalent (novelty detection)
Glial profile:
  - Very high astrocyte activity (plasticity gating is critical here)
  - Rapid D-serine dynamics (fast gating for one-shot learning)
  - Strong volume transmission (coordinate encoding across subregions)
  - Low myelination (flexibility over speed)
  - Active microglia (ongoing memory consolidation via pruning)
Learning: Three-factor with astrocyte gating (rapid, gated Hebbian)
Input: Multimodal pattern (from other structures)
Output: Memory traces, pattern completions, novelty signals
Special: Can "replay" stored patterns during consolidation phases
```

#### Working Memory Buffer
```
Purpose: Short-term maintenance of active information
Architecture:
  - Recurrent loops with self-sustaining activity
  - Inhibitory gating (controls what enters/exits the buffer)
  - Limited capacity (competition between items)
Glial profile:
  - Astrocyte calcium dynamics ARE the memory (slow decay = persistence)
  - Gap junction coupling maintains coherent buffer state
  - No microglia pruning (structure must remain stable)
  - High myelination (fast recurrent loops for maintenance)
Learning: Minimal weight changes; state is in astrocyte calcium, not weights
Input: Gated input (only enters when gate opens)
Output: Current buffer contents (read without destroying)
Special: Capacity limited by number of astrocyte domains
```

#### Long-Term Memory Consolidation
```
Purpose: Transfer from hippocampus-like rapid storage to stable cortical storage
Architecture:
  - Slow interleaved replay of hippocampal traces
  - Gradual weight changes in cortical target structures
  - Sleep-like consolidation phases
Glial profile:
  - Microglia-driven pruning (remove redundant connections after consolidation)
  - Progressive myelination (stabilize consolidated pathways)
  - Astrocyte-mediated synaptic scaling (normalize after consolidation)
Learning: Slow Hebbian during replay, followed by myelination
Input: Replay signals from hippocampus analog
Output: Stable representations in connected cortical structures
Special: Operates primarily during "sleep" phases
```

### Motor/Output Structures

#### Motor Cortex Analog
```
Purpose: Generate precisely timed output sequences
Architecture:
  - Hierarchical: high-level goals → movement plans → motor commands
  - Timing circuits (delay chains for sequence generation)
  - Feedback loops (for online correction)
Glial profile:
  - Very heavy myelination (precise timing is essential)
  - Oligodendrocyte-dominated
  - Low astrocyte plasticity (stable motor programs)
  - Microglia maintain established pathways
Learning: Initially high plasticity (skill acquisition), then myelination locks it in
Input: Goal signals from planning structures
Output: Precisely timed activation sequences
Special: Myelination level indicates skill mastery
```

#### Cerebellum Analog
```
Purpose: Error correction, timing calibration, predictive models
Architecture:
  - Massive fan-out (one input → many parallel fibers)
  - Single modifiable synapse per output (Purkinje cell analog)
  - Climbing fiber error signal (single strong teaching input)
  - Inhibitory output (suppresses incorrect responses)
Glial profile:
  - Bergmann glia analog (specialized astrocytes for precise synaptic control)
  - Very fine-grained astrocyte domains (per-synapse modulation)
  - Heavy myelination of input pathways (precise timing)
  - Low microglia (stable structure)
Learning: Supervised by climbing fiber error; astrocyte gates which synapses learn
Input: Sensory context (mossy fibers) + error signal (climbing fiber)
Output: Corrective signals (inhibitory)
Special: Extremely fast learning at single synapses; slow structural change
```

### Regulatory/Control Structures

#### Amygdala Analog (Threat/Salience Detection)
```
Purpose: Rapid detection of salient/threatening patterns, emotional tagging
Architecture:
  - Fast pathway (crude but quick pattern matching)
  - Slow pathway (detailed analysis, can override fast pathway)
  - Broad output projections (modulates many other structures)
Glial profile:
  - Reactive astrocytes (rapid state changes in response to threat)
  - Strong volume transmission (broadcast alert signals widely)
  - Moderate myelination of fast pathway (speed matters)
  - Active microglia (threat learning involves structural changes)
Learning: Rapid fear conditioning (one-shot, astrocyte-gated)
Input: Sensory patterns (from sensory structures)
Output: Salience/threat signal (broadcast to all connected structures)
Special: Can override other structures via volume-transmitted alert
```

#### Prefrontal Cortex Analog (Executive Control)
```
Purpose: Planning, inhibition, rule-following, context maintenance
Architecture:
  - Highly recurrent (maintains context over long periods)
  - Sparse, selective connectivity to other structures
  - Inhibitory control over motor and emotional structures
  - Late-myelinating (remains plastic longest)
Glial profile:
  - Large astrocyte domains (broad integration)
  - Slow calcium dynamics (long time constants for context)
  - Late and selective myelination (flexibility preserved)
  - Active microglia (ongoing refinement throughout "lifetime")
  - Strong gap junction coupling (coherent state across large region)
Learning: Slow, context-dependent; three-factor with reward modulation
Input: Context from all other structures
Output: Inhibitory control signals, goal representations, rule signals
Special: Last structure to stabilize; most plastic throughout lifetime
```

#### Basal Ganglia Analog (Action Selection)
```
Purpose: Select one action from competing options; reinforcement learning
Architecture:
  - Competitive inhibition (winner-take-all between action candidates)
  - Direct pathway (facilitate selected action)
  - Indirect pathway (suppress competing actions)
  - Dopamine-modulated plasticity
Glial profile:
  - Moderate astrocyte density
  - Volume transmission carries reward/dopamine signal
  - Microglia prune unused action pathways
  - Myelination of frequently selected actions (habit formation)
Learning: Reward-modulated three-factor (dopamine as third factor)
Input: Action candidates from cortical structures
Output: Selected action (disinhibition of one motor program)
Special: Reward signal delivered via volume transmission
```

### Integration Structures

#### Thalamus Analog (Relay and Gating)
```
Purpose: Route information between structures; gate what reaches cortex
Architecture:
  - Point-to-point relay connections (faithful signal transmission)
  - Inhibitory gating (reticular nucleus analog)
  - Feedback from cortex controls what gets relayed
Glial profile:
  - Heavy myelination (fast, faithful relay)
  - Low astrocyte modulation (don't distort the signal)
  - Minimal microglia (stable relay structure)
  - Gating controlled by external inhibitory input
Learning: Minimal (relay doesn't learn; gating is controlled externally)
Input: Signals from sensory structures and subcortical regions
Output: Gated, relayed signals to cortical structures
Special: The "switchboard" — controls information routing
```

#### Corpus Callosum Analog (Inter-Hemisphere Communication)
```
Purpose: Long-range communication between distant structures
Architecture:
  - Long-range axonal projections
  - High bandwidth (many parallel fibers)
  - Bidirectional
Glial profile:
  - Extremely heavy myelination (speed over long distances)
  - Oligodendrocyte-dominated
  - Minimal astrocyte modulation (faithful transmission)
Learning: Myelination adapts to match timing between structures
Input/Output: Bidirectional between connected structures
Special: Delay matching is critical (oligodendrocyte synchronization)
```

---

## Composition Rules

### How Structures Connect

Structures connect through defined **tracts** (analogous to white matter):

```
Structure A ──[tract]──> Structure B

Tract properties:
  - Bandwidth (how many parallel connections)
  - Delay (determined by distance + myelination)
  - Modulation (can other structures gate this tract?)
  - Bidirectionality (one-way or two-way?)
  - Plasticity (can new connections form in this tract?)
```

### Spatial Arrangement

When structures are composed into a system, they occupy positions in the shared spatial embedding:

```
+------------------------------------------------------------------+
|                     SPATIAL VOLUME                                 |
|                                                                    |
|   [Visual Cortex]     [Auditory Cortex]     [Somatosensory]      |
|        |                    |                      |               |
|        +--------[Thalamus]--+----------------------+               |
|                     |                                              |
|   [Hippocampus]     |     [Prefrontal Cortex]                    |
|        |            |            |                                 |
|        +---[Working Memory]------+                                |
|                     |                                              |
|   [Amygdala]        |     [Basal Ganglia]                        |
|        |            |            |                                 |
|        +------------+------------+                                |
|                     |                                              |
|              [Motor Cortex]                                       |
|                     |                                              |
|              [Cerebellum]                                         |
|                                                                    |
+------------------------------------------------------------------+

Spatial proximity determines:
- Volume transmission reach (nearby structures share chemical signals)
- Tract delay (distant structures have longer communication delays)
- Glial field overlap (adjacent structures may share astrocyte domains at boundaries)
```

### Assembly Patterns

**Pattern 1: Sensory-Motor Loop**
```
Sensory Input → [Sensory Cortex] → [Thalamus] → [Motor Cortex] → Motor Output
                                         ↑
                              [Cerebellum] (error correction)
```

**Pattern 2: Cognitive Architecture**
```
[Sensory Cortices] → [Thalamus] → [Prefrontal Cortex] → [Basal Ganglia] → [Motor]
                          ↑               ↑                      ↑
                   [Hippocampus]    [Working Memory]        [Amygdala]
```

**Pattern 3: Minimal Reactive Agent**
```
Sensor → [Amygdala] → [Motor Cortex] → Actuator
              ↑
        [Basal Ganglia] (action selection)
```

---

## Development Roadmap

### When This Becomes Possible

The modular brain structure library depends on the research plan reaching certain milestones:

| Prerequisite | Required Research Step | Why |
|-------------|----------------------|-----|
| Spatial embedding works | Step 01 | Structures need spatial coordinates |
| Astrocyte domains work | Step 03 | Every structure has astrocyte configuration |
| Microglia pruning works | Step 05 | Structures need developmental pruning |
| Multi-timescale works | Step 08 | Structures have different timescale profiles |
| Continual learning works | Step 09 | Structures must learn without forgetting |
| Full ecosystem works | Step 10 | Structures combine all glial mechanisms |
| Local learning works | Step 13 | Biologically faithful learning within structures |
| Temporal simulation works | Step 17 | Timing between structures matters |
| Myelination works | Step 18 | Tracts between structures need delay control |
| Full spatial sim works | Step 20 | Structures exist in shared spatial volume |

**Earliest possible start**: After Phase 1 (Step 10) — can prototype structures at Level 1
**Full implementation**: After Phase 4 (Step 20) — structures with full temporal-spatial fidelity

### Development Phases for the Library

**Phase A: Single-structure prototypes (after Step 10)**
- Implement one structure at a time
- Validate that each structure performs its intended function in isolation
- Establish the interface specification format

**Phase B: Pairwise composition (after Step 17)**
- Connect two structures and verify they communicate correctly
- Test that timing/delay between structures works
- Validate that glial fields at boundaries interact appropriately

**Phase C: Multi-structure systems (after Step 18)**
- Assemble 3-5 structures into functional systems
- Test emergent behaviors from composition
- Validate that the system is more than the sum of its parts

**Phase D: Full architecture (after Step 20)**
- Assemble complete cognitive architectures from the library
- Test on complex, multi-modal tasks
- Validate that the modular approach scales

---

## Design Principles for the Library

1. **Each structure is self-contained**: It can be instantiated and tested independently
2. **Interfaces are standardized**: All structures use the same port specification format
3. **Glial configuration is the differentiator**: The same neural architecture with different glial profiles produces different computational behavior
4. **Composition is spatial**: Connecting structures means placing them in shared space
5. **Development is biological**: Structures go through a developmental phase (over-connected → pruned → myelinated → stable)
6. **Presets are starting points**: Each structure comes with default parameters that can be tuned for specific applications
7. **The library grows**: New structures can be added as new computational needs are identified

---

## Relationship to Existing AI Architectures

| AI Concept | Brain Structure Analog | Key Difference |
|-----------|----------------------|----------------|
| CNN | Visual Cortex | Glial domains add spatial modulation; myelination adds timing |
| RNN/LSTM | Working Memory + Hippocampus | Astrocyte calcium IS the memory; not just gates |
| Transformer attention | Astrocyte-mediated cross-synapse integration | Spatial, continuous, multi-timescale |
| Reinforcement learning | Basal Ganglia + Dopamine (volume transmission) | Reward is spatially broadcast, not per-weight |
| Mixture of experts | Thalamic gating + multiple cortical structures | Gating is spatial and glial, not learned routing |
| Memory-augmented networks | Hippocampus + Consolidation | Structural memory (topology) + chemical memory (calcium) |
| Self-supervised learning | Predictive coding across structures | Error is volume-transmitted, not backpropagated |

---

## Open Questions

1. **How many structures are needed for general intelligence?** Biology uses ~100 distinct brain regions. Is there a minimal set?
2. **Can structures be learned rather than designed?** Can the system discover its own modular architecture through development?
3. **How do structures negotiate at boundaries?** When two structures share a spatial boundary, how do their glial fields interact?
4. **What is the right granularity?** Is "visual cortex" one structure or six (V1, V2, V3, V4, MT, IT)?
5. **Can structures be hot-swapped?** Can you replace one structure with an upgraded version without retraining the whole system?
