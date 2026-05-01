# Step 19: In-Transit Signal-Glia Interaction

```
SIMULATION FIDELITY: Level 2-3 (Transitional to Full Spatial-Temporal)
SIGNAL MODEL: Signals propagate through spatial grid, interacting with glial fields en route
NETWORK STATE DURING INFERENCE: Continuously evolving (signals and glia co-evolve in real time)
GLIAL INTERACTION WITH SIGNALS: IN-TRANSIT (signals are modulated as they pass through glial regions)
NOTE: This is the critical experiment that determines whether Level 3 is necessary.
      If in-transit interaction produces qualitatively different results from Level 2
      (delayed teleportation), then full spatial-temporal simulation is justified.
      If results are similar, Level 2 is sufficient and Level 3 is unnecessary overhead.
```

## The Claim Being Tested

Signals that propagate through space and interact with glial fields during transit produce qualitatively different computation than signals that simply teleport after a delay. The spatial path matters, not just the endpoints.

## Why This Is the Go/No-Go for Level 3

Level 2 treats signals as "delayed teleportation" — a signal disappears from the source and appears at the destination after d timesteps. Nothing happens to it in between.

Level 3 treats signals as "spatial propagation" — a signal travels through intermediate space, and that space has properties (glial fields, chemical concentrations) that modify the signal during transit.

If these produce the same results, Level 3 is unnecessary. If they differ, Level 3 captures something real.

## Experiment 19.1: Implement Spatial Signal Propagation

### Architecture

```python
class SpatialPropagationEngine:
    """Signals propagate through a spatial grid, interacting with fields en route."""
    
    def __init__(self, spatial_grid, glial_fields, connections):
        """
        spatial_grid: 3D grid defining the simulation volume
        glial_fields: dict of field names -> 3D arrays (astrocyte Ca, volume chemicals, etc.)
        connections: list of (source_pos, target_pos, weight) — paths through space
        """
        self.grid = spatial_grid
        self.fields = glial_fields
        self.connections = connections
        
        # Precompute paths: for each connection, the sequence of grid cells it traverses
        self.paths = {}
        for conn_id, conn in enumerate(connections):
            self.paths[conn_id] = self.compute_path(conn.source_pos, conn.target_pos)
        
        # Signals in flight: (conn_id, current_position_along_path, signal_value)
        self.signals_in_flight = []
    
    def compute_path(self, source, target):
        """Compute the spatial path from source to target (Bresenham-like in 3D)."""
        # Returns list of grid cell indices the signal passes through
        path = bresenham_3d(
            self.grid.pos_to_cell(source),
            self.grid.pos_to_cell(target)
        )
        return path
    
    def propagate_step(self):
        """Advance all signals in flight by one spatial step."""
        arrived = []
        still_flying = []
        
        for signal in self.signals_in_flight:
            conn_id, path_idx, value = signal
            path = self.paths[conn_id]
            
            # Get current grid cell
            current_cell = path[path_idx]
            
            # IN-TRANSIT INTERACTION: signal is modified by local glial field
            local_modulation = self.compute_local_modulation(current_cell)
            value *= local_modulation  # Signal is attenuated/amplified
            
            # BIDIRECTIONAL: signal also affects the local glial field
            self.deposit_signal_trace(current_cell, value)
            
            # Advance along path
            # Speed depends on local myelination field
            local_speed = self.get_local_speed(current_cell)
            new_path_idx = path_idx + local_speed
            
            if new_path_idx >= len(path):
                # Signal has arrived at destination
                arrived.append((conn_id, value))
            else:
                still_flying.append((conn_id, int(new_path_idx), value))
        
        self.signals_in_flight = still_flying
        return arrived
    
    def compute_local_modulation(self, cell):
        """
        How does the local glial environment modify a signal passing through?
        
        Based on:
        - Astrocyte calcium at this location (high Ca = gain modulation)
        - Volume-transmitted chemicals (adenosine = suppression, ATP = facilitation)
        - Extracellular potassium (high K+ = depolarization, affects propagation)
        """
        ca_level = self.fields['astrocyte_calcium'][cell]
        adenosine = self.fields['adenosine'][cell]
        atp = self.fields['atp'][cell]
        potassium = self.fields['extracellular_k'][cell]
        
        # Gain modulation
        gain = 1.0
        gain *= (1.0 + 0.2 * (ca_level - 0.5))      # Ca modulates gain mildly
        gain *= (1.0 - 0.5 * adenosine)               # Adenosine suppresses
        gain *= (1.0 + 0.3 * atp)                     # ATP facilitates
        gain *= (1.0 + 0.1 * (potassium - 3.0))      # K+ affects excitability
        
        return np.clip(gain, 0.1, 3.0)
    
    def deposit_signal_trace(self, cell, signal_value):
        """Signal passing through a cell leaves a trace that glia can detect."""
        # This is how glia "sense" neural activity passing through their territory
        self.fields['neural_activity_trace'][cell] += abs(signal_value) * 0.01
    
    def get_local_speed(self, cell):
        """Propagation speed at this grid cell (myelination-dependent)."""
        myelin = self.fields['myelination'][cell]
        # Speed: 1 cell/step (unmyelinated) to 10 cells/step (fully myelinated)
        return 1 + int(9 * myelin)
```

## Experiment 19.2: Level 2 vs. Level 3 Comparison

### The Critical Experiment

Run the SAME network, SAME task, SAME glial mechanisms under:
- **Level 2**: Signals teleport after delay (no in-transit interaction)
- **Level 3**: Signals propagate through space (with in-transit interaction)

### Protocol

1. Set up identical networks with identical initial conditions
2. Train both for the same number of effective timesteps
3. Compare:
   - Final accuracy
   - Representation quality
   - Emergent behaviors (do new phenomena appear at Level 3?)
   - Glial field dynamics (do fields evolve differently when signals interact with them?)

### What Would Make Level 3 Worthwhile

Level 3 is justified if ANY of these are true:
- Accuracy is measurably higher (>2% on benchmark tasks)
- Qualitatively new behaviors emerge (spatial filtering, interference patterns, resonance)
- The glial system works BETTER when it can sense signals in transit (not just at endpoints)
- Certain tasks become solvable that Level 2 cannot solve

### What Would Make Level 3 Unnecessary

Level 3 is unnecessary if:
- Accuracy is the same (within noise)
- No new behaviors emerge
- The 10-100x additional cost produces no measurable benefit

## Experiment 19.3: Spatial Filtering by Glial Fields

### The Question

Can glial fields act as spatial filters on signals in transit? (Like how a lens focuses light, or how a medium filters frequencies)

### Concept

If the glial field has spatial structure (high-gain regions and low-gain regions), signals passing through different paths will be differentially modulated. This creates a form of spatial filtering:

```
Signal from A to C via path 1 (through high-gain region):
  A ---[high gain]---> C    (signal amplified)

Signal from B to C via path 2 (through low-gain region):
  B ---[low gain]----> C    (signal attenuated)

Result: C receives A's signal more strongly than B's,
        even if the weights are equal, because the PATHS differ.
```

This is computation performed by the spatial medium, not by the weights.

### Protocol

1. Create a network where two inputs have equal-weight paths to an output
2. Establish a glial field with spatial structure (one path through high-gain, one through low-gain)
3. Measure: does the output preferentially respond to the input whose path goes through high-gain?
4. Compare to Level 2: at Level 2, both signals arrive with equal strength (no path effects)

## Experiment 19.4: Bidirectional Signal-Glia Coupling

### The Question

When signals deposit traces as they pass through glial territory, does this create useful feedback loops?

### Concept

```
Signal passes through astrocyte domain
    -> Astrocyte detects signal (activity trace deposited)
    -> Astrocyte calcium rises
    -> Astrocyte modulates gain in its domain
    -> NEXT signal passing through is modulated differently
    -> This creates a form of short-term spatial memory:
       "A signal recently passed through here, so the next one will be treated differently"
```

This is a form of **spatial adaptation** — the medium remembers what passed through it and responds differently to subsequent signals.

### Protocol

1. Send repeated signals along the same path
2. Measure: does the glial response to the Nth signal differ from the 1st?
3. Does this adaptation improve computation? (e.g., novelty detection, habituation)

## Experiment 19.5: Emergent Spatial Phenomena

### What to Look For

Phenomena that can ONLY exist with spatial propagation (impossible at Level 2):

- **Interference**: Two signals traveling through the same region interact (constructive/destructive)
- **Resonance**: Signals bouncing between regions create standing waves in glial fields
- **Spatial filtering**: Glial field structure selectively passes certain signal patterns
- **Path-dependent computation**: Same source-destination pair produces different results depending on which spatial path the signal takes
- **Waveguiding**: Glial fields create "channels" that guide signal propagation along preferred paths

### Measurement

For each phenomenon:
- Does it occur? (Qualitative observation)
- Does it improve computation? (Quantitative measurement)
- Is it controllable? (Can the glial system learn to exploit it?)

## Success Criteria

- Level 3 produces measurably different results from Level 2 on at least one task
- At least one emergent spatial phenomenon is observed
- Bidirectional signal-glia coupling creates useful adaptation
- The go/no-go decision is clear: either Level 3 is justified or it isn't

## Deliverables

- `src/spatial_propagation.py`: SpatialPropagationEngine
- `src/in_transit_modulation.py`: Signal-glia interaction during propagation
- `src/spatial_grid.py`: 3D grid with field storage
- `experiments/level2_vs_level3.py`: The critical comparison
- `experiments/spatial_filtering.py`: Path-dependent signal modulation
- `experiments/emergent_phenomena.py`: Search for spatial phenomena
- `results/level_comparison.csv`: Quantitative Level 2 vs. Level 3 results
- `results/spatial_phenomena_catalog.md`: Observed emergent behaviors

## Estimated Timeline

8-12 weeks. This is novel engineering with many unknowns.

## Go/No-Go Decision

After this step, decide:
- **If Level 3 adds value**: Proceed to Step 20 (full optimized simulation)
- **If Level 3 adds no value**: Stay at Level 2, document why, save the compute
