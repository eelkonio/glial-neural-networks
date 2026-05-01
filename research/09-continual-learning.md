# Step 09: Topology-as-Memory for Catastrophic Forgetting

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static
GLIAL INTERACTION WITH SIGNALS: Learning-only (topology protection affects what CAN be learned)
NOTE: At Level 2+, myelinated (protected) pathways would also be FASTER pathways,
      meaning old knowledge is not just structurally protected but also temporally
      privileged — signals along old pathways arrive sooner than signals along new ones.
      This creates a natural "old knowledge dominates" bias during inference.
```

## The Claim Being Tested

Structural protection (pruned topology + myelinated pathways) prevents catastrophic forgetting more effectively than weight-based regularization (EWC, SI, PackNet). The spatial separation of old knowledge (in stable, pruned regions) from new learning (in plastic, available regions) provides orthogonal protection that weight regularization cannot achieve.

## Why This Matters

The first critical review identified this as one of the strongest claims: "Weight regularization protects numerical values; topological pruning protects structural degrees of freedom. These are orthogonal protection mechanisms, and the combination is genuinely more robust than either alone." This experiment tests that claim directly.

## Experiment 9.1: Continual Learning Benchmark Setup

### Tasks

Use standard continual learning benchmarks:

**Split-MNIST**: MNIST split into 5 tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9)
**Permuted-MNIST**: 5 tasks, each with a different fixed permutation of pixels
**Split-CIFAR-10**: CIFAR-10 split into 5 tasks (2 classes each)
**Sequential tasks with interference**: Tasks that share features but have conflicting outputs

### Network

MLP with 3 hidden layers (400-400-400) for MNIST tasks.
Small CNN for CIFAR-10 tasks.
Both with spatial embedding from Step 01.

## Experiment 9.2: Implement Topology-Based Memory Protection

### Myelination Mechanism

```python
class MyelinationSystem:
    """Stabilizes important pathways by reducing their plasticity."""
    
    def __init__(self, n_weights, myelination_threshold=0.8):
        self.myelin_level = np.zeros(n_weights)  # 0 = unmyelinated, 1 = fully myelinated
        self.threshold = myelination_threshold
        self.activity_history = np.zeros(n_weights)
        
    def update(self, weight_activity, weight_importance):
        """Gradually myelinate consistently active, important weights."""
        # Weights that are consistently active and important get myelinated
        myelination_drive = weight_activity * weight_importance
        
        # Slow myelination (takes many steps)
        self.myelin_level += 0.001 * (myelination_drive - self.myelin_level)
        self.myelin_level = np.clip(self.myelin_level, 0, 1)
    
    def get_plasticity_mask(self):
        """Myelinated weights have reduced plasticity."""
        # Fully myelinated = 0.01x learning rate
        # Unmyelinated = 1.0x learning rate
        return 1.0 - 0.99 * self.myelin_level
    
    def get_myelinated_mask(self):
        """Binary mask of which weights are considered myelinated."""
        return self.myelin_level > self.threshold
```

### Spatial Memory Allocation

```python
class SpatialMemoryAllocator:
    """Allocates spatial regions for new tasks, protecting old regions."""
    
    def __init__(self, weight_positions, astrocyte_network):
        self.positions = weight_positions
        self.astrocytes = astrocyte_network
        self.task_regions = {}  # task_id -> set of weight indices
        self.protected_regions = set()  # indices that are protected
        
    def allocate_for_new_task(self, task_id):
        """Find available (unprotected, unmyelinated) spatial region for new task."""
        available = set(range(len(self.positions))) - self.protected_regions
        
        if len(available) < MIN_REGION_SIZE:
            # No space available: need to negotiate with existing tasks
            # (this is where the stability-plasticity tradeoff lives)
            return self.negotiate_space(task_id)
        
        # Prefer spatially contiguous available regions
        # (so the new task gets a coherent astrocyte domain)
        region = self.find_contiguous_available(available)
        self.task_regions[task_id] = region
        return region
    
    def protect_task(self, task_id):
        """After learning a task, protect its region."""
        if task_id in self.task_regions:
            self.protected_regions.update(self.task_regions[task_id])
```

### Combined System

```python
class TopologicalMemorySystem:
    """Full system: pruning + myelination + spatial allocation."""
    
    def __init__(self, model, weight_positions):
        self.model = model
        self.positions = weight_positions
        self.myelination = MyelinationSystem(model.n_weights)
        self.pruning_mask = np.ones(model.n_weights)  # From microglia
        self.spatial_allocator = SpatialMemoryAllocator(weight_positions, None)
        
    def get_effective_lr_mask(self):
        """Combined mask: pruned weights get 0, myelinated get reduced, rest get full."""
        mask = self.pruning_mask * self.myelination.get_plasticity_mask()
        return mask
    
    def after_task_learned(self, task_id):
        """Called after a task is learned: protect and stabilize."""
        # 1. Identify which weights were important for this task
        important_weights = self.compute_task_importance(task_id)
        
        # 2. Myelinate important weights (reduce future plasticity)
        self.myelination.update(
            weight_activity=important_weights,
            weight_importance=important_weights
        )
        
        # 3. Protect the spatial region
        self.spatial_allocator.protect_task(task_id)
        
        # 4. Prune unimportant weights in this region (free up capacity)
        unimportant = (important_weights < 0.1) & (self.pruning_mask == 1)
        # Don't prune immediately; let microglia handle it naturally
```

## Experiment 9.3: Comparison Against Baselines

### Methods to Compare

1. **Naive fine-tuning**: Train sequentially, no protection (lower bound)
2. **EWC (Elastic Weight Consolidation)**: Penalize changes to important weights
3. **SI (Synaptic Intelligence)**: Online importance estimation + regularization
4. **PackNet**: Prune after each task, freeze pruned structure
5. **Progressive Neural Networks**: Add new capacity per task (upper bound on forgetting)
6. **Topology-only**: Pruning + myelination, no spatial embedding
7. **Spatial topology**: Full system with spatial embedding + astrocyte domains
8. **Full glial**: Spatial topology + microglia + astrocyte modulation

### Metrics

For each method, after learning all 5 tasks:
- **Average accuracy**: Mean accuracy across all tasks
- **Forgetting**: Average drop in accuracy on old tasks after learning new ones
- **Forward transfer**: Does learning task k help with task k+1?
- **Backward transfer**: Does learning task k+1 help with task k?
- **Capacity utilization**: What fraction of network is used?
- **Compute cost**: Total training time

## Experiment 9.4: Structural vs. Weight Protection

### The Core Test

Directly compare the two protection mechanisms:

**Weight protection (EWC-style)**:
```
Loss = task_loss + lambda * sum_i[ F_i * (w_i - w_i_old)^2 ]
```
Protects the *values* of important weights.

**Structural protection (topology-based)**:
```
Effective_LR_i = base_lr * (1 - myelin_i) * pruning_mask_i
```
Protects the *existence and stability* of important pathways.

**Combined**:
```
Both weight regularization AND structural protection active simultaneously.
```

### Protocol

1. Learn task A with each protection method
2. Learn task B (which partially conflicts with A)
3. Measure retention of task A performance
4. Specifically measure: which weights changed? Which were protected?

### Key Measurement

For the combined method: are the two protections truly orthogonal? Measure:
- Weights protected by EWC but not by topology (value-protected, structurally free)
- Weights protected by topology but not by EWC (structurally frozen, value-free)
- Weights protected by both (double protection)
- Weights protected by neither (fully plastic)

If the overlap is low, the protections are genuinely orthogonal and the combination should be strictly better than either alone.

## Experiment 9.5: Spatial Separation of Tasks

### The Question

Does the spatial embedding naturally separate different tasks into different regions? Does this separation emerge from the glial dynamics or need to be imposed?

### Protocol

1. Train on 5 sequential tasks with full glial system
2. After all tasks, analyze: which spatial regions are associated with which tasks?
3. Measure spatial overlap between task representations
4. Compare to: random assignment of tasks to regions (control)

### Visualization

Create a spatial map showing:
- Color = which task primarily uses each weight
- Intensity = how important the weight is for that task
- Boundaries = where task regions meet

### Expected Result

If the glial system works as theorized:
- Tasks should occupy largely non-overlapping spatial regions
- Astrocyte domain boundaries should align with task boundaries
- Myelination should be highest in task-specific regions
- Plastic (unmyelinated) regions should be at boundaries or in unused space

## Experiment 9.6: Capacity Scaling

### The Question

How does the number of tasks the system can learn scale with network size?

### Protocol

For network sizes [1K, 5K, 20K, 100K, 500K weights]:
- How many sequential tasks can be learned before forgetting becomes significant?
- How does this compare to EWC and PackNet at the same network sizes?

### Expected Result

Topology-based protection should scale better than weight regularization because:
- Pruning frees capacity (dead weights become available for new tasks)
- Spatial allocation prevents interference without growing the network
- Myelination cost is O(1) per weight (just a scalar), not O(n) like Fisher matrix

## Success Criteria

- Full glial system achieves lower forgetting than EWC and SI
- Structural and weight protection are measurably orthogonal (overlap < 30%)
- Combined protection outperforms either alone
- Tasks naturally separate into distinct spatial regions
- Capacity scales at least linearly with network size (not sublinearly)

## Deliverables

- `src/myelination.py`: Myelination system
- `src/spatial_allocator.py`: Spatial memory allocation
- `src/topological_memory.py`: Combined topology-based memory system
- `experiments/continual_learning_comparison.py`: Full benchmark comparison
- `experiments/orthogonality_test.py`: Structural vs. weight protection analysis
- `experiments/spatial_separation.py`: Task region visualization
- `results/forgetting_curves.png`: Forgetting over sequential tasks
- `results/task_spatial_map.png`: Spatial distribution of task knowledge
- `results/protection_orthogonality.png`: Venn diagram of protection mechanisms

## Estimated Timeline

4-5 weeks. This is a major experiment with many baselines and measurements.

## Connection to Critical Reviews

The first review noted: "If old knowledge is encoded partly in pruned topology and not only in weight values, then it is structurally protected from gradient-based overwriting in a way that parameter regularization cannot achieve." This experiment directly tests that claim.

The second review noted the topology-as-memory argument "conflates structural scaffolding with knowledge storage." This experiment will clarify: does topology protection preserve *performance* (the thing we care about) or just *structure* (which may or may not help)?
