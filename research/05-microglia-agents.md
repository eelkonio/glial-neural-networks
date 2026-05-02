# Step 05: Mobile Pruning Agents with Spatial Patrol

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static (pruning mask is fixed during forward pass)
GLIAL INTERACTION WITH SIGNALS: Learning-only (pruning affects topology, not signal timing)
NOTE: At Level 2+, pruning a connection also removes its delay from the timing graph,
      which can change synchronization patterns. This effect is invisible at Level 1.
```

## The Claim Being Tested

Mobile pruning agents that are spatially localized, accumulate evidence over time, and make context-aware pruning decisions produce better network compression than standard magnitude-based or gradient-based pruning. The spatial constraint (agents can only observe their local territory) is a feature, not a limitation.

## Why This Matters

Standard pruning treats all weights as equally accessible and makes decisions based on global or per-weight statistics. Microglial pruning introduces locality, mobility, evidence accumulation, and spatial context. This step tests whether these properties translate into better pruning outcomes.

## Experiment 5.1: Implement Microglia Agents

### Agent Implementation

```python
class MicrogliaAgent:
    """A mobile pruning agent that patrols the network's spatial embedding."""
    
    def __init__(self, position, territory_radius=0.1, evidence_threshold=0.8):
        self.position = np.array(position)    # Current 3D position
        self.territory_radius = territory_radius
        self.evidence_threshold = evidence_threshold
        self.state = 'surveilling'  # surveilling | activated | pruning | migrating
        
        # Evidence accumulator: per-weight running score
        self.evidence = {}  # weight_idx -> accumulated prune score
        self.observation_count = {}  # weight_idx -> times observed
        
        # Migration state
        self.velocity = np.zeros(3)
        self.migration_target = None
        
    def get_territory(self, weight_positions):
        """Return indices of weights within patrol radius."""
        distances = np.linalg.norm(weight_positions - self.position, axis=1)
        return np.where(distances < self.territory_radius)[0]
    
    def survey(self, territory_indices, weight_stats):
        """Observe weights in territory and accumulate evidence."""
        for idx in territory_indices:
            stats = weight_stats[idx]
            
            # Compute prune score (multi-signal)
            eat_me = 0.0
            eat_me += (1.0 - stats['activation_frequency']) * 0.25
            eat_me += (1.0 - stats['gradient_magnitude_normalized']) * 0.25
            eat_me += stats['redundancy_score'] * 0.25
            eat_me += (1.0 - stats['weight_magnitude_normalized']) * 0.25
            
            # Compute protect score
            protect = 0.0
            protect += stats['correlation_with_neighbors'] * 0.3
            protect += stats['is_unique_path'] * 0.4
            protect += stats['astrocyte_protect_signal'] * 0.3
            
            # Net evidence (positive = should prune)
            net = eat_me - protect
            
            # Accumulate with exponential moving average
            if idx not in self.evidence:
                self.evidence[idx] = 0.0
                self.observation_count[idx] = 0
            
            self.observation_count[idx] += 1
            alpha = 1.0 / (self.observation_count[idx] + 10)  # Slow accumulation
            self.evidence[idx] = (1 - alpha) * self.evidence[idx] + alpha * net
    
    def decide(self):
        """Decide whether to prune, migrate, or continue surveilling."""
        if not self.evidence:
            return 'continue', None
        
        # Check for pruning candidates
        max_idx = max(self.evidence, key=self.evidence.get)
        max_evidence = self.evidence[max_idx]
        
        if max_evidence > self.evidence_threshold:
            self.state = 'pruning'
            return 'prune', max_idx
        
        return 'continue', None
    
    def migrate(self, attraction_field, other_agent_positions, dt=0.01):
        """Move toward high-attraction regions, away from other agents."""
        # Attraction: gradient of error/distress field at current position
        attraction = compute_field_gradient(attraction_field, self.position)
        
        # Repulsion: from other agents (territorial)
        repulsion = np.zeros(3)
        for other_pos in other_agent_positions:
            diff = self.position - other_pos
            dist = np.linalg.norm(diff) + 1e-8
            if dist < self.territory_radius * 3:
                repulsion += diff / (dist**3)
        
        # Random walk (exploration)
        noise = np.random.randn(3) * 0.01
        
        # Update velocity and position
        self.velocity = 0.9 * self.velocity + dt * (
            0.5 * attraction + 0.3 * repulsion + 0.2 * noise
        )
        self.position += self.velocity * dt
        
        # Clear evidence for weights no longer in territory
        # (agent has moved away, old observations become stale)
        self.decay_old_evidence()
```

### Agent Pool Manager

```python
class MicrogliaPool:
    """Manages a population of microglia agents."""
    
    def __init__(self, n_agents, weight_positions, network):
        self.weight_positions = weight_positions
        self.network = network
        
        # Initialize agents at random positions
        self.agents = []
        for _ in range(n_agents):
            pos = weight_positions[np.random.randint(len(weight_positions))]
            self.agents.append(MicrogliaAgent(pos))
        
        # Pruning mask (1 = active, 0 = pruned)
        self.mask = np.ones(len(weight_positions))
        
        # Attraction field (updated periodically)
        self.attraction_field = np.zeros(len(weight_positions))
    
    def update_attraction_field(self, losses_per_region, gradient_variances):
        """Compute what attracts agents: high error, high variance."""
        self.attraction_field = (
            0.6 * losses_per_region / (losses_per_region.max() + 1e-8) +
            0.4 * gradient_variances / (gradient_variances.max() + 1e-8)
        )
    
    def step(self, weight_stats):
        """One step of the microglia pool."""
        prune_events = []
        
        for agent in self.agents:
            # Get territory
            territory = agent.get_territory(self.weight_positions)
            # Only survey active (non-pruned) weights
            active_territory = territory[self.mask[territory] == 1]
            
            if len(active_territory) > 0:
                # Survey
                agent.survey(active_territory, weight_stats)
                
                # Decide
                action, target = agent.decide()
                
                if action == 'prune' and target is not None:
                    self.mask[target] = 0
                    prune_events.append(target)
                    agent.evidence.pop(target, None)
                    agent.state = 'surveilling'
            
            # Migrate
            other_positions = [a.position for a in self.agents if a is not agent]
            agent.migrate(self.attraction_field, other_positions)
        
        return prune_events
```

## Experiment 5.2: Microglia Pruning vs. Standard Pruning

### Baselines

1. **No pruning**: Full dense network
2. **Magnitude pruning**: Remove smallest weights globally (one-shot at various sparsities)
3. **Gradual magnitude pruning**: Iteratively remove smallest weights during training
4. **Movement pruning**: Remove weights with smallest gradient * weight product
5. **Random pruning**: Remove random weights (lower bound)

### Microglia Variants

6. **Microglia (full)**: Mobile agents with evidence accumulation and spatial context
7. **Microglia (no mobility)**: Agents fixed in place (ablation: is mobility important?)
8. **Microglia (no evidence)**: Agents prune immediately based on single observation (ablation: is accumulation important?)
9. **Microglia (no spatial context)**: Agents use only per-weight stats, ignore neighbors (ablation: is spatial context important?)

### Protocol

Train MLP on CIFAR-10 to target sparsities of [50%, 70%, 80%, 90%, 95%].
For each method and sparsity level, measure:
- Final test accuracy
- Accuracy at same compute budget (wall-clock time)
- Pruning distribution (where in the network were weights removed?)
- Connectivity preservation (are there disconnected subgraphs?)

## Experiment 5.3: Agent Count and Territory Size

### The Question

How many agents are needed, and how large should their territories be?

### Protocol

For a 256-256-256 MLP (~200K weights), sweep:
- n_agents: [2, 4, 8, 16, 32, 64]
- territory_radius: [0.05, 0.1, 0.15, 0.2, 0.3] (fraction of embedding diameter)

Measure: pruning quality (accuracy at 80% sparsity) and compute overhead.

### Expected Result

- Too few agents: pruning is too slow, network remains bloated
- Too many agents: overhead is high, agents interfere with each other
- Territory too small: agents can't assess spatial context
- Territory too large: agents lose locality benefit

## Experiment 5.4: Pruning During Training vs. Post-Training

### The Question

Should microglia prune during training (continuous remodeling) or after training (post-hoc compression)?

### Protocol

1. **During training**: Agents active from step 0, pruning continuously
2. **Delayed start**: Agents activate after 50% of training (let network learn first)
3. **Post-training**: Train full network, then deploy agents to prune
4. **Developmental schedule**: Aggressive early pruning, tapering off (biological analog)

### Hypothesis

The developmental schedule (aggressive early, tapering late) should work best, matching the biological observation that synaptic pruning is most intense during development and decreases with maturity.

## Experiment 5.5: Interaction with Astrocyte Layer

### The Question

Do microglia agents make better decisions when they can read astrocyte state?

### Protocol

1. **Microglia only**: No astrocyte layer, agents use raw weight statistics
2. **Microglia + astrocyte signals**: Agents receive "protect" signals from astrocytes for weights in high-calcium (high-activity) domains
3. **Microglia + astrocyte distress**: Agents are attracted to regions where astrocytes show distress (high calcium sustained = reactive state)

### Expected Result

Astrocyte signals should improve pruning quality by:
- Preventing pruning of actively-used weights (protect signal)
- Directing agents to problematic regions (distress signal)

## Success Criteria

- Full microglia pruning achieves higher accuracy than magnitude pruning at same sparsity
- Ablation studies show that mobility, evidence accumulation, AND spatial context each contribute
- Developmental pruning schedule outperforms constant-rate pruning
- Astrocyte interaction improves pruning decisions measurably
- **Redundancy and uniqueness metrics are validated** (high-redundancy weights can be removed without accuracy loss)
- **Permuted embedding control shows reduced benefit** (spatial context matters, not just evidence accumulation)

## Concrete Metric Definitions (from Critical Review 3)

The `survey()` function uses `redundancy_score` and `is_unique_path` which require concrete, validated definitions:

### redundancy_score

**Definition**: The maximum cosine similarity between a weight's activation pattern and any other weight's activation pattern within the same layer.

```python
def compute_redundancy_score(weight_idx, layer_activations, n_samples=100):
    """How replaceable is this weight?
    
    High redundancy = another weight in the same layer produces nearly
    identical activation patterns, so this weight is dispensable.
    """
    # Get activation pattern for this weight across n_samples inputs
    target_pattern = layer_activations[:, weight_idx]  # (n_samples,)
    
    # Compare to all other weights in same layer
    all_patterns = layer_activations  # (n_samples, n_weights_in_layer)
    
    # Cosine similarity with every other weight
    similarities = cosine_similarity(target_pattern, all_patterns)
    similarities[weight_idx] = 0  # Exclude self
    
    return similarities.max()  # Max similarity to any other weight
```

**Validation experiment**: Remove the top-10% highest-redundancy weights. If accuracy drops less than removing random weights, the metric is valid.

### is_unique_path

**Definition**: Binary indicator of whether removing this weight would disconnect any input-output path in the network graph (considering only weights above a magnitude threshold).

```python
def compute_is_unique_path(weight_idx, adjacency_matrix, magnitude_threshold=0.01):
    """Is this weight the only connection between two nodes?
    
    A weight is a unique path if removing it (and all weights below
    magnitude_threshold) would disconnect the graph between its
    source and target neurons.
    """
    # Build graph of "significant" connections (above threshold)
    significant_graph = adjacency_matrix > magnitude_threshold
    
    # Remove this weight
    modified_graph = significant_graph.copy()
    modified_graph[source_neuron, target_neuron] = False
    
    # Check if source can still reach target via alternative paths
    reachable = bfs(modified_graph, source_neuron)
    return 1.0 if target_neuron not in reachable else 0.0
```

**Validation experiment**: Remove weights with `is_unique_path = 1.0`. Accuracy should drop significantly more than removing random weights of similar magnitude.

### Metric Validation Protocol (Experiment 5.0 — runs before 5.2)

Before using these metrics in the full pruning comparison:
1. Compute redundancy_score for all weights in a trained network
2. Remove top-10% highest-redundancy weights → measure accuracy drop
3. Remove top-10% lowest-redundancy weights → measure accuracy drop
4. If high-redundancy removal causes LESS accuracy drop than low-redundancy removal, the metric is validated
5. Repeat for is_unique_path: removing unique-path weights should cause MORE damage

If either metric fails validation, replace with simpler alternatives (e.g., weight magnitude × gradient magnitude for redundancy, node degree for uniqueness).

## Experiment 5.6: Bayesian Evidence Accumulation (from Critical Review 3)

### The Question

The current evidence accumulation is heuristic (weighted sum of eat_me signals). A principled alternative: each agent maintains a posterior probability that each weight is "useless."

### Bayesian Formulation

```python
class BayesianMicrogliaAgent(MicrogliaAgent):
    """Microglia agent with Bayesian evidence accumulation."""
    
    def __init__(self, position, prior_useless=0.1, **kwargs):
        super().__init__(position, **kwargs)
        self.prior_useless = prior_useless  # Sparsity prior
        self.log_posterior = {}  # weight_idx -> log P(useless | observations)
    
    def survey_bayesian(self, territory_indices, weight_stats):
        """Bayesian update of pruning posterior."""
        for idx in territory_indices:
            stats = weight_stats[idx]
            
            # Initialize with prior
            if idx not in self.log_posterior:
                self.log_posterior[idx] = np.log(self.prior_useless / (1 - self.prior_useless))
            
            # Likelihood ratio: P(observations | useless) / P(observations | useful)
            # Useless weights: low activation, low gradient, high redundancy
            log_lr = 0.0
            log_lr += self._activation_likelihood_ratio(stats['activation_frequency'])
            log_lr += self._gradient_likelihood_ratio(stats['gradient_magnitude_normalized'])
            log_lr += self._redundancy_likelihood_ratio(stats['redundancy_score'])
            
            # Bayesian update
            self.log_posterior[idx] += log_lr
    
    def _activation_likelihood_ratio(self, freq):
        """P(freq | useless) / P(freq | useful)"""
        # Useless weights have low activation frequency
        p_useless = scipy.stats.beta.pdf(freq, a=1, b=5)  # Skewed toward 0
        p_useful = scipy.stats.beta.pdf(freq, a=2, b=2)   # Uniform-ish
        return np.log(p_useless / (p_useful + 1e-10) + 1e-10)
    
    def decide_bayesian(self):
        """Prune when posterior probability of uselessness exceeds threshold."""
        if not self.log_posterior:
            return 'continue', None
        
        max_idx = max(self.log_posterior, key=self.log_posterior.get)
        posterior_useless = sigmoid(self.log_posterior[max_idx])
        
        if posterior_useless > self.evidence_threshold:
            return 'prune', max_idx
        
        return 'continue', None
```

### Comparison

Add to the ablation study:
- **Heuristic evidence** (original): Weighted sum of eat_me signals
- **Bayesian evidence**: Posterior probability with calibrated likelihoods

### Expected Benefit

- Principled uncertainty quantification (agent knows how confident it is)
- Natural connection to variational dropout literature
- Less susceptible to spurious decisions (requires consistent evidence)
- Calibratable threshold (posterior probability has a clear interpretation)

## Experiment 5.7: Permuted Embedding Control

### The Question (from Critical Review 3)

Does spatial context actually help microglia pruning, or would the same evidence accumulation work without meaningful spatial structure?

### Protocol

1. Microglia with good embedding (from Step 01)
2. Microglia with randomly permuted embedding (same positions, shuffled assignment)
3. Compare pruning quality at matched sparsity levels

### Interpretation

- If permuted performs similarly → spatial context is not the mechanism; evidence accumulation alone is sufficient
- If permuted performs worse → spatial locality genuinely helps pruning decisions

## Deliverables

- `src/microglia.py`: MicrogliaAgent and MicrogliaPool classes
- `src/weight_stats.py`: Weight statistics computation (activation frequency, redundancy, etc.)
- `experiments/pruning_comparison.py`: Full comparison against baselines
- `experiments/ablation_study.py`: Ablation of microglia components
- `results/pruning_accuracy_curves.png`: Accuracy vs. sparsity for all methods
- `results/agent_trajectories.mp4`: Visualization of agent movement during training

## Estimated Timeline

4-5 weeks. Agent implementation is moderate; the many comparison experiments take time.
