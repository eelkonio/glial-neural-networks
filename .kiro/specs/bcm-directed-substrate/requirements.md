# Requirements Document

## Introduction

This document specifies the requirements for the BCM-Directed Substrate (Step 12b) — a biologically faithful local learning rule where direction (LTP vs LTD) emerges from postsynaptic calcium levels relative to a sliding threshold (BCM theory). The astrocyte D-serine gate modulates whether synapses can reach high-calcium states, and heterosynaptic competition within astrocyte domains provides local credit assignment. This rule implements the LocalLearningRule protocol from Step 12 and reuses CalciumDynamics and DomainAssignment from Step 13.

## Glossary

- **BCMDirectedRule**: The core learning rule class that computes signed weight updates using BCM theory, D-serine gating, and heterosynaptic competition.
- **Synapse_Calcium**: Per-neuron scalar representing postsynaptic calcium level, computed as the batch-mean of absolute post-activation.
- **Theta**: Sliding BCM threshold per domain, updated via exponential moving average of domain mean activity.
- **Direction**: Signed signal (synapse_calcium - theta) indicating LTP (positive) or LTD (negative) for each neuron.
- **D_Serine_Gate**: Boolean per-domain signal from CalciumDynamics indicating whether the astrocyte has released D-serine (enabling high calcium / LTP).
- **Heterosynaptic_Competition**: Zero-centering of direction within each astrocyte domain, creating local winner-take-all dynamics.
- **LocalMLP**: The multi-layer perceptron from Step 12 that provides forward_with_states for layer-local training.
- **LayerState**: Data structure containing pre_activation, post_activation, weights, and layer_index for one layer.
- **CalciumDynamics**: Li-Rinzel calcium model from Step 13 that drives the D-serine gate.
- **DomainAssignment**: Spatial partitioning of neurons into astrocyte domains from Step 13.
- **Training_Loop**: The per-epoch training procedure that applies BCMDirectedRule to each layer after forward pass.
- **Experiment_Runner**: The comparison framework that evaluates multiple conditions across seeds.

## Requirements

### Requirement 1: BCM Direction Computation

**User Story:** As a researcher, I want the learning rule to derive update direction from local calcium levels relative to a sliding threshold, so that synapses can undergo both LTP and LTD without global error signals.

#### Acceptance Criteria

1. WHEN a LayerState is provided to BCMDirectedRule.compute_update, THE BCMDirectedRule SHALL compute synapse_calcium as the batch-mean of absolute post-activation for each output neuron.
2. WHEN synapse_calcium exceeds the domain theta for a neuron, THE BCMDirectedRule SHALL assign positive direction (LTP) to that neuron.
3. WHEN synapse_calcium is below the domain theta for a neuron, THE BCMDirectedRule SHALL assign negative direction (LTD) to that neuron.
4. THE BCMDirectedRule SHALL produce weight deltas containing both positive and negative values when neurons have varying activity levels relative to theta.

### Requirement 2: Sliding Threshold (Theta) Homeostasis

**User Story:** As a researcher, I want the BCM threshold to adapt to recent activity levels, so that the rule self-regulates and prevents runaway potentiation or depression.

#### Acceptance Criteria

1. WHEN compute_update is called, THE BCMDirectedRule SHALL update theta using exponential moving average: theta = theta_decay × theta + (1 - theta_decay) × domain_activities.
2. WHILE domain activity remains constant at value a, THE BCMDirectedRule SHALL converge theta toward a as the number of steps increases.
3. WHEN domain activity increases, THE BCMDirectedRule SHALL increase theta, making LTP harder to achieve.
4. WHEN domain activity decreases, THE BCMDirectedRule SHALL decrease theta, making LTP easier to achieve.
5. THE BCMDirectedRule SHALL initialize theta to theta_init on first call for each layer.

### Requirement 3: D-Serine Gating

**User Story:** As a researcher, I want astrocyte D-serine release to gate whether synapses can reach high calcium states, so that the biological mechanism of astrocyte-enabled LTP is faithfully modeled.

#### Acceptance Criteria

1. WHEN the D-serine gate is open for a domain, THE BCMDirectedRule SHALL amplify synapse_calcium for neurons in that domain by factor (1 + d_serine_boost).
2. WHEN the D-serine gate is closed for a domain, THE BCMDirectedRule SHALL leave synapse_calcium unchanged for neurons in that domain.
3. WHEN use_d_serine is set to False, THE BCMDirectedRule SHALL skip all D-serine amplification regardless of gate state.
4. THE BCMDirectedRule SHALL step CalciumDynamics with domain_activities each call to drive gate transitions.

### Requirement 4: Heterosynaptic Competition

**User Story:** As a researcher, I want local competition within astrocyte domains to provide credit assignment, so that the most active neurons in a domain get LTP while less active ones get LTD.

#### Acceptance Criteria

1. WHEN competition_strength is 1.0, THE BCMDirectedRule SHALL zero-center direction within each domain so that the mean direction per domain is approximately zero.
2. WHEN competition is applied, THE BCMDirectedRule SHALL preserve the relative ordering of neurons within each domain.
3. WHEN a domain contains fewer than 2 neurons, THE BCMDirectedRule SHALL skip competition for that domain.
4. WHEN use_competition is set to False, THE BCMDirectedRule SHALL skip all heterosynaptic competition.

### Requirement 5: Protocol Compliance and Output Shape

**User Story:** As a researcher, I want BCMDirectedRule to be a drop-in replacement for any LocalLearningRule, so that it integrates with the existing Step 12 training infrastructure.

#### Acceptance Criteria

1. THE BCMDirectedRule SHALL implement compute_update accepting a LayerState and returning a weight delta tensor.
2. THE BCMDirectedRule SHALL return weight deltas with shape exactly matching (out_features, in_features) of the layer weights.
3. THE BCMDirectedRule SHALL implement reset() that clears all internal state (theta).
4. THE BCMDirectedRule SHALL expose a name attribute equal to "bcm_directed".

### Requirement 6: Numerical Stability

**User Story:** As a researcher, I want the learning rule to remain numerically stable under all input conditions, so that training runs complete without NaN or explosion.

#### Acceptance Criteria

1. THE BCMDirectedRule SHALL clip weight delta Frobenius norm to clip_delta.
2. IF synapse_calcium contains NaN values, THEN THE BCMDirectedRule SHALL replace them with zero and continue operation.
3. THE BCMDirectedRule SHALL maintain theta as non-negative by clamping after each update.
4. WHEN CalciumDynamics is stepped, THE CalciumDynamics SHALL keep calcium in [0, ca_max] and h in [0, 1].

### Requirement 7: Training Loop Integration

**User Story:** As a researcher, I want a training loop that applies BCMDirectedRule per-layer after forward pass, so that I can train the network end-to-end with local learning.

#### Acceptance Criteria

1. WHEN train_epoch is called, THE Training_Loop SHALL perform forward_with_states, then compute_update for each layer, then apply weight deltas.
2. THE Training_Loop SHALL compute cross-entropy loss for monitoring without using it for learning.
3. THE Training_Loop SHALL return mean training loss for the epoch.
4. WHEN training completes 5 epochs on FashionMNIST, THE Training_Loop SHALL produce accuracy above 10% (better than chance).

### Requirement 8: Experiment Comparison

**User Story:** As a researcher, I want to compare BCM-directed variants against baselines, so that I can quantify the contribution of each component (D-serine, competition).

#### Acceptance Criteria

1. THE Experiment_Runner SHALL evaluate 5 conditions: bcm_no_astrocyte, bcm_d_serine, bcm_full, three_factor_reward baseline, and backprop.
2. THE Experiment_Runner SHALL run each condition with 3 random seeds for statistical reliability.
3. THE Experiment_Runner SHALL record per-epoch accuracy and loss for each condition and seed.
4. THE Experiment_Runner SHALL produce a results summary comparing all conditions.
5. WHEN the full experiment completes, THE Experiment_Runner SHALL save results with timestamps.

### Requirement 9: Ablation Independence

**User Story:** As a researcher, I want each component (D-serine, competition) to be independently disableable, so that I can isolate the contribution of each mechanism.

#### Acceptance Criteria

1. WHEN use_d_serine is False and use_competition is False, THE BCMDirectedRule SHALL produce updates depending only on post_activation, pre_activation, theta_decay, and lr.
2. WHEN use_d_serine is True and use_competition is False, THE BCMDirectedRule SHALL apply D-serine amplification without zero-centering.
3. WHEN use_d_serine is False and use_competition is True, THE BCMDirectedRule SHALL apply zero-centering without D-serine amplification.

### Requirement 10: Domain Partition Integrity

**User Story:** As a researcher, I want every neuron assigned to exactly one domain with no gaps, so that the competition and theta tracking cover all neurons.

#### Acceptance Criteria

1. THE DomainAssignment SHALL assign every neuron in a layer to exactly one domain.
2. THE DomainAssignment SHALL produce ceil(out_features / domain_size) domains per layer.
3. THE DomainAssignment SHALL ensure no neuron belongs to multiple domains.
