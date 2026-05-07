# Requirements Document

## Introduction

This document specifies the requirements for Step 14: Predictive Coding + BCM — a local learning rule that combines BCM-directed signed updates (from Step 12b) with inter-layer prediction errors as the missing task-relevant information channel.

The core insight is that each layer can maintain a prediction of the next layer's **domain-level** activation; the prediction error (actual domain activity - predicted domain activity) provides a signed, local, and informative signal that tells the current layer what it is "getting wrong." This prediction error modulates the BCM direction signal, making it task-relevant — neurons in domains whose aggregate activity deviates from prediction receive stronger learning signals.

**Domain-level prediction** is the primary operating mode. Rather than predicting each individual neuron's activation in the next layer (128-dimensional, noisy, expensive), we predict each *domain's* mean activation (8-dimensional with domain_size=16). This is:
- **More biologically faithful** — astrocytes operate at the domain level (~50μm, ~100-1000 synapses), not per-synapse
- **Less noisy** — averaging over ~16 neurons per domain reduces variance in the prediction signal
- **Computationally cheaper** — prediction weights shrink from (128×128) to (8×8) per layer
- **Sufficient for information needs** — the key signal is which domains are "surprised" vs "satisfied"

The system integrates all components built in Steps 12, 13, and 12b:
- BCM direction (Step 12b) — provides signed updates (LTP/LTD)
- Calcium dynamics and D-serine gating (Step 13) — provides magnitude control
- Domain assignment (Step 13) — provides spatial organization and the unit of prediction
- Domain-level prediction errors (new) — provides task-relevant information

The architecture is the same 5-layer LocalMLP (784→128→128→128→128→10) from previous steps. Prediction operates between adjacent layers at the domain level: each layer's domain activities are used to predict the next layer's domain activities.

## Glossary

- **PredictiveBCMRule**: The core learning rule class that combines BCM direction with domain-level prediction error modulation to produce task-relevant signed weight updates.
- **Domain_Activity**: Mean absolute activation of neurons within an astrocyte domain. Shape (n_domains,) per layer.
- **Prediction_Weights**: Per-layer linear mapping P_i from layer i domain activities to predicted layer i+1 domain activities. Shape (n_domains_next, n_domains_current) — typically (8, 8).
- **Domain_Prediction_Error**: Signed difference between actual next-layer domain activities and predicted next-layer domain activities: e_i = actual_domains_{i+1} - P_i(domains_i). Shape (n_domains_next,).
- **Information_Signal**: A per-domain scalar derived from prediction error that modulates the BCM direction for all neurons in that domain. Computed by projecting domain prediction error back through P_i^T. Shape (n_domains_current,).
- **BCM_Direction**: Signed signal (synapse_calcium - theta) from Step 12b indicating LTP or LTD for each neuron. Shape (out_features,).
- **Combined_Update**: The product of BCM direction and domain-level information signal (broadcast to neurons within each domain), yielding task-relevant signed weight updates.
- **LocalMLP**: The multi-layer perceptron from Step 12 providing forward_with_states for layer-local training.
- **LayerState**: Data structure containing pre_activation, post_activation, weights, and layer_index for one layer.
- **CalciumDynamics**: Li-Rinzel calcium model from Step 13 that drives the D-serine gate.
- **DomainAssignment**: Spatial partitioning of neurons into astrocyte domains from Step 13. Domains are the unit of prediction and modulation.
- **BCMDirectedRule**: The BCM direction computation from Step 12b (reused internally).
- **Training_Loop**: The per-epoch training procedure that applies PredictiveBCMRule to each layer after forward pass.
- **Experiment_Runner**: The comparison framework that evaluates multiple conditions across seeds.
- **Prediction_Learning_Rate**: Separate learning rate for updating prediction weights (lr_pred), distinct from the main weight learning rate.
- **Granularity**: Operating level for predictions — "domain" (primary, 8-dimensional) or "neuron" (optional comparison, 128-dimensional).

## Requirements

### Requirement 1: Prediction Weight Initialization and Structure

**User Story:** As a researcher, I want each layer to maintain a linear prediction of the next layer's domain-level activation, so that prediction errors can be computed locally between adjacent layers at the biologically appropriate spatial scale.

#### Acceptance Criteria

1. THE PredictiveBCMRule SHALL maintain prediction weights P_i of shape (n_domains_next, n_domains_current) for each layer i except the last, where n_domains is typically 8 (with domain_size=16 and 128 neurons per layer).
2. WHEN initialized, THE PredictiveBCMRule SHALL set prediction weights using Xavier uniform initialization scaled for the small matrix dimensions.
3. THE PredictiveBCMRule SHALL store prediction weights as torch tensors on the same device as the model weights.
4. WHEN reset is called, THE PredictiveBCMRule SHALL reinitialize all prediction weights to their initial distribution.
5. THE PredictiveBCMRule SHALL support a configurable option to use fixed random prediction weights (feedback alignment style) instead of learned predictions.
6. THE PredictiveBCMRule SHALL support an optional "neuron" granularity mode with prediction weights of shape (next_layer_size, current_layer_size) for comparison experiments.

### Requirement 2: Domain-Level Prediction Error Computation

**User Story:** As a researcher, I want prediction errors computed at the domain level as the difference between actual and predicted next-layer domain activities, so that each layer receives a signed local signal about what its domains collectively get wrong.

#### Acceptance Criteria

1. WHEN compute_update is called for layer i, THE PredictiveBCMRule SHALL compute domain_activities_current as the mean absolute activation per domain for layer i.
2. WHEN compute_update is called for layer i, THE PredictiveBCMRule SHALL compute predicted_next_domains = P_i @ domain_activities_current.
3. WHEN compute_update is called for layer i, THE PredictiveBCMRule SHALL compute domain_prediction_error = actual_next_domains - predicted_next_domains, where actual_next_domains is the mean absolute activation per domain for layer i+1.
4. THE PredictiveBCMRule SHALL produce domain prediction errors that are signed (containing both positive and negative values across domains).
5. FOR the last hidden layer (layer before output), THE PredictiveBCMRule SHALL use the output layer's domain activities as the actual next-layer domains, OR use one-hot encoded labels aggregated to domain level as the prediction target (configurable).
6. THE domain prediction error SHALL have shape (n_domains_next,) — typically 8-dimensional, providing a compact but informative error signal.

#### Acceptance Criteria

1. WHEN compute_update is called for layer i, THE PredictiveBCMRule SHALL compute predicted_next = P_i @ mean_output_i (batch-mean of current layer output).
2. WHEN compute_update is called for layer i, THE PredictiveBCMRule SHALL compute prediction_error = mean_actual_next - predicted_next, where mean_actual_next is the batch-mean of the actual next layer activation.
3. THE PredictiveBCMRule SHALL produce prediction errors that are signed (containing both positive and negative values).
4. WHEN computing prediction error for the last hidden layer (layer before output), THE PredictiveBCMRule SHALL use the network output (logits) as the actual next-layer activation.
5. IF the actual next-layer activation is not available (last layer in sequence), THEN THE PredictiveBCMRule SHALL use the target labels (one-hot encoded) as the prediction target.

### Requirement 3: Information Signal from Domain Prediction Error

**User Story:** As a researcher, I want domain prediction errors projected back to the current layer's domains to produce a per-domain information signal, so that each domain knows how much its aggregate output contributes to prediction failures downstream.

#### Acceptance Criteria

1. WHEN domain_prediction_error is computed for layer i, THE PredictiveBCMRule SHALL compute domain_information_signal = P_i^T @ domain_prediction_error, producing a vector of shape (n_domains_current,).
2. THE PredictiveBCMRule SHALL broadcast the domain_information_signal to all neurons within each domain: each neuron receives the information signal of its domain.
3. THE PredictiveBCMRule SHALL normalize the domain_information_signal by dividing by its L2 norm plus epsilon to prevent scale issues.
4. THE PredictiveBCMRule SHALL produce information signals that are signed (both positive and negative values across domains).
5. WHEN the domain prediction error is zero (perfect prediction), THE PredictiveBCMRule SHALL produce a zero information signal.
6. THE information signal broadcast SHALL preserve the domain structure: all neurons in domain d receive the same information_signal[d] value.

### Requirement 4: BCM Direction with Prediction Error Modulation

**User Story:** As a researcher, I want the BCM direction signal modulated by the prediction error information, so that the combined signal is both signed (from BCM) and task-relevant (from prediction error).

#### Acceptance Criteria

1. WHEN compute_update is called, THE PredictiveBCMRule SHALL compute BCM direction as (synapse_calcium - theta) following the Step 12b algorithm.
2. THE PredictiveBCMRule SHALL combine BCM direction and information signal via element-wise multiplication: combined = direction * information_signal.
3. WHEN both direction and information_signal are positive for a neuron, THE PredictiveBCMRule SHALL produce a positive (LTP) update for that neuron.
4. WHEN direction and information_signal have opposite signs for a neuron, THE PredictiveBCMRule SHALL produce a negative (LTD) update for that neuron.
5. THE PredictiveBCMRule SHALL support a configurable combination mode (multiplicative, additive, or threshold modulation) with multiplicative as the default.

### Requirement 5: Prediction Weight Learning

**User Story:** As a researcher, I want prediction weights to learn from domain-level prediction errors, so that predictions improve over training and the error signal becomes more informative.

#### Acceptance Criteria

1. WHEN compute_update is called, THE PredictiveBCMRule SHALL update prediction weights using: delta_P = lr_pred * outer(domain_prediction_error, domain_activities_current).
2. THE PredictiveBCMRule SHALL use a separate learning rate (lr_pred) for prediction weight updates, configurable independently from the main learning rate.
3. WHEN learn_predictions is set to False, THE PredictiveBCMRule SHALL keep prediction weights fixed (no updates).
4. THE PredictiveBCMRule SHALL clip prediction weight updates by Frobenius norm to prevent explosion.
5. FOR ALL valid domain activity pairs presented repeatedly, THE prediction weight update rule SHALL reduce domain prediction error magnitude monotonically (convergence property).
6. BECAUSE prediction weights are small (8×8), THE prediction learning should converge quickly — within a few epochs for stable input statistics.

### Requirement 6: Domain-Level Surprise Modulation

**User Story:** As a researcher, I want domains with high prediction error (high surprise) to have amplified learning rates and domains with low prediction error (good predictions) to have reduced learning rates, so that computational resources focus on where the model is most wrong.

#### Acceptance Criteria

1. WHEN domain_prediction_error is computed, THE PredictiveBCMRule SHALL compute domain_surprise as the absolute value of the domain information signal per domain.
2. WHEN domain_surprise for a domain exceeds the mean domain_surprise across all domains, THE PredictiveBCMRule SHALL amplify the learning rate for neurons in that domain by a factor of (1 + surprise_boost × normalized_surprise).
3. WHEN domain_surprise for a domain is below the mean, THE PredictiveBCMRule SHALL reduce the learning rate for neurons in that domain proportionally.
4. THE PredictiveBCMRule SHALL support disabling domain-level surprise modulation via a use_domain_modulation flag.
5. THE surprise modulation SHALL be bounded to prevent any single domain from dominating learning (max amplification factor configurable, default 3.0).

### Requirement 7: D-Serine Gating Driven by Prediction Error

**User Story:** As a researcher, I want D-serine gating from calcium dynamics to be driven by domain prediction error (surprise) rather than raw activity, so that domains with high surprise actively learn while domains with low surprise consolidate.

#### Acceptance Criteria

1. WHEN the D-serine gate is open for a domain, THE PredictiveBCMRule SHALL amplify synapse_calcium for neurons in that domain by factor (1 + d_serine_boost).
2. WHEN the D-serine gate is closed for a domain, THE PredictiveBCMRule SHALL leave synapse_calcium unchanged for neurons in that domain.
3. THE PredictiveBCMRule SHALL step CalciumDynamics with domain_surprise (absolute domain prediction error) rather than raw domain activity, so that gate opening is driven by prediction failure rather than mere activity.
4. WHEN use_d_serine is set to False, THE PredictiveBCMRule SHALL skip all D-serine amplification.
5. THIS design means domains that are "surprised" (high prediction error) will have their calcium driven up → gate opens → D-serine released → LTP enabled. Domains that predict well → low calcium → gate closed → biased toward LTD/consolidation.

### Requirement 8: Heterosynaptic Competition

**User Story:** As a researcher, I want heterosynaptic competition within domains to provide local credit assignment, so that within each domain the neurons most relevant to reducing prediction error get LTP.

#### Acceptance Criteria

1. WHEN competition_strength is 1.0, THE PredictiveBCMRule SHALL zero-center the combined signal within each domain so that the mean per domain is approximately zero.
2. WHEN competition is applied, THE PredictiveBCMRule SHALL preserve the relative ordering of neurons within each domain.
3. WHEN use_competition is set to False, THE PredictiveBCMRule SHALL skip all heterosynaptic competition.

### Requirement 9: Weight Update Computation

**User Story:** As a researcher, I want the final weight update to combine all signals into a single delta, so that the training loop can apply it directly to layer weights.

#### Acceptance Criteria

1. THE PredictiveBCMRule SHALL compute weight delta as: delta_W = lr * combined_signal * outer(post_direction, mean_pre), where combined_signal incorporates BCM direction, prediction error information, and competition.
2. THE PredictiveBCMRule SHALL return weight deltas with shape exactly matching (out_features, in_features) of the layer weights.
3. THE PredictiveBCMRule SHALL clip weight delta Frobenius norm to clip_delta.
4. IF any intermediate computation produces NaN, THEN THE PredictiveBCMRule SHALL replace NaN with zero and continue operation.

### Requirement 10: Protocol Compliance

**User Story:** As a researcher, I want PredictiveBCMRule to implement the LocalLearningRule protocol, so that it integrates with the existing training infrastructure from Step 12.

#### Acceptance Criteria

1. THE PredictiveBCMRule SHALL implement compute_update accepting a LayerState and returning a weight delta tensor.
2. THE PredictiveBCMRule SHALL implement reset() that clears all internal state (theta, prediction weights, calcium dynamics).
3. THE PredictiveBCMRule SHALL expose a name attribute equal to "predictive_bcm".
4. THE PredictiveBCMRule SHALL accept a sequence of LayerStates (all layers) in compute_update so that prediction errors can reference adjacent layers.

### Requirement 11: Configuration

**User Story:** As a researcher, I want all hyperparameters configurable via a dataclass, so that I can easily run ablations and hyperparameter sweeps.

#### Acceptance Criteria

1. THE PredictiveBCMRule SHALL accept a PredictiveBCMConfig dataclass containing: lr, lr_pred, theta_decay, theta_init, d_serine_boost, competition_strength, clip_delta, clip_pred_delta, combination_mode, use_d_serine, use_competition, use_domain_modulation, learn_predictions.
2. WHEN no config is provided, THE PredictiveBCMRule SHALL use sensible defaults (lr=0.01, lr_pred=0.1, theta_decay=0.99, theta_init=0.1, d_serine_boost=1.0, competition_strength=1.0, clip_delta=1.0, clip_pred_delta=0.5, combination_mode="multiplicative", use_d_serine=True, use_competition=True, use_domain_modulation=True, learn_predictions=True).
3. THE PredictiveBCMConfig SHALL be a frozen dataclass to prevent accidental mutation during training.

### Requirement 12: Training Loop Integration

**User Story:** As a researcher, I want a training loop that applies PredictiveBCMRule using all layer states simultaneously, so that prediction errors between adjacent layers can be computed.

#### Acceptance Criteria

1. WHEN train_epoch is called, THE Training_Loop SHALL perform forward_with_states to collect all layer states, then pass all states to PredictiveBCMRule for update computation.
2. THE Training_Loop SHALL compute cross-entropy loss for monitoring without using it for learning.
3. THE Training_Loop SHALL return mean training loss and per-layer prediction error magnitudes for the epoch.
4. THE Training_Loop SHALL apply weight deltas to model weights and prediction weight deltas to prediction weights within the same batch step.

### Requirement 13: Experiment Comparison

**User Story:** As a researcher, I want to compare predictive coding variants against baselines, so that I can quantify the contribution of domain-level prediction errors and their interaction with BCM direction.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL evaluate 6 conditions: predictive_bcm_full (BCM + domain prediction error + D-serine + competition), predictive_bcm_no_astrocyte (BCM + domain prediction error only), predictive_only (domain prediction error without BCM direction), bcm_only (BCM without prediction error, Step 12b baseline), predictive_neuron_level (optional: neuron-level prediction for comparison), and backprop.
2. THE Experiment_Runner SHALL run each condition with 3 random seeds (42, 123, 456) for statistical reliability.
3. THE Experiment_Runner SHALL train for 50 epochs with batch_size=128 on FashionMNIST.
4. THE Experiment_Runner SHALL record per-epoch test accuracy, training loss, and mean domain prediction error per layer.
5. WHEN the full experiment completes, THE Experiment_Runner SHALL save results as JSON with timestamps and produce a summary comparing all conditions.
6. THE Experiment_Runner SHALL report whether domain-level prediction (8-dimensional) performs comparably to neuron-level prediction (128-dimensional), validating the domain-as-entity hypothesis.

### Requirement 14: Success Criteria Validation

**User Story:** As a researcher, I want clear success criteria evaluated automatically, so that I can determine whether prediction errors provide meaningful task-relevant information.

#### Acceptance Criteria

1. WHEN the experiment completes, THE Experiment_Runner SHALL report whether any predictive coding condition achieves accuracy above 10% (above chance).
2. WHEN the experiment completes, THE Experiment_Runner SHALL report whether the combination (predictive_bcm_full) outperforms both predictive_only and bcm_only individually.
3. WHEN the experiment completes, THE Experiment_Runner SHALL report whether prediction errors decrease over training (indicating predictions improve).
4. THE Experiment_Runner SHALL compare results against the forward-forward baseline of 16.5% from Step 12.

### Requirement 15: Computational Constraints

**User Story:** As a researcher, I want the experiment to complete in reasonable time on CPU, so that iteration speed is maintained.

#### Acceptance Criteria

1. THE Training_Loop SHALL complete 50 epochs of FashionMNIST training (all conditions, all seeds) in less than 4 hours on CPU.
2. THE PredictiveBCMRule with domain-level prediction SHALL add no more than 20% computational overhead per batch compared to BCMDirectedRule alone (due to the small 8×8 prediction matrices).
3. THE PredictiveBCMRule SHALL not require GPU for the 784→128→128→128→128→10 architecture.
4. THE domain-level prediction (8×8 matrices) SHALL be at least 10× faster than neuron-level prediction (128×128 matrices) per prediction step.

### Requirement 16: Prediction Error Correctness Properties

**User Story:** As a researcher, I want property-based tests verifying the mathematical correctness of prediction error computation, so that I can trust the mechanism is implemented correctly.

#### Acceptance Criteria

1. FOR ALL valid layer output pairs (layer_i, layer_{i+1}), THE PredictiveBCMRule SHALL produce prediction_error = actual - predicted with correct sign (positive when actual > predicted, negative when actual < predicted).
2. FOR ALL prediction weight matrices P and layer outputs x, THE PredictiveBCMRule SHALL satisfy: information_signal = P^T @ (actual_next - P @ x).
3. FOR ALL inputs where actual_next equals P @ x exactly, THE PredictiveBCMRule SHALL produce zero prediction error and zero information signal.
4. FOR ALL valid inputs, THE PredictiveBCMRule SHALL produce combined updates that contain both positive and negative values (signed updates preserved from BCM).

### Requirement 17: Prediction Weight Convergence Property

**User Story:** As a researcher, I want to verify that prediction weights converge toward accurate predictions when presented with consistent input-output pairs, so that the prediction learning rule is correct.

#### Acceptance Criteria

1. FOR ALL fixed input-output pairs (x, y) presented repeatedly, THE prediction weight update rule (delta_P = lr_pred * outer(error, x)) SHALL reduce prediction error magnitude monotonically (within numerical tolerance).
2. FOR ALL fixed input-output pairs, THE prediction weights SHALL converge such that P @ x approximates y after sufficient iterations.
3. THE PredictiveBCMRule SHALL produce prediction errors whose magnitude decreases over training epochs on FashionMNIST (measured as mean absolute prediction error per layer).

### Requirement 18: Ablation Independence

**User Story:** As a researcher, I want each component independently disableable, so that I can isolate the contribution of prediction errors, BCM direction, D-serine, and competition.

#### Acceptance Criteria

1. WHEN prediction error is disabled (predictive_only=False equivalent via bcm_only condition), THE PredictiveBCMRule SHALL produce updates identical to BCMDirectedRule from Step 12b.
2. WHEN BCM direction is disabled (predictive_only condition), THE PredictiveBCMRule SHALL use prediction error information as the sole direction signal.
3. WHEN use_d_serine is False and use_competition is False and use_domain_modulation is False, THE PredictiveBCMRule SHALL produce updates depending only on BCM direction, prediction error, and the outer product structure.
4. WHEN learn_predictions is False, THE PredictiveBCMRule SHALL keep prediction weights fixed throughout training.

