# Implementation Plan: Predictive Coding + BCM (Step 14)

## Overview

This plan implements a local learning rule combining BCM-directed signed updates with inter-layer domain-level prediction errors. The core algorithm maintains small (8×8) prediction matrices between adjacent layers; prediction errors modulate BCM direction to produce task-relevant weight updates. Implementation builds on Steps 12, 12b, and 13 infrastructure (LocalMLP, DomainAssignment, CalciumDynamics).

## Tasks

- [x] 1. Set up project structure, imports, and configuration
  - [x] 1.1 Create directory structure and __init__.py files
    - Create `steps/14-predictive-coding-bcm/` with subdirectories: `code/`, `code/scripts/`, `code/tests/`, `data/`, `results/`, `docs/`
    - Add `__init__.py` to `code/`, `code/scripts/`, `code/tests/`
    - Create `steps/14-predictive-coding-bcm/README.md` with step overview
    - _Requirements: 11, 15_

  - [x] 1.2 Create step_imports.py bridge module
    - Import from Step 12: `LocalMLP`, `LayerState`, `LocalLearningRule`, `get_fashion_mnist_loaders`, `ThreeFactorRule`, `GlobalRewardThirdFactor`
    - Import from Step 13: `CalciumDynamics`, `CalciumConfig`, `DomainAssignment`, `DomainConfig`
    - Import from Step 12b: `BCMDirectedRule`, `BCMConfig`
    - Follow the same `_step_context` pattern used in `steps/12b-bcm-directed/code/step_imports.py`
    - _Requirements: 10, 12_

  - [x] 1.3 Create PredictiveBCMConfig frozen dataclass
    - Implement `predictive_bcm_config.py` with frozen dataclass containing: `lr`, `lr_pred`, `theta_decay`, `theta_init`, `d_serine_boost`, `competition_strength`, `clip_delta`, `clip_pred_delta`, `combination_mode`, `use_d_serine`, `use_competition`, `use_domain_modulation`, `learn_predictions`, `max_surprise_amplification`, `granularity`, `fixed_predictions`
    - Use defaults: lr=0.01, lr_pred=0.1, theta_decay=0.99, theta_init=0.1, d_serine_boost=1.0, competition_strength=1.0, clip_delta=1.0, clip_pred_delta=0.5, combination_mode="multiplicative", use_d_serine=True, use_competition=True, use_domain_modulation=True, learn_predictions=True, max_surprise_amplification=3.0, granularity="domain", fixed_predictions=False
    - _Requirements: 11.1, 11.2, 11.3_

- [x] 2. Implement PredictiveBCMRule core algorithm
  - [x] 2.1 Implement PredictiveBCMRule __init__ and prediction weight initialization
    - Initialize prediction weights P_i of shape (n_domains_next, n_domains_current) for each layer i except the last, using Xavier uniform initialization
    - Support "neuron" granularity mode with shape (next_layer_size, current_layer_size)
    - Support fixed_predictions mode (feedback alignment style)
    - Store prediction weights as torch tensors on the configured device
    - Initialize sliding threshold theta per layer
    - Set `name = "predictive_bcm"`
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 10.3_

  - [x] 2.2 Implement domain activity computation and prediction error
    - Compute `domain_activities_current = mean(|post_i|)` per domain → shape (n_domains,)
    - Compute `domain_activities_next = mean(|post_{i+1}|)` per domain → shape (n_domains,)
    - Compute `predicted_next = P_i @ domain_activities_current` → shape (n_domains_next,)
    - Compute `domain_prediction_error = domain_activities_next - predicted_next` → shape (n_domains_next,) SIGNED
    - Handle last hidden layer: use output layer's domain activities as target
    - Support neuron-level granularity variant
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 2.3 Implement information signal computation
    - Compute `domain_information = P_i^T @ domain_prediction_error` → shape (n_domains_current,)
    - Normalize: `domain_information /= (||domain_information|| + eps)`
    - Broadcast to neurons: `info_per_neuron[j] = domain_information[domain_of_j]` → shape (out_features,)
    - Ensure zero prediction error produces zero information signal
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 2.4 Implement BCM direction with surprise-driven calcium
    - Compute `synapse_calcium = mean(|post_i|)` per neuron (batch-averaged) → shape (out_features,)
    - Step calcium dynamics with `|domain_prediction_error|` (surprise) instead of raw activity
    - Apply D-serine boost when gate is open (if use_d_serine=True)
    - Update theta: EMA of domain_activities_current
    - Compute `bcm_direction = synapse_calcium - neuron_theta` → shape (out_features,) SIGNED
    - _Requirements: 4.1, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 2.5 Implement combination, competition, surprise modulation, and weight delta
    - Compute `combined = bcm_direction * info_per_neuron` (multiplicative default)
    - Support additive and threshold combination modes
    - Apply heterosynaptic competition: zero-center combined within domains (if use_competition=True)
    - Apply surprise modulation: amplify learning in surprised domains, reduce in unsurprised (if use_domain_modulation=True), bounded by max_surprise_amplification
    - Compute `delta_W = lr * outer(combined, mean_pre)` → shape (out_features, in_features)
    - Clip delta_W Frobenius norm to clip_delta
    - Replace NaN with zero in all intermediate computations
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 6.1, 6.2, 6.3, 6.4, 6.5, 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 9.4_

  - [x] 2.6 Implement prediction weight learning
    - Compute `delta_P = lr_pred * outer(domain_prediction_error, domain_activities_current)`
    - Clip prediction weight update Frobenius norm to clip_pred_delta
    - Apply update only when learn_predictions=True and fixed_predictions=False
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 2.7 Implement compute_all_updates, compute_update shim, get_prediction_errors, and reset
    - `compute_all_updates(states: list[LayerState]) -> list[torch.Tensor]`: orchestrates the full algorithm for all layers
    - `compute_update(state: LayerState)`: raises NotImplementedError directing to compute_all_updates
    - `get_prediction_errors() -> dict[int, torch.Tensor]`: returns last-computed prediction errors per layer
    - `reset()`: reinitializes prediction weights, clears theta, resets calcium dynamics
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 1.4_

- [x] 3. Checkpoint — Ensure core rule works
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement property-based tests
  - [ ]* 4.1 Write property test: Prediction Error Sign Correctness
    - **Property 1: Prediction Error Sign Correctness**
    - For any pair of actual and predicted domain activity vectors, verify sign(error[d]) == sign(actual[d] - predicted[d]) and that error contains both positive and negative values when actual ≠ predicted
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 2.3, 2.4, 16.1**

  - [ ]* 4.2 Write property test: Information Signal Mathematical Identity
    - **Property 2: Information Signal Mathematical Identity**
    - For any P, x, y: information_signal == P^T @ (y - P @ x) within floating-point tolerance
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 3.1, 16.2**

  - [ ]* 4.3 Write property test: Zero Prediction Error Produces Zero Information Signal
    - **Property 3: Zero Prediction Error Produces Zero Information Signal**
    - When actual_next = P @ x, both prediction error and information signal are zero
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 3.5, 16.3**

  - [ ]* 4.4 Write property test: Combined Updates Are Signed
    - **Property 4: Combined Updates Are Signed**
    - For non-degenerate inputs, combined signal contains both positive and negative values
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 4.2, 16.4**

  - [ ]* 4.5 Write property test: Prediction Weight Convergence
    - **Property 5: Prediction Weight Convergence**
    - For fixed (x, y) pairs presented repeatedly, |y - P@x| decreases monotonically
    - Use Hypothesis with 100 examples
    - **Validates: Requirements 5.5, 17.1, 17.2**

  - [ ]* 4.6 Write property test: Domain Broadcast Preserves Structure
    - **Property 6: Domain Broadcast Preserves Structure**
    - All neurons in domain d receive identical information_signal[d], no cross-domain contamination
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 3.2, 3.6**

  - [ ]* 4.7 Write property test: Normalization Produces Unit Norm
    - **Property 7: Normalization Produces Unit Norm**
    - Non-zero vectors have L2 norm ≈ 1.0 after normalization; zero input → zero output
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 3.3**

  - [ ]* 4.8 Write property test: Multiplicative Combination Correctness
    - **Property 8: Multiplicative Combination Correctness**
    - Combined == element-wise product of BCM direction and information signal; sign rules hold
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 4.2, 4.3, 4.4**

  - [ ]* 4.9 Write property test: Output Shape Matches Weights
    - **Property 9: Output Shape Matches Weights**
    - Weight deltas have shape (out_features, in_features); prediction weights have shape (n_domains_next, n_domains_current)
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 1.1, 2.6, 9.2**

  - [ ]* 4.10 Write property test: Delta Norm Bounded
    - **Property 10: Delta Norm Bounded**
    - Weight delta norm ≤ clip_delta; prediction weight delta norm ≤ clip_pred_delta
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 5.4, 9.3**

  - [ ]* 4.11 Write property test: Fixed Predictions Immutability
    - **Property 11: Fixed Predictions Immutability**
    - With learn_predictions=False, prediction weights remain unchanged after processing inputs
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 1.5, 5.3, 18.4**

  - [ ]* 4.12 Write property test: Surprise Modulation Bounded and Directional
    - **Property 12: Surprise Modulation Bounded and Directional**
    - Amplification ≤ max_surprise_amplification; above-mean surprise → amplification > 1; below-mean → < 1; disabled → all 1.0
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 6.2, 6.3, 6.4, 6.5**

  - [ ]* 4.13 Write property test: Heterosynaptic Competition Zero-Centers Within Domains
    - **Property 13: Heterosynaptic Competition Zero-Centers Within Domains**
    - With competition_strength=1.0, mean within each domain ≈ 0; with use_competition=False, signal unchanged
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 8.1, 8.3**

  - [ ]* 4.14 Write property test: Domain Activity Aggregation Correctness
    - **Property 14: Domain Activity Aggregation Correctness**
    - domain_activities[d] == mean of |post_activation| over batch and neurons in domain d
    - Use Hypothesis with 200 examples
    - **Validates: Requirements 2.1**

  - [ ]* 4.15 Write property test: Ablation Independence — BCM-Only Mode
    - **Property 15: Ablation Independence — BCM-Only Mode**
    - With prediction error disabled, updates depend only on BCM direction, D-serine, and competition
    - Use Hypothesis with 100 examples
    - **Validates: Requirements 18.1, 18.3**

- [x] 5. Checkpoint — Ensure property tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement training loop
  - [x] 6.1 Implement train_epoch_predictive function
    - Forward pass with `model.forward_with_states(x)` to collect all LayerStates
    - Pass all states to `rule.compute_all_updates(states)` to get weight deltas
    - Apply weight deltas to model weights
    - Compute cross-entropy loss for monitoring (not for learning)
    - Return dict with `train_loss` and `prediction_errors` (per-layer mean absolute prediction error)
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x] 6.2 Implement evaluate function
    - Compute test accuracy and test loss on held-out data
    - Reuse pattern from Step 12b's evaluate function
    - _Requirements: 13.4, 14.1_

  - [x]* 6.3 Write integration test for training loop
    - Verify train_epoch_predictive runs without error for 1 epoch
    - Verify return format contains train_loss and prediction_errors
    - Verify prediction_errors dict has entries for layers 0-3
    - _Requirements: 12.1, 12.3_

- [x] 7. Implement experiment runner
  - [x] 7.1 Create ExperimentCondition dataclass and condition factory functions
    - Define `ExperimentCondition` dataclass with fields for name, config, flags, description
    - Implement 6 condition factories: `get_predictive_bcm_full`, `get_predictive_bcm_no_astrocyte`, `get_predictive_only`, `get_bcm_only`, `get_predictive_neuron_level`, `get_backprop`
    - Implement `get_all_conditions()` returning all 6
    - _Requirements: 13.1_

  - [x] 7.2 Implement run_condition function
    - Accept condition, n_epochs, batch_size, seed, device, verbose parameters
    - Set up model, data loaders, and appropriate rule/optimizer based on condition
    - Train for n_epochs, recording per-epoch test accuracy, train loss, and prediction errors
    - Return results dict with condition name, seed, final accuracy, epoch results
    - _Requirements: 13.2, 13.3, 13.4_

  - [x] 7.3 Implement run_experiment orchestrator
    - Run all conditions across all seeds (42, 123, 456)
    - Save results as JSON with timestamps
    - Produce summary comparing all conditions
    - _Requirements: 13.2, 13.5_

- [x] 8. Implement scripts
  - [x] 8.1 Create run_quick.py smoke test script
    - Run 5 epochs, 1 seed, predictive_bcm_full condition only
    - Print per-epoch accuracy and prediction errors
    - Verify basic functionality before full experiment
    - _Requirements: 15.1, 15.3_

  - [x] 8.2 Create run_experiment.py full experiment script
    - Run 50 epochs × 3 seeds × 6 conditions
    - Save results to `steps/14-predictive-coding-bcm/results/`
    - Print progress and timing information
    - _Requirements: 13.2, 13.3, 13.5, 15.1_

  - [x] 8.3 Create analyze_results.py analysis script
    - Load results JSON from results directory
    - Compute mean ± std accuracy across seeds for each condition
    - Evaluate success criteria: above chance, combination > parts, prediction errors decrease, domain ≈ neuron
    - Compare against forward-forward baseline (16.5%)
    - Generate summary markdown in results directory
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 13.6_

- [x] 9. Checkpoint — Ensure full pipeline works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Create documentation
  - [ ] 10.1 Create docs/decisions.md
    - Document key design decisions: domain-level vs neuron-level prediction, surprise-driven calcium, multiplicative combination
    - Document the biological grounding for each choice
    - Reference the design document's rationale sections
    - _Requirements: 13.6_

  - [ ] 10.2 Finalize README.md
    - Add usage instructions for running quick test and full experiment
    - Document the 6 experimental conditions
    - Add expected results section (to be filled after experiment)
    - _Requirements: 13.5_

- [x] 11. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- The implementation language is Python (matching all prior steps and the design document)
- All 15 property-based tests use the `hypothesis` library with 100-200 examples each
- The core algorithm operates at domain level (8-dimensional) for efficiency; neuron-level (128-dimensional) is a comparison condition only
