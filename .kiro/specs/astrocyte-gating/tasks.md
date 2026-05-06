# Implementation Plan: Astrocyte D-Serine Gating (Step 13)

## Overview

This plan implements Step 13 of the glia-augmented neural network research plan: three variants of an astrocyte-derived gating signal as a drop-in replacement for Step 12's ThirdFactorInterface. The implementation is ordered by dependency — project setup and stability fix first, then calcium dynamics and domain assignment, then the three gate variants, then integration with ThreeFactorRule, then experiments (performance comparison, central prediction test, ablations).

Python + PyTorch, targeting MPS GPU on M4 Pro. Reuses Step 12's ThreeFactorRule, LocalMLP, data pipeline, and experiment runner from `steps/12-local-learning-rules/code/`. Reuses Step 01's spatial embedding from `steps/01-spatial-embedding/code/`. Expected runtime for full comparison: 3–6 hours (50 epochs × 3 seeds × 6+ conditions).

## Tasks

- [ ] 1. Set up project structure and imports from Step 12
  - [ ] 1.1 Create directory structure and module scaffolding
    - Create `steps/13-astrocyte-gating/` with subdirectories: `docs/`, `code/`, `code/gates/`, `code/calcium/`, `code/domains/`, `code/experiment/`, `code/tests/`, `data/`, `results/`
    - Create `__init__.py` files for all Python packages
    - Create `steps/13-astrocyte-gating/README.md` summarizing the step's purpose, how to run experiments, and how to interpret results
    - Create `steps/13-astrocyte-gating/docs/decisions.md` for recording design decisions
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ] 1.2 Set up imports and verify Step 12 reuse
    - Create `code/imports.py` or equivalent module that imports ThreeFactorRule, ThirdFactorInterface, LocalMLP, data pipeline, and experiment runner from `steps/12-local-learning-rules/code/`
    - Import spatial embedding utilities from `steps/01-spatial-embedding/code/`
    - Verify imports work by running a quick smoke test (instantiate LocalMLP, load FashionMNIST)
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 2. Implement three-factor error signal stability fix
  - [ ] 2.1 Implement error clipping and eligibility normalization in `code/stability.py`
    - Implement `clip_error_signal(error, threshold=10.0)` that clips magnitude while preserving sign
    - Implement `normalize_eligibility(trace, norm_threshold=100.0, safe_constant=1.0)` that normalizes trace to unit norm × safe_constant when Frobenius norm exceeds threshold
    - These functions will be applied within the ThreeFactorRule integration (wrapping Step 12's rule)
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ]* 2.2 Write property tests for stability fix (Properties 1, 2)
    - **Property 1: Sign-Preserving Error Clipping** — for any error tensor, clipped output has no element exceeding threshold in absolute value, and sign of every non-zero element is preserved
    - **Property 2: Eligibility Trace Norm Bounding** — for any trace exceeding norm threshold, normalized output has norm equal to safe_constant, and direction (unit vector) is preserved
    - **Validates: Requirements 2.1, 2.2, 2.4**

  - [ ] 2.3 Verify stability fix enables 50-epoch training without NaN/Inf
    - Run ThreeFactorRule with layer-wise error + stability fix for 50 epochs on FashionMNIST (single seed)
    - Assert no NaN or Inf in any weight tensor at any epoch
    - _Requirements: 2.3_

- [ ] 3. Implement Li-Rinzel calcium dynamics
  - [ ] 3.1 Implement `CalciumDynamics` class in `code/calcium/li_rinzel.py`
    - Implement the Li-Rinzel two-variable system: cytoplasmic [Ca²⁺] and IP3 receptor inactivation h
    - Implement fluxes: J_channel (IP3-dependent ER release), J_pump (SERCA reuptake), J_leak (passive ER leak)
    - IP3 production proportional to domain activity (glutamate spillover analog)
    - Vectorized over N domains using PyTorch tensors
    - Clamp calcium to [0, ca_max] and h to [0, 1] after each step
    - Implement `step()`, `get_calcium()`, `get_gate_open()`, `reset()`, `state_dict()`, `load_state_dict()`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ] 3.2 Implement `CalciumConfig` dataclass in `code/calcium/config.py`
    - All Li-Rinzel parameters with defaults from design document
    - Include c0, c1, a2, d2, K_pump constants
    - _Requirements: 3.4_

  - [ ]* 3.3 Write property tests for calcium dynamics (Properties 3, 4)
    - **Property 3: Calcium Concentration Invariant** — for any sequence of domain activities (including extremes), calcium stays in [0, ca_max] and h stays in [0, 1] after each step
    - **Property 4: IP3 Proportionality** — for any two activities a₁ > a₂ ≥ 0, IP3 production for a₁ ≥ IP3 production for a₂
    - **Validates: Requirements 3.3, 3.6**

- [ ] 4. Implement astrocyte domain assignment
  - [ ] 4.1 Implement `DomainAssignment` class in `code/domains/assignment.py`
    - Partition output neurons per layer into non-overlapping domains of configurable size (default 16)
    - "spatial" mode: k-means clustering on Step 01 3D coordinates for spatially coherent domains
    - "random" mode: random assignment for ablation
    - Fallback: contiguous index partitioning if embedding not available (with logged warning)
    - Implement `get_domain_indices()`, `get_domain_distances()`, `get_neuron_to_domain()`
    - Compute once at init, immutable thereafter
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 4.2 Implement `DomainConfig` dataclass in `code/domains/config.py`
    - domain_size, mode, embedding_path, seed fields with defaults from design
    - _Requirements: 4.2, 4.3_

  - [ ]* 4.3 Write property tests for domain assignment (Properties 5, 6)
    - **Property 5: Domain Partition Validity** — for any layer size and domain_size config, produces exactly ceil(out_features / domain_size) domains where every neuron belongs to exactly one domain (no overlaps, no unassigned)
    - **Property 6: Domain Assignment Immutability** — calling get_domain_indices multiple times returns identical results
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [ ] 5. Checkpoint — Verify calcium dynamics and domain assignment
  - Ensure all tests pass, ask the user if questions arise.
  - Verify CalciumDynamics produces bounded calcium with sustained input
  - Verify DomainAssignment partitions 128-unit layers into 8 domains correctly
  - Verify spatial embedding loads from Step 01

- [ ] 6. Implement Binary Gate (Variant A)
  - [ ] 6.1 Implement `BinaryGate` class in `code/gates/binary_gate.py`
    - Implement ThirdFactorInterface protocol (compute_signal, reset)
    - Output 1.0 for neurons in domains where Ca > threshold, 0.0 otherwise
    - Update calcium dynamics each step based on domain mean absolute activation
    - Same gate value for all neurons within a domain
    - Output shape: (out_features,)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 6.2 Write property tests for binary gate (Properties 7, 8)
    - **Property 7: Binary Gate Threshold Semantics** — output is exactly 1.0 where Ca > threshold and 0.0 where Ca ≤ threshold; all neurons in same domain get same value
    - **Property 8: Binary Gate Blocks Closed-Domain Updates** — weight update is exactly zero for weights in closed domains when combined with ThreeFactorRule
    - **Validates: Requirements 5.2, 5.4, 5.5**

- [ ] 7. Implement Directional Gate (Variant B)
  - [ ] 7.1 Implement `DirectionalGate` class in `code/gates/directional_gate.py`
    - Implement ThirdFactorInterface protocol (compute_signal, reset)
    - Maintain EMA prediction of domain activity (configurable decay, default 0.95)
    - Compute activity error = current − predicted per domain
    - Normalize error per domain (zero-mean, unit-variance)
    - Output = calcium_state × normalized_error; zero where Ca ≤ threshold
    - Output shape: (out_features,) — signed value per neuron
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 7.2 Write property tests for directional gate (Properties 9, 10, 11)
    - **Property 9: Directional Gate EMA Dynamics** — activity prediction follows EMA formula with configured decay rate
    - **Property 10: Directional Gate Output Formula** — output = c × normalize(a − p) when c > threshold, 0.0 otherwise; sign matches sign of (a − p)
    - **Property 11: Directional Gate Error Normalization** — after normalization, max absolute error is bounded
    - **Validates: Requirements 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**

- [ ] 8. Implement Volume Teaching Gate (Variant C)
  - [ ] 8.1 Implement `VolumeTeachingGate` class in `code/gates/volume_teaching.py`
    - Implement ThirdFactorInterface protocol (compute_signal, reset)
    - Compute domain-local error from activity vs label-derived target (random projection of one-hot)
    - Build Gaussian diffusion kernel from inter-domain distances (sigma = mean inter-domain distance)
    - Diffuse error: received_signal = kernel @ source_errors
    - Apply gap junction calcium coupling between adjacent domains
    - Gate: signal × (Ca > threshold); zero where calcium below threshold
    - Expand domain signal to per-neuron output shape (out_features,)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ]* 8.2 Write property tests for volume teaching gate (Properties 12, 13, 14)
    - **Property 12: Volume Teaching Gaussian Diffusion** — received signal equals sum of source errors weighted by exp(-d²/2σ²); closer domains receive stronger signal
    - **Property 13: Volume Teaching Calcium Gating** — output is zero for neurons in domains where Ca < threshold regardless of diffused signal magnitude
    - **Property 14: Gap Junction Calcium Equilibration** — after coupling, calcium difference between adjacent domains is reduced; total calcium is conserved
    - **Validates: Requirements 7.3, 7.4, 7.5, 7.7**

- [ ] 9. Checkpoint — Verify all gate variants
  - Ensure all tests pass, ask the user if questions arise.
  - Verify each gate variant produces output of correct shape (out_features,)
  - Verify BinaryGate output is exactly 0.0 or 1.0
  - Verify DirectionalGate output is signed
  - Verify VolumeTeachingGate diffusion attenuates with distance

- [ ] 10. Integrate gates with ThreeFactorRule and implement GateConfig
  - [ ] 10.1 Implement `GateConfig` and `ExperimentCondition` dataclasses in `code/experiment/config.py`
    - GateConfig: variant, prediction_decay, diffusion_sigma, gap_junction_strength, n_classes
    - ExperimentCondition: name, gate_config, calcium_config, domain_config, learning_rate, tau, use_stability_fix, error_clip_threshold, eligibility_norm_threshold
    - _Requirements: 8.1, 8.3_

  - [ ] 10.2 Implement gate-ThreeFactorRule integration in `code/experiment/training.py`
    - Create training loop that wires gate variants into ThreeFactorRule as third_factor
    - Apply stability fix (error clipping + eligibility normalization) within the training step
    - Include astrocyte state checkpointing (calcium, h, activity predictions) every 10 epochs
    - Verify no special-case code — gates used identically to Step 12 placeholder implementations
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 13.6_

  - [ ]* 10.3 Write property test for interface compatibility (Property 15)
    - **Property 15: Gate Output Shape Compatibility** — for any gate variant and valid layer activation (batch, out_features), compute_signal output has shape (out_features,) compatible with ThreeFactorRule
    - **Validates: Requirements 8.2**

  - [ ] 10.4 Verify 50-epoch training with each gate variant completes without NaN/Inf
    - Run a quick 5-epoch smoke test with each gate variant + ThreeFactorRule
    - Assert no NaN or Inf in weights, calcium, or gate signals
    - _Requirements: 8.5_

- [ ] 11. Checkpoint — Verify integration
  - Ensure all tests pass, ask the user if questions arise.
  - Verify all three gate variants train for 5 epochs without numerical issues
  - Verify checkpoint save/load round-trip preserves calcium and prediction state
  - Verify gate variants produce different training dynamics (not all identical)

- [ ] 12. Implement experiment runner with timestamps
  - [ ] 12.1 Implement `ExperimentRunner` in `code/experiment/runner.py`
    - Reuse/adapt Step 12's experiment runner infrastructure
    - Support running individual conditions or all conditions via CLI arguments
    - Print UTC datetime before and after each condition run (timestamp between runs)
    - Log all hyperparameters, seeds, library versions, hardware info to JSON metadata file
    - Include UTC timestamp in output filenames (format: YYYYMMDD_HHMMSS)
    - Checkpoint model + astrocyte state every 10 epochs to `data/`
    - Seed management: set Python, NumPy, PyTorch seeds per condition/seed combo
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

  - [ ] 12.2 Implement metrics collection in `code/experiment/metrics.py`
    - Record per-epoch: test accuracy, train loss, weight norms, gate_fraction_open, gate_temporal_autocorr, spatial_correlation
    - Store results in CSV format in `results/` with timestamps in filename
    - Implement `EpochResult` and `ConditionResult` dataclasses from design
    - _Requirements: 9.4, 9.5_

  - [ ] 12.3 Implement CLI entry point in `code/experiment/run_experiments.py`
    - Parse command-line arguments for condition selection, seed override, epoch count
    - Support `--condition` flag for individual conditions
    - Support `--all` flag for full comparison
    - Print datetime.utcnow() before and after each condition
    - _Requirements: 13.5_

- [ ] 13. Implement performance comparison experiment
  - [ ] 13.1 Define all experimental conditions in `code/experiment/conditions.py`
    - (a) Three-factor with random gate (Step 12 baseline)
    - (b) Three-factor with global reward (Step 12 baseline)
    - (c) Three-factor with binary astrocyte gate (Variant A)
    - (d) Three-factor with directional astrocyte gate (Variant B)
    - (e) Three-factor with volume teaching signal (Variant C)
    - (f) Backpropagation baseline
    - All using identical architecture (784→128→128→128→128→10), dataset (FashionMNIST), batch size (128), 50 epochs, 3 seeds
    - _Requirements: 9.1, 9.2_

  - [ ] 13.2 Implement comparison visualization in `code/experiment/comparison.py`
    - Generate summary comparison table (CSV) with mean/std accuracy per condition
    - Generate accuracy bar chart (`results/accuracy_comparison.png`)
    - Generate convergence curves plot (`results/convergence_curves.png`)
    - _Requirements: 9.6_

- [ ] 14. Implement central prediction test
  - [ ] 14.1 Implement `compute_central_prediction()` in `code/experiment/central_prediction.py`
    - Compute benefit_under_local_rules = best_gated_accuracy − ungated_baseline_accuracy
    - Compute benefit_under_backprop = 0.14 (Phase 1 measured value)
    - Compare magnitudes and compute confidence interval from seed variance
    - Generate bar chart with error bars (`results/central_prediction_test.png`)
    - State conclusion: "hypothesis confirmed", "hypothesis refuted", or "inconclusive"
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 15. Implement calcium dynamics ablation experiment
  - [ ] 15.1 Implement calcium ablation conditions in `code/experiment/ablation_calcium.py`
    - (a) Full Li-Rinzel calcium dynamics (directional gate)
    - (b) Simple threshold gate (gate = 1 if activity > threshold, else 0)
    - (c) Linear EMA filter (gate = EMA of activity)
    - (d) Random gate with matched sparsity
    - Same experimental conditions across all four mechanisms
    - Report accuracy, convergence speed, gate statistics (fraction open, temporal autocorrelation)
    - Generate comparison table and visualization in `results/`
    - Print timestamps before/after each ablation condition
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 16. Implement spatial domain structure ablation experiment
  - [ ] 16.1 Implement spatial ablation conditions in `code/experiment/ablation_spatial.py`
    - (a) Spatial assignment based on Step 01 embedding proximity (directional gate)
    - (b) Random assignment with same domain sizes (directional gate)
    - Same experimental conditions across both strategies
    - Report accuracy, convergence speed, spatial correlation of weight updates
    - Generate comparison table and visualization in `results/`
    - Print timestamps before/after each ablation condition
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 17. Checkpoint — Verify experiment scripts
  - Ensure all tests pass, ask the user if questions arise.
  - Run a quick 2-epoch smoke test of the full experiment runner with one condition
  - Verify CSV output has correct schema
  - Verify timestamps are printed between conditions
  - Verify metadata JSON is written correctly

- [ ] 18. Run full experiments and generate results
  - [ ] 18.1 Run performance comparison (50 epochs, 3 seeds, all 6 conditions)
    - Execute all conditions with timestamps between runs
    - Save results to `results/` with timestamped filenames
    - Generate accuracy_comparison.png and convergence_curves.png
    - _Requirements: 9.1, 9.2, 9.3, 9.5_

  - [ ] 18.2 Run central prediction test
    - Compute benefit_under_local_rules vs benefit_under_backprop
    - Generate central_prediction_test.png
    - Write conclusion to results summary
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ] 18.3 Run calcium dynamics ablation (50 epochs, 3 seeds, 4 mechanisms)
    - Execute with timestamps between conditions
    - Generate ablation comparison table and visualization
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ] 18.4 Run spatial domain structure ablation (50 epochs, 3 seeds, 2 strategies)
    - Execute with timestamps between conditions
    - Generate ablation comparison table and visualization
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [ ] 18.5 Generate summary report
    - Create `results/summary.md` documenting key findings, central prediction result, ablation insights, and implications for subsequent steps
    - Include all accuracy numbers, confidence intervals, and conclusions
    - _Requirements: 13.4_

- [ ] 19. Final checkpoint — Ensure all tests pass and results are complete
  - Ensure all tests pass, ask the user if questions arise.
  - Verify all result files exist in `results/` directory
  - Verify summary.md documents central prediction test outcome
  - Verify timestamps appear in experiment logs
  - Verify no NaN/Inf in any training run across all conditions and seeds

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones
- Property tests validate the 15 correctness properties from the design document
- All experiment scripts MUST print `datetime.utcnow()` before and after each condition run
- Expected runtime: 3–6 hours for the full comparison (tasks 18.1–18.4)
- The stability fix (task 2) is critical — Step 12 showed loss explosion without it
- Variant C (Volume Teaching) is the most complex and most likely to approach backprop performance
- The central prediction test (task 14) is the scientific payoff of the entire step
