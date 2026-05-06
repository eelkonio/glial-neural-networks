# Requirements Document

## Introduction

This document specifies the requirements for implementing Step 13 (Astrocyte D-Serine Gating) of the glia-augmented neural network research plan. Step 13 is the pivotal experiment of Phase 2, testing whether astrocyte-derived gating signals can transform weak local learning rules into competitive algorithms by providing spatially-local directional credit assignment.

Step 12 established that local learning rules without glia achieve only 10–16.5% accuracy on FashionMNIST (vs backprop's 89%), with a credit assignment gap of 72%. The three-factor rule with global reward shows the strongest spatial correlation (−0.364, 104× stronger than backprop's −0.003) and best inter-layer coordination (CKA 0.85), confirming that spatial structure matters under local rules. However, Step 12 also revealed that a simple activity-threshold gate is insufficient — the astrocyte must provide DIRECTIONAL credit assignment (a teaching signal that carries error information), not just binary plasticity gating.

This step implements three variants of the astrocyte gate as a ThirdFactorInterface drop-in replacement: (A) binary gate based on calcium threshold, (B) directional gate that carries activity-error information, and (C) volume-transmitted teaching signal diffused through spatial fields. The implementation includes Li-Rinzel calcium dynamics, spatial domain structure from Step 01's embedding, and ablation studies to isolate which biological features contribute to performance.

The central prediction being tested: the benefit of astrocyte gating under local rules should EXCEED the benefit of astrocyte modulation under backpropagation (Phase 1's +0.14%), confirming that glia are constitutive components of local learning rather than mere modulators.

## References

- Li, Y.-X. & Rinzel, J. (1994). "Equations for InsP3 Receptor-mediated [Ca2+]i Oscillations Derived from a Detailed Kinetic Model." *Journal of Theoretical Biology*, 166(4), 461–473. — Li-Rinzel model for IP3-dependent calcium dynamics in astrocytes.
- Henneberger, C., Papouin, T., Bhatt, D.K., Bhatt, D.K., & Bhatt, D.K. (2010). "Long-term potentiation depends on release of D-serine from astrocytes." *Nature*, 463(7278), 232–236. — Experimental evidence that astrocytic D-serine is required for NMDA-dependent LTP.
- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018). "Eligibility Traces and Plasticity on Behavioral Time Scales." *Frontiers in Neural Circuits*, 12, 53. — Three-factor learning rules with eligibility traces.
- De Pittà, M., Goldberg, M., Bhatt, D.K., & Bhatt, D.K. (2009). "Glutamate regulation of calcium and IP3 oscillating and pulsating dynamics in astrocytes." *Journal of Biological Physics*, 35(4), 383–411. — Computational models of astrocyte calcium signaling.
- Araque, A., Carmignoto, G., Bhatt, D.K., & Bhatt, D.K. (2014). "Gliotransmitters Travel in Time and Space." *Neuron*, 81(4), 728–739. — Volume transmission and spatial diffusion of gliotransmitters.

## Glossary

- **Astrocyte_Plasticity_Gate**: A module that implements the ThirdFactorInterface protocol, computing a gating/teaching signal for the three-factor learning rule based on astrocyte calcium dynamics and domain activity patterns
- **Calcium_Dynamics**: The Li-Rinzel model of intracellular calcium in astrocytes, including IP3-dependent release from the endoplasmic reticulum (ER), calcium-induced calcium release (CICR), SERCA pump reuptake, and leak currents
- **D_Serine_Release**: The threshold-dependent release of D-serine by astrocytes when intracellular calcium exceeds a threshold, which in biology gates NMDA-dependent synaptic plasticity
- **Astrocyte_Domain**: A spatial territory of weights governed by a single astrocyte, derived from the spatial embedding computed in Step 01; all weights within a domain receive the same gating signal
- **Binary_Gate**: Variant A of the astrocyte gate where the signal is 0 or 1 based on whether calcium exceeds a threshold — gates whether plasticity occurs but provides no directional information
- **Directional_Gate**: Variant B of the astrocyte gate where the signal carries both magnitude and sign, computed as the product of calcium state and an activity-error signal — provides directional credit assignment
- **Volume_Teaching_Signal**: Variant C of the astrocyte gate where an error signal is broadcast and diffused through the spatial field, attenuating with distance from the source — provides spatially-graded directional teaching
- **Activity_Error_Signal**: The difference between a domain's current activity pattern and its predicted (expected) activity pattern, where the prediction is learned by the astrocyte via exponential moving average — signals novelty or mismatch
- **Teaching_Signal**: A signal that carries directional information about which way weights should change (positive or negative), as opposed to a binary gate that only determines whether change occurs
- **ThirdFactorInterface**: The pluggable protocol defined in Step 12 (compute_signal method accepting layer activations, layer index, labels, and loss) that the astrocyte gate must implement as a drop-in replacement
- **Three_Factor_Rule**: The learning rule from Step 12 where weight change equals eligibility_trace × third_factor_signal × learning_rate; the astrocyte gate replaces the third_factor_signal
- **Eligibility_Trace**: A decaying memory at each synapse recording recent pre/post co-activation, which is converted to a weight change only when modulated by the third factor (astrocyte gate)
- **Li_Rinzel_Model**: A reduced two-variable model of IP3-receptor-mediated calcium oscillations, with variables for cytoplasmic calcium concentration and the fraction of IP3 receptors not inactivated by calcium
- **IP3**: Inositol trisphosphate, a second messenger produced when glutamate binds to metabotropic receptors on the astrocyte; triggers calcium release from the ER
- **CICR**: Calcium-Induced Calcium Release — positive feedback where cytoplasmic calcium promotes further calcium release from the ER through IP3 receptors
- **SERCA_Pump**: Sarco/Endoplasmic Reticulum Ca2+-ATPase — actively pumps calcium back into the ER, providing the recovery mechanism that terminates calcium transients
- **Gap_Junction_Coupling**: Diffusive coupling of calcium between adjacent astrocyte domains through gap junction channels, enabling calcium waves to propagate across the astrocyte network
- **Spatial_Embedding**: The spectral embedding of weights into 3D space computed in Step 01, used to define astrocyte domain territories and spatial distance relationships
- **Credit_Assignment_Gap**: The 72-percentage-point performance difference between backpropagation (89%) and the best local rule (16.5%) on FashionMNIST, primarily caused by the inability of local rules to propagate error information to early layers
- **Central_Prediction**: The hypothesis that astrocyte gating provides greater benefit under local learning rules than under backpropagation — i.e., (accuracy with gate − accuracy without gate) under local rules exceeds the +0.14% benefit measured under backprop in Phase 1
- **LocalMLP**: The 784→128→128→128→128→10 architecture with detached forward pass used for all experiments, identical to Phase 1's DeeperMLP but with gradient flow blocked between layers
- **FashionMNIST**: The Fashion-MNIST dataset (10 clothing categories, 28×28 grayscale images), the primary benchmark for all Phase 2 experiments

## Requirements

### Requirement 1: Project Directory Structure

**User Story:** As a researcher, I want Step 13 organized in its own directory following the established convention, so that code, data, and results are self-contained and traceable.

#### Acceptance Criteria

1. THE Step_Directory SHALL be located at `steps/13-astrocyte-gating/` relative to the project root
2. THE Step_Directory SHALL contain subdirectories named `docs/`, `code/`, `data/`, and `results/` for organizing step artifacts
3. THE Step_Directory SHALL contain a `README.md` file summarizing the step's purpose, how to run experiments, and how to interpret results
4. THE Step_Directory SHALL contain a `docs/decisions.md` file recording design decisions and their rationale

### Requirement 2: Three-Factor Error Signal Stability Fix

**User Story:** As a researcher, I want the three-factor rule with layer-wise error to be numerically stable, so that it can serve as a reliable baseline and substrate for astrocyte gating.

#### Acceptance Criteria

1. WHEN the layer-wise error signal magnitude exceeds a configurable threshold (default 10.0), THE Three_Factor_Rule SHALL clip the error signal to that threshold before applying it as the third factor
2. WHEN the eligibility trace norm exceeds a configurable threshold (default 100.0), THE Three_Factor_Rule SHALL normalize the trace to unit norm scaled by a safe constant (default 1.0)
3. THE Three_Factor_Rule with layer-wise error SHALL complete 50 epochs of training on FashionMNIST without producing NaN or Inf values in any weight tensor across all 3 random seeds
4. THE stability fix SHALL preserve the directional information of the error signal (clipping magnitude without changing sign)

### Requirement 3: Li-Rinzel Calcium Dynamics Model

**User Story:** As a researcher, I want a biologically-grounded calcium dynamics model for the astrocyte, so that the gating signal emerges from realistic intracellular signaling rather than arbitrary thresholds.

#### Acceptance Criteria

1. THE Calcium_Dynamics model SHALL implement the Li-Rinzel two-variable system: cytoplasmic calcium concentration [Ca²⁺] and IP3 receptor inactivation variable h
2. THE Calcium_Dynamics model SHALL include IP3-dependent calcium release from the ER (J_channel), SERCA pump reuptake (J_pump), and ER leak current (J_leak)
3. THE Calcium_Dynamics model SHALL compute IP3 production rate proportional to the mean absolute activity in the astrocyte's domain (glutamate spillover analog)
4. THE Calcium_Dynamics model SHALL use configurable parameters: IP3 production rate, calcium threshold for D-serine release, SERCA pump rate, ER leak rate, and IP3 receptor kinetics
5. THE Calcium_Dynamics model SHALL operate on PyTorch tensors and support batched computation across multiple astrocyte domains simultaneously
6. THE Calcium_Dynamics model SHALL maintain numerically stable calcium concentrations (bounded between 0 and a physiological maximum of 10.0 μM equivalent) across all training conditions

### Requirement 4: Astrocyte Domain Assignment

**User Story:** As a researcher, I want each astrocyte to govern a spatial territory of weights derived from the Step 01 embedding, so that the gating signal respects the spatial structure that Step 12 showed is meaningful under local rules.

#### Acceptance Criteria

1. THE Domain_Assignment SHALL partition weights in each layer into non-overlapping astrocyte domains using spatial proximity in the Step 01 embedding coordinates
2. THE Domain_Assignment SHALL use configurable domain size (default: 16 weights per domain for 128-unit layers, yielding 8 domains per layer)
3. THE Domain_Assignment SHALL support a random-assignment mode (for ablation) where weights are assigned to domains randomly rather than by spatial proximity
4. THE Domain_Assignment SHALL be computed once at initialization and remain fixed throughout training
5. WHEN the spatial embedding from Step 01 is not available, THE Domain_Assignment SHALL fall back to contiguous index-based partitioning with a logged warning

### Requirement 5: Binary Gate (Variant A)

**User Story:** As a researcher, I want a binary astrocyte gate that permits or blocks plasticity based on calcium threshold, so that I have a baseline showing the effect of structured gating without directional information.

#### Acceptance Criteria

1. THE Binary_Gate SHALL implement the ThirdFactorInterface protocol with the same compute_signal signature as Step 12's third-factor providers
2. THE Binary_Gate SHALL output 1.0 for weights in domains where calcium exceeds the D-serine release threshold, and 0.0 otherwise
3. THE Binary_Gate SHALL update calcium dynamics at each training step based on the domain's mean absolute activation level
4. THE Binary_Gate SHALL apply the same gate value to all weights within a single astrocyte domain (domain-level control)
5. WHEN integrated with the Three_Factor_Rule, THE Binary_Gate SHALL produce weight updates only in domains where the gate is open (calcium above threshold)

### Requirement 6: Directional Gate (Variant B)

**User Story:** As a researcher, I want a directional astrocyte gate that carries error information derived from domain activity patterns, so that the gate provides credit assignment rather than just plasticity selection.

#### Acceptance Criteria

1. THE Directional_Gate SHALL implement the ThirdFactorInterface protocol with the same compute_signal signature as Step 12's third-factor providers
2. THE Directional_Gate SHALL maintain a running prediction of each domain's expected activity pattern using an exponential moving average (configurable decay, default 0.95)
3. THE Directional_Gate SHALL compute the Activity_Error_Signal as the difference between the domain's current mean activation and its predicted activation (current − predicted)
4. THE Directional_Gate SHALL compute the output signal as: calcium_state × activity_error_signal, where calcium_state provides magnitude gating and activity_error_signal provides direction
5. THE Directional_Gate SHALL produce signed output (positive or negative) that indicates the direction weights should change, not just whether they should change
6. THE Directional_Gate SHALL normalize the activity error signal per domain to prevent magnitude differences between domains from dominating the learning signal
7. WHEN calcium is below the D-serine release threshold, THE Directional_Gate SHALL output zero regardless of the activity error signal (calcium gates the teaching signal)

### Requirement 7: Volume-Transmitted Teaching Signal (Variant C)

**User Story:** As a researcher, I want a volume-transmitted teaching signal that diffuses error information through the spatial field, so that I can test whether spatially-graded teaching can approach backpropagation performance.

#### Acceptance Criteria

1. THE Volume_Teaching_Signal SHALL implement the ThirdFactorInterface protocol with the same compute_signal signature as Step 12's third-factor providers
2. THE Volume_Teaching_Signal SHALL compute a source error signal at each domain based on the mismatch between domain activity and a label-derived target (using the domain's projection of the one-hot label, analogous to Step 12's layer-wise error but domain-local)
3. THE Volume_Teaching_Signal SHALL diffuse the error signal spatially using a Gaussian kernel over the spatial embedding distances, with configurable diffusion radius (default: mean inter-domain distance)
4. THE Volume_Teaching_Signal SHALL attenuate the diffused signal with distance from the source domain, so that nearby domains receive stronger teaching signals than distant ones
5. THE Volume_Teaching_Signal SHALL combine the diffused error signal with calcium-dependent gating (signal is zero where calcium is below threshold)
6. THE Volume_Teaching_Signal SHALL produce a per-weight signed signal that indicates both magnitude and direction of desired weight change
7. THE Volume_Teaching_Signal SHALL use gap junction coupling to propagate calcium between adjacent domains, enabling coordinated gating across the spatial field

### Requirement 8: Integration with Three-Factor Rule

**User Story:** As a researcher, I want the astrocyte gate variants to plug directly into Step 12's ThreeFactorRule without modifying the core learning rule, so that the comparison is fair and the interface contract is validated.

#### Acceptance Criteria

1. WHEN any astrocyte gate variant is passed as the third_factor argument to ThreeFactorRule, THE Three_Factor_Rule SHALL use it identically to the Step 12 placeholder implementations (no special-case code)
2. THE astrocyte gate variants SHALL return tensors compatible with the Three_Factor_Rule's modulation logic: scalar, (out_features,), or (out_features, in_features) shape
3. THE astrocyte gate variants SHALL accept the same arguments as ThirdFactorInterface.compute_signal: layer_activations, layer_index, labels, global_loss, and prev_loss
4. THE astrocyte gate variants SHALL include a reset method that clears calcium state and activity predictions between epochs (matching the ThreeFactorRule.reset protocol)
5. WHEN the astrocyte gate is used with the Three_Factor_Rule, THE combined system SHALL complete 50 epochs of FashionMNIST training without numerical instability (no NaN or Inf in weights)

### Requirement 9: Performance Comparison Experiment

**User Story:** As a researcher, I want to compare all gate variants against baselines on FashionMNIST, so that I can quantify the benefit of astrocyte gating for local learning.

#### Acceptance Criteria

1. THE Performance_Experiment SHALL compare the following conditions: (a) three-factor with random gate (Step 12 baseline), (b) three-factor with global reward (Step 12 baseline), (c) three-factor with binary astrocyte gate (Variant A), (d) three-factor with directional astrocyte gate (Variant B), (e) three-factor with volume teaching signal (Variant C), (f) backpropagation baseline
2. THE Performance_Experiment SHALL use the identical architecture (784→128→128→128→128→10 LocalMLP), dataset (FashionMNIST), batch size, and epoch count as Step 12
3. THE Performance_Experiment SHALL run each condition with at least 3 random seeds and report mean and standard deviation of test accuracy
4. THE Performance_Experiment SHALL record per-epoch test accuracy, training loss, and weight norms for convergence analysis
5. THE Performance_Experiment SHALL store results in CSV format in the `results/` subdirectory with timestamps in the filename
6. THE Performance_Experiment SHALL generate a summary comparison table and bar chart visualization saved to the `results/` subdirectory

### Requirement 10: Central Prediction Test

**User Story:** As a researcher, I want to formally test whether astrocyte gating benefits local rules more than it benefits backpropagation, so that I can validate or refute the core hypothesis of Phase 2.

#### Acceptance Criteria

1. THE Central_Prediction_Test SHALL compute benefit_under_local_rules as the accuracy difference between the best astrocyte-gated condition and the ungated three-factor baseline (from Step 12)
2. THE Central_Prediction_Test SHALL compute benefit_under_backprop as 0.14 percentage points (the measured benefit from Phase 1, Step 03)
3. THE Central_Prediction_Test SHALL report whether benefit_under_local_rules exceeds benefit_under_backprop, with the magnitude of the difference
4. THE Central_Prediction_Test SHALL compute a confidence interval for benefit_under_local_rules using the variance across random seeds
5. THE Central_Prediction_Test SHALL produce a visualization comparing the two benefit magnitudes (bar chart with error bars) saved to `results/central_prediction_test.png`
6. THE Central_Prediction_Test SHALL state a clear conclusion: "hypothesis confirmed" if benefit_under_local_rules > benefit_under_backprop with non-overlapping confidence intervals, "hypothesis refuted" if benefit_under_local_rules ≤ benefit_under_backprop, or "inconclusive" if confidence intervals overlap

### Requirement 11: Calcium Dynamics Ablation

**User Story:** As a researcher, I want to test whether the full Li-Rinzel calcium dynamics provide benefit over simpler gating mechanisms, so that I can determine whether biological complexity is computationally necessary.

#### Acceptance Criteria

1. THE Calcium_Ablation SHALL compare four gating mechanisms using the directional gate (Variant B) as the base: (a) full Li-Rinzel calcium dynamics, (b) simple threshold gate (gate = 1 if mean activity > threshold, else 0), (c) linear exponential moving average filter (gate = EMA of activity), (d) random gate with matched sparsity (same fraction of open gates as the full model)
2. THE Calcium_Ablation SHALL use identical experimental conditions (architecture, dataset, seeds, epochs) across all four mechanisms
3. THE Calcium_Ablation SHALL report accuracy, convergence speed, and gate statistics (fraction of time open, temporal autocorrelation of gate signal) for each mechanism
4. THE Calcium_Ablation SHALL produce a comparison table and visualization saved to the `results/` subdirectory
5. IF the full Li-Rinzel model outperforms the simple threshold by more than 2 percentage points, THEN THE Calcium_Ablation SHALL report which nonlinear feature (hysteresis, oscillations, or threshold sharpness) contributes most by running additional targeted ablations

### Requirement 12: Spatial Domain Structure Ablation

**User Story:** As a researcher, I want to test whether spatial domain assignment matters compared to random assignment, so that I can validate whether the spatial embedding from Step 01 provides meaningful structure for astrocyte gating.

#### Acceptance Criteria

1. THE Spatial_Ablation SHALL compare two domain assignment strategies using the directional gate (Variant B): (a) spatial assignment based on Step 01 embedding proximity, (b) random assignment with the same domain sizes
2. THE Spatial_Ablation SHALL use identical experimental conditions (architecture, dataset, seeds, epochs) across both strategies
3. THE Spatial_Ablation SHALL report accuracy, convergence speed, and spatial correlation of weight updates (using the same metric as Step 12) for each strategy
4. THE Spatial_Ablation SHALL produce a comparison table and visualization saved to the `results/` subdirectory
5. IF spatial assignment outperforms random assignment by more than 2 percentage points, THEN THE Spatial_Ablation SHALL report this as evidence that spatial structure is computationally meaningful for astrocyte gating

### Requirement 13: Experiment Infrastructure and Reproducibility

**User Story:** As a researcher, I want all experiments to be fully reproducible with logged metadata and timestamps, so that results can be referenced by subsequent steps and the research is scientifically rigorous.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL reuse or adapt the experiment infrastructure from Step 12 (runner, metrics collection, seed management)
2. THE Experiment_Runner SHALL log all hyperparameters, random seeds, library versions, and hardware information to a JSON metadata file in the `results/` subdirectory
3. WHEN an experiment completes, THE Experiment_Runner SHALL include a UTC timestamp in the output filename (format: YYYYMMDD_HHMMSS)
4. THE Experiment_Runner SHALL generate a summary markdown file in the `results/` subdirectory documenting key findings, the central prediction test result, and implications for subsequent steps
5. THE Experiment_Runner SHALL support running individual conditions or all conditions via command-line arguments
6. THE Experiment_Runner SHALL checkpoint model state and astrocyte state (calcium, predictions) every 10 epochs to the `data/` subdirectory so that training can be resumed

### Requirement 14: Code Reuse from Step 12

**User Story:** As a researcher, I want Step 13 to reuse Step 12's infrastructure and interfaces, so that the comparison is fair and the astrocyte gate is validated as a true drop-in replacement.

#### Acceptance Criteria

1. THE Step_13 code SHALL import and use the ThreeFactorRule class from Step 12 without modification to the core learning rule logic
2. THE Step_13 code SHALL import and use the ThirdFactorInterface protocol from Step 12 as the contract for all astrocyte gate implementations
3. THE Step_13 code SHALL reuse or adapt the LocalMLP architecture, data pipeline, and experiment runner from Step 12
4. THE Step_13 code SHALL reuse the spatial embedding coordinates from Step 01 for domain assignment
5. WHERE Step 12 infrastructure requires extension (adding astrocyte state checkpointing, additional metrics), THE Step_13 code SHALL extend rather than replace the original modules

