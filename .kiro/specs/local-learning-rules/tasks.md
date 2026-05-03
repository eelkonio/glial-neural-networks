# Implementation Plan: Local Learning Rules (Step 12)

## Overview

This plan implements Step 12 of the glia-augmented neural network research plan: implementing five local learning rules WITHOUT glia to establish baselines for Step 13 (astrocyte gating). The implementation is ordered by dependency — shared infrastructure first, then the network architecture, then learning rules (three-factor rule prioritized as the substrate for Step 13), then experiments that compose them. Property-based tests are written alongside the components they validate.

Python + PyTorch, targeting MPS GPU on M4 Pro. Reuses infrastructure patterns from `steps/01-spatial-embedding/`. Expected runtime for full comparison: 2–4 hours (50 epochs × 3 seeds × 8 rule configurations).

## Tasks

- [ ] 1. Set up project structure and shared infrastructure
  - [ ] 1.1 Create directory structure and module scaffolding
    - Create `steps/12-local-learning-rules/` with subdirectories: `docs/`, `code/`, `code/rules/`, `code/network/`, `code/experiment/`, `code/data/`, `code/tests/`, `data/`, `results/`
    - Create `__init__.py` files for all Python packages
    - Create `steps/12-local-learning-rules/README.md` summarizing the step's purpose, how to run experiments, and how to interpret results
    - Create `steps/12-local-learning-rules/docs/decisions.md` for recording design decisions
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ] 1.2 Set up data pipeline in `code/data/fashion_mnist.py`
    - Implement `get_fashion_mnist_loaders(batch_size, data_dir, num_workers)` returning train/test DataLoaders
    - Normalize pixel values to [0, 1] range
    - Reuse data loading patterns from `steps/01-spatial-embedding/code/data.py`
    - Implement `ForwardForwardDataAdapter` class wrapping a base loader to yield `(x_pos, x_neg, labels)` tuples
    - Implement `embed_label(x, labels, n_classes)` and `generate_negative(x, labels, n_classes)` as static methods
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 14.2_

  - [ ]* 1.3 Write property tests for data pipeline (Properties 9, 15)
    - **Property 9: Forward-forward data preparation** — `embed_label` sets correct pixel, `generate_negative` produces incorrect label with unchanged image pixels
    - **Property 15: Data pipeline and FF adapter invariants** — all pixel values in [0, 1], positive samples have correct label, negative samples have incorrect label
    - **Validates: Requirements 6.3, 6.4, 13.2, 13.3**

- [ ] 2. Implement network architecture
  - [ ] 2.1 Implement `LayerState` dataclass and `LocalLearningRule` protocol in `code/rules/base.py`
    - Define `LayerState` with fields: pre_activation, post_activation, weights, bias, layer_index, labels, global_loss
    - Define `LocalLearningRule` protocol with `name` property, `compute_update(state) -> Tensor`, and `reset()` method
    - Define `ThirdFactorInterface` protocol with `name` property and `compute_signal(...)` method
    - _Requirements: 2.2, 5.3, 5.9_

  - [ ] 2.2 Implement `LocalLayer` in `code/network/local_layer.py`
    - Single linear layer that exposes pre/post activations
    - Detaches output from computation graph when in local learning mode
    - ReLU activation (except output layer which uses no activation)
    - _Requirements: 2.3, 2.4_

  - [ ] 2.3 Implement `LocalMLP` in `code/network/local_mlp.py`
    - Architecture: 784→128→128→128→128→10 (same as Phase 1 DeeperMLP)
    - Implement `forward(x, detach=True)` with optional inter-layer detachment
    - Implement `forward_with_states(x) -> list[LayerState]` collecting per-layer state
    - Implement `get_layer_activations(x) -> list[Tensor]` for analysis
    - Reuse architecture dimensions from `steps/01-spatial-embedding/code/model.py`
    - _Requirements: 2.1, 2.3, 2.4, 2.5, 14.1_

  - [ ]* 2.4 Write property tests for network architecture (Properties 1, 2, 3)
    - **Property 1: Forward pass shape contract** — `forward_with_states` returns 5 LayerState objects with correct shapes (784→128, 128→128, 128→128, 128→128, 128→10)
    - **Property 2: Gradient locality invariant** — with detach=True, backward from layer i produces no gradients in layers j < i
    - **Property 3: Pluggable rule produces valid updates** — any rule's `compute_update` returns tensor matching weight shape
    - **Validates: Requirements 2.2, 2.3, 2.4, 2.5, 5.3**

- [ ] 3. Implement Hebbian and Oja's rules (quick baselines)
  - [ ] 3.1 Implement `HebbianRule` in `code/rules/hebbian.py`
    - Compute update: Δw = η · mean_over_batch(outer(post, pre)) − λ · weights
    - Configurable learning rate η (default 0.01) and decay rate λ (default 0.001)
    - Implement weight explosion guard (clip if norm > 100.0)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ] 3.2 Implement `OjaRule` in `code/rules/oja.py`
    - Compute update: Δw = η · mean_over_batch(post · (pre − post · w))
    - Configurable learning rate η (default 0.01)
    - Self-normalizing — no explicit weight decay needed
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 3.3 Write property tests for Hebbian and Oja (Properties 4, 5)
    - **Property 4: Hebbian update formula** — output equals η · mean(outer(post, pre)) − λ · weights within floating-point tolerance
    - **Property 5: Oja's rule formula and norm boundedness** — output equals η · mean(post · (pre − post · w)); weight norm stays bounded below 2.0 after 1000 random updates
    - **Validates: Requirements 3.1, 3.2, 4.1, 4.4**

- [ ] 4. Checkpoint — Verify infrastructure and baselines
  - Ensure all tests pass, ask the user if questions arise.
  - Verify LocalMLP forward pass produces correct output shapes
  - Verify Hebbian and Oja rules produce valid weight updates
  - Verify FashionMNIST data loads correctly

- [ ] 5. Implement three-factor learning rule (CRITICAL — substrate for Step 13)
  - [ ] 5.1 Implement third-factor signal providers in `code/rules/three_factor.py`
    - Implement `RandomNoiseThirdFactor(sigma=0.1)` — returns N(0, σ²) noise
    - Implement `GlobalRewardThirdFactor(baseline_decay=0.99)` — returns (prev_loss − current_loss) − running_baseline
    - Implement `LayerWiseErrorThirdFactor(n_classes=10)` — returns local error signal per layer using random projection of label one-hot
    - _Requirements: 5.4, 5.7, 5.8, 5.9_

  - [ ] 5.2 Implement `ThreeFactorRule` in `code/rules/three_factor.py`
    - Maintain eligibility trace per weight: e(t) = (1 − 1/τ) · e(t−1) + mean_over_batch(outer(post, pre))
    - Compute weight update: Δw = e · M · η where M is the third factor signal
    - Decay eligibility trace after weight update (trace is consumed)
    - Accept third factor through pluggable ThirdFactorInterface
    - Configurable: lr (0.01), tau (100), sigma (0.1)
    - Implement eligibility overflow guard (normalize if > 1e6)
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.6_

  - [ ]* 5.3 Write property tests for three-factor rule (Properties 6, 7)
    - **Property 6: Three-factor update cycle** — eligibility trace follows recurrence e(t) = (1−1/τ)·e(t−1) + outer(post, pre); weight update = e · M · η; trace magnitude decreases after update
    - **Property 7: Global reward signal computation** — signal equals (prev_loss − current_loss) − running_baseline with correct baseline EMA update
    - **Validates: Requirements 5.1, 5.2, 5.6, 5.7**

- [ ] 6. Implement forward-forward algorithm
  - [ ] 6.1 Implement `ForwardForwardRule` in `code/rules/forward_forward.py`
    - Per-layer Adam optimizers (created via `setup_optimizers(model)`)
    - Goodness computation: G = (h²).sum(dim=-1)
    - Per-layer loss: L = −log(σ(G_pos − θ)) − log(σ(θ − G_neg))
    - Layer normalization before passing activations to next layer
    - Auto-compute threshold from first batch if None
    - Implement `train_step(model, x_pos, x_neg) -> list[float]` returning per-layer losses
    - Implement `classify(model, x) -> Tensor` finding label with highest cumulative goodness
    - Configurable: lr (0.03), threshold (auto), n_negative_samples (1), n_classes (10)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [ ]* 6.2 Write property tests for forward-forward (Properties 8, 9, 10)
    - **Property 8: Forward-forward goodness and loss computation** — goodness equals (h²).sum(dim=-1); loss equals −log(σ(G_pos−θ)) − log(σ(θ−G_neg))
    - **Property 9: Forward-forward data preparation** — embed_label and generate_negative produce correct outputs (tested in 1.3, verify here with FF-specific context)
    - **Property 10: Forward-forward layer normalization** — after normalization, mean ≈ 0 (±0.01) and variance ≈ 1 ([0.9, 1.1]) per sample
    - **Validates: Requirements 6.2, 6.3, 6.4, 6.5, 6.6**

- [ ] 7. Implement predictive coding rule
  - [ ] 7.1 Implement `PredictiveCodingRule` in `code/rules/predictive_coding.py`
    - Top-down prediction weights: W_predict from each layer to layer below
    - Implement `setup_predictions(model)` to initialize prediction weights
    - Prediction error: ε = input − W_predict @ representation_above
    - Inference iterations: update representations to minimize prediction error (default 20 steps)
    - Weight updates: ΔW_up = η · mean(outer(error, repr)); ΔW_predict = η · mean(outer(error, repr_above))
    - Supervised signal at top layer biasing toward correct class
    - Implement `train_step(model, x, labels) -> float` returning total prediction error
    - Implement inference divergence guard (clamp representations, abort if errors > 1000)
    - Configurable: lr (0.01), inference_lr (0.1), n_inference_steps (20), n_classes (10)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ]* 7.2 Write property test for predictive coding (Property 11)
    - **Property 11: Predictive coding local update formulas** — prediction error = input − W_predict @ repr_above; ΔW_up = η · mean(outer(error, repr)); ΔW_predict = η · mean(outer(error, repr_above))
    - **Validates: Requirements 7.2, 7.3, 7.4**

- [ ] 8. Implement backpropagation baseline
  - [ ] 8.1 Implement backprop training loop in `code/experiment/runner.py`
    - Use LocalMLP with `detach=False` for standard gradient flow
    - Adam optimizer with default lr=1e-3
    - CrossEntropyLoss
    - Same epoch count, batch size, and data as local rules
    - Validate achieves ≥88% test accuracy on FashionMNIST
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 9. Checkpoint — Verify all learning rules train
  - Ensure all tests pass, ask the user if questions arise.
  - Run a quick smoke test (2–3 epochs) for each rule to verify training loops execute without error
  - Verify forward-forward classification produces predictions
  - Verify predictive coding inference iterations converge

- [ ] 10. Implement experiment infrastructure
  - [ ] 10.1 Implement `ExperimentRunner` in `code/experiment/runner.py`
    - Seed management: set Python, NumPy, PyTorch seeds at experiment start
    - Per-rule training loops with different train_step logic per rule type
    - Periodic checkpointing (every 10 epochs) to `data/` subdirectory
    - Metadata logging (hyperparameters, seeds, library versions, hardware) to JSON
    - Support running individual rules or all rules via function arguments
    - Reuse runner patterns from `steps/01-spatial-embedding/code/experiment/`
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 14.3_

  - [ ] 10.2 Implement `PerformanceMetrics` in `code/experiment/metrics.py`
    - Record per-epoch: train/test accuracy, train/test loss, weight norms per layer
    - Compute convergence speed: first epoch reaching 90% of final accuracy
    - Compute stability: std of test accuracy over final 10 epochs
    - Implement linear probe accuracy on frozen hidden layer activations
    - Store results in structured CSV with columns: rule, seed, epoch, accuracy, loss, weight_norms, repr_quality
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 10.3 Write property tests for metrics (Properties 12, 14)
    - **Property 12: Performance metric computations** — convergence_epoch is first index where accuracy >= 0.9 * max; stability = std(accuracy[-10:]); None if threshold never reached
    - **Property 14: Seed determinism** — same config + same seed produces identical final test accuracy (bitwise)
    - **Validates: Requirements 9.2, 9.3, 12.2**

  - [ ] 10.4 Implement comparison and visualization in `code/experiment/comparison.py`
    - Generate summary comparison table (CSV) with mean/std accuracy per rule
    - Generate accuracy bar chart (`results/accuracy_comparison.png`)
    - Generate convergence curves plot (`results/convergence_curves.png`)
    - Generate weight norm trajectory plot (`results/weight_norm_trajectories.png`)
    - _Requirements: 9.7_

- [ ] 11. Implement deficiency analysis
  - [ ] 11.1 Implement `DeficiencyAnalysis` in `code/experiment/deficiency.py`
    - `compute_credit_assignment_reach(model, rule, x, labels)` — run parallel backprop pass, correlate true gradient with rule's update signal per layer
    - `compute_weight_stability(weight_norm_history)` — analyze growth rate, oscillation amplitude, flag unbounded growth
    - `compute_representation_redundancy(activations)` — mean pairwise cosine similarity between hidden units per layer
    - `compute_inter_layer_coordination(activations)` — CKA between adjacent layer representations
    - Produce structured summary: worst layer, dominant deficiency, recommended intervention per rule
    - Generate credit assignment heatmap (`results/credit_assignment_heatmap.png`)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 11.2 Write property test for representation analysis (Property 13)
    - **Property 13: Representation analysis metric bounds** — redundancy in [-1, 1]; identical columns → redundancy = 1.0; CKA in [0, 1]; CKA(X, X) = 1.0
    - **Validates: Requirements 10.3, 10.4**

- [ ] 12. Implement spatial embedding quality measurement
  - [ ] 12.1 Implement `SpatialEmbeddingQuality` in `code/experiment/spatial_quality.py`
    - Compute pairwise correlations between local weight update signals (not backprop gradients)
    - Measure Pearson correlation between spatial distances (from Phase 1 spectral embedding) and update-signal correlations
    - Compare correlation under local rules vs under backpropagation
    - Report results per rule in `results/spatial_quality.csv`
    - Reuse spectral embedding utilities from `steps/01-spatial-embedding/code/spatial/`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 14.4_

- [ ] 13. Checkpoint — Verify analysis infrastructure
  - Ensure all tests pass, ask the user if questions arise.
  - Run deficiency analysis on a model trained for 5 epochs to verify output format
  - Verify spatial quality measurement produces valid correlations
  - Verify all CSV/JSON outputs have correct schema

- [ ] 14. Run full experiments and generate results
  - [ ] 14.1 Run all learning rules (50 epochs, 3 seeds each)
    - Execute: hebbian, oja, three_factor_random, three_factor_reward, three_factor_error, forward_forward, predictive_coding, backprop
    - Save checkpoints every 10 epochs
    - Log all metadata to `results/metadata_{timestamp}.json`
    - _Requirements: 9.1, 9.5, 12.1, 12.5_

  - [ ] 14.2 Run deficiency analysis on all trained models
    - Compute credit assignment reach, weight stability, representation redundancy, inter-layer coordination for each rule
    - Generate `results/deficiency_analysis.md` with per-rule characterization
    - Generate `results/credit_assignment_heatmap.png`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ] 14.3 Run spatial embedding quality analysis
    - Compute spatial correlation for each rule
    - Compare with backprop correlation
    - Generate `results/spatial_quality.csv`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ] 14.4 Generate summary and comparison outputs
    - Generate `results/summary_table.csv` with mean accuracy, std, convergence, stability per rule
    - Generate `results/accuracy_comparison.png` and `results/convergence_curves.png`
    - Generate `results/summary.md` documenting key findings and implications for Step 13
    - Identify which deficiencies the astrocyte gate should address
    - _Requirements: 9.7, 12.3_

- [ ] 15. Final checkpoint — Ensure all tests pass and results are complete
  - Ensure all tests pass, ask the user if questions arise.
  - Verify backprop baseline achieves ≥88% test accuracy
  - Verify all result files exist in `results/` directory
  - Verify summary.md documents implications for Step 13 (astrocyte gating)

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The three-factor rule (task 5) is the MOST IMPORTANT implementation — it is the substrate for Step 13's astrocyte gating
- Forward-forward needs per-layer optimizers and label embedding in the input
- Predictive coding needs top-down prediction weights and inference iterations before weight updates
- Deficiency analysis requires a parallel backprop pass for gradient correlation measurement
- Each task references specific requirements for traceability
- Property tests validate the 15 correctness properties from the design document
- Checkpoints ensure incremental validation at major milestones
- Expected runtime: 2–4 hours for the full comparison (task 14)
