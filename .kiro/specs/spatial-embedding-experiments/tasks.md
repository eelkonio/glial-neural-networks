# Implementation Plan: Spatial Embedding Experiments (Step 01)

## Overview

This plan implements Step 01 of the glia-augmented neural network research plan: assigning spatial coordinates to every weight in a neural network and validating whether spatial structure carries useful information for optimization. The implementation is ordered by dependency — infrastructure and core components first, then experiments that compose them. Property-based tests are written alongside the components they validate.

## Tasks

- [ ] 1. Set up project structure and dependencies
  - [ ] 1.1 Create directory structure and pyproject.toml
    - Create `steps/01-spatial-embedding/` with subdirectories: `docs/`, `code/`, `code/embeddings/`, `code/spatial/`, `code/experiment/`, `code/visualization/`, `code/tests/`, `data/`, `results/`
    - Create `pyproject.toml` with dependencies: torch, torchvision, numpy, scipy, scikit-learn, hypothesis, pytest, matplotlib, pandas
    - Create `__init__.py` files for all Python packages
    - Create `steps/01-spatial-embedding/README.md` summarizing the step's purpose, how to run experiments, and how to interpret results
    - Create `steps/01-spatial-embedding/docs/decisions.md` for recording design decisions
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 14.4_

- [ ] 2. Implement baseline MLP and data loading
  - [ ] 2.1 Implement BaselineMLP in `code/model.py`
    - 2-hidden-layer MLP: 784 → 256 (ReLU) → 256 (ReLU) → 10
    - Implement `get_weight_count()`, `get_weight_metadata()`, `get_flat_weights()`, `get_flat_gradients()` methods
    - Total weights: 784×256 + 256×256 + 256×10 = 203,264 (excluding biases)
    - _Requirements: 2.1, 2.4_

  - [ ] 2.2 Implement MNIST data loading in `code/data.py`
    - Use torchvision MNIST dataset with standard train/test split
    - Configurable batch size (default 128)
    - Data normalization to [0, 1]
    - _Requirements: 2.2_

  - [ ] 2.3 Implement topographic task in `code/topographic_task.py`
    - 16×16 sensor grid (256 inputs) with spatially correlated signals
    - Configurable correlation length (default 3.0), n_classes (default 10)
    - Generate train (50K) and test (10K) datasets with configurable seed
    - Implement `get_ground_truth_embedding()` returning known-correct spatial structure
    - _Requirements: 19.1, 19.2, 19.3, 19.4_

  - [ ] 2.4 Verify baseline MLP achieves ≥95% test accuracy on MNIST
    - Train with Adam optimizer (lr=1e-3) for 20 epochs
    - Save model checkpoint to `data/` subdirectory
    - _Requirements: 2.3, 2.5_

- [ ] 3. Implement embedding strategy protocol and simple embeddings
  - [ ] 3.1 Define EmbeddingStrategy protocol in `code/embeddings/base.py`
    - Protocol with `name` property and `embed(model, **kwargs) -> np.ndarray` method
    - Contract: returns shape `(N_weights, 3)` with values in [0, 1]
    - _Requirements: 3.2, 4.2, 5.3, 6.3, 7.3, 8.4_

  - [ ] 3.2 Implement LinearEmbedding in `code/embeddings/linear.py`
    - Map weight (layer_idx, source_neuron, target_neuron) to (layer/total_layers, source/max_neurons, target/max_neurons)
    - Normalize all coordinates to [0, 1]
    - _Requirements: 3.1, 3.2_

  - [ ] 3.3 Implement RandomEmbedding in `code/embeddings/random.py`
    - Uniform random 3D coordinates with configurable seed (default 42)
    - Deterministic: same seed produces identical output
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 3.4 Write property tests for embedding output contract and determinism
    - **Property 1: Embedding output contract (shape and range)** — all strategies produce (N_weights, 3) in [0, 1]
    - **Property 2: Embedding determinism** — identical inputs produce identical outputs
    - **Property 3: Linear embedding formula** — verify coordinate formula
    - **Validates: Requirements 3.1, 3.2, 4.1, 4.2, 4.3, 14.2**

- [ ] 4. Implement graph-based embeddings
  - [ ] 4.1 Implement SpectralEmbedding in `code/embeddings/spectral.py`
    - Build neuron-level adjacency from weight magnitudes
    - Compute graph Laplacian, extract smallest non-trivial eigenvectors via `scipy.sparse.linalg.eigsh`
    - Handle disconnected components separately, combine with spatial offsets
    - Normalize to [0, 1]
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ] 4.2 Implement LayeredClusteredEmbedding in `code/embeddings/layered_clustered.py`
    - x-coordinate = normalized layer depth (identical for all weights in same layer)
    - y, z coordinates from spectral embedding within each layer's weight matrix
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ]* 4.3 Write property test for layered-clustered x-coordinate
    - **Property 4: Layered clustered x-coordinate preserves layer structure** — all weights in same layer have identical x-coordinates equal to layer_idx/total_layers
    - **Validates: Requirements 7.1**

- [ ] 5. Implement data-dependent embeddings
  - [ ] 5.1 Implement CorrelationEmbedding in `code/embeddings/correlation.py`
    - Compute pairwise activation correlations by running data through network
    - Use scikit-learn MDS to embed correlation distance matrix into 3D
    - Subsample to 5000 weights for tractability, interpolate remaining
    - Configurable n_batches (default 10)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ] 5.2 Implement DevelopmentalEmbedding in `code/embeddings/developmental.py`
    - Initialize random positions, iteratively apply attractive/repulsive forces based on gradient correlations
    - Attractive force for positively correlated pairs, repulsive for uncorrelated/negative
    - Configurable n_steps (1000), position_lr (0.01), record_interval (50)
    - Record quality score at each interval for convergence tracking
    - Clip forces to prevent explosion
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 5.3 Write property test for developmental force direction
    - **Property 5: Developmental force direction** — positive correlation → attractive force, zero/negative → repulsive force
    - **Validates: Requirements 8.2**

- [ ] 6. Implement adversarial and differentiable embeddings
  - [ ] 6.1 Implement AdversarialEmbedding in `code/embeddings/adversarial.py`
    - Compute gradient correlations from partially-trained model
    - Use MDS on NEGATED correlation matrix (anti-MDS)
    - Highly correlated weights get maximally distant positions
    - Normalize to [0, 1]
    - _Requirements: 15.1, 15.2, 15.3_

  - [ ] 6.2 Implement DifferentiableEmbedding in `code/embeddings/differentiable.py`
    - Positions as `nn.Parameter` of shape `(N_weights, 3)`
    - Spatial coherence loss: penalize high-correlation pairs that are spatially distant
    - Configurable lambda_spatial (0.01), position_lr (1e-3)
    - Sigmoid normalization to keep positions in [0, 1]
    - `initialize()`, `compute_spatial_loss()`, `embed()` methods
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_

  - [ ]* 6.3 Write property tests for adversarial and differentiable embeddings
    - **Property 11: Adversarial embedding produces negative quality score** — correlated weights are far apart
    - **Property 12: Differentiable embedding positions remain in [0, 1]** — sigmoid/clamping ensures range
    - **Validates: Requirements 15.3, 16.4**

- [ ] 7. Checkpoint - Ensure all embedding strategies work
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement spatial operations
  - [ ] 8.1 Implement KNNGraph in `code/spatial/knn_graph.py`
    - Build KNN graph using `scipy.spatial.cKDTree`
    - Configurable k (default 10), clamp k to N-1 if k >= N
    - Expose `neighbor_indices`, `neighbor_distances`, `get_neighbors(idx)` 
    - _Requirements: 10.1_

  - [ ] 8.2 Implement SpatialLRCoupling in `code/spatial/lr_coupling.py`
    - Formula: `effective_lr[i] = (1 - alpha) * base_lr[i] + alpha * mean(base_lr[neighbors[i]])`
    - Configurable alpha in [0, 1] (default 0.5)
    - `apply_to_optimizer()` method compatible with Adam optimizer
    - Clamp LR multipliers to [0.01, 10.0] for stability
    - _Requirements: 10.2, 10.3, 10.4_

  - [ ] 8.3 Implement QualityMeasurement in `code/spatial/quality.py`
    - Pearson correlation between pairwise spatial distances and pairwise gradient correlations
    - Average over at least 50 batches for stable estimates
    - Bootstrap resampling for 95% confidence interval (n_bootstrap=1000)
    - Random subsampling when pairs exceed max_pairs (10M)
    - Handle degenerate cases (constant distances/correlations → score 0.0)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ] 8.4 Implement SpatialCoherence in `code/spatial/coherence.py`
    - Compute top-k PCA components (default k=10) of weight matrix after training
    - Pearson correlation between pairwise spatial distances and PCA projection similarities
    - _Requirements: 18.1, 18.2, 18.3, 18.4_

  - [ ] 8.5 Implement TemporalQualityTracker in `code/spatial/temporal_tracking.py`
    - Record quality score at configurable intervals (default every 2 epochs)
    - Store trajectory as list of (epoch, step, score) tuples
    - `detect_degradation(threshold=0.5)` method
    - _Requirements: 17.1, 17.2, 17.3_

  - [ ]* 8.6 Write property tests for spatial operations
    - **Property 6: Quality score is Pearson correlation** — verify formula matches scipy.stats.pearsonr
    - **Property 7: Confidence interval contains point estimate** — ci_lower <= score <= ci_upper
    - **Property 8: Spatial LR coupling formula** — verify formula, alpha=0 → no coupling, alpha=1 → full averaging
    - **Property 9: Subsampling threshold** — pairs > max_pairs triggers subsampling
    - **Validates: Requirements 9.1, 9.3, 9.5, 10.2, 10.3**

- [ ] 9. Implement experiment infrastructure
  - [ ] 9.1 Implement ExperimentRunner in `code/experiment/runner.py`
    - Seed management: set Python, NumPy, PyTorch seeds
    - Metadata logging: hyperparameters, library versions, hardware info, git hash to JSON
    - `run_condition()` for single condition/seed, `run_comparison()` for all conditions × seeds
    - _Requirements: 14.1, 14.2, 14.3_

  - [ ] 9.2 Implement reproducibility utilities in `code/experiment/reproducibility.py`
    - `set_seeds(seed)` for Python, NumPy, PyTorch
    - Hardware info collection (GPU, CUDA, CPU)
    - Library version collection (torch, numpy, scipy, sklearn)
    - _Requirements: 14.1, 14.2_

  - [ ] 9.3 Implement visualization in `code/visualization/plots.py`
    - Scatter plot: embedding quality vs performance delta
    - Regression plot: boundary condition with fitted line
    - Three-point curve: adversarial → random → good
    - Trajectory plot: developmental convergence over steps
    - Temporal quality: quality over training time per method
    - Spatial coherence: coupled vs uncoupled comparison
    - All plots saved as PNG to `results/`
    - _Requirements: 11.5, 12.3, 13.2_

- [ ] 10. Checkpoint - Ensure all components integrate correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Implement comparison experiment
  - [ ] 11.1 Implement comparison experiment in `code/experiment/comparison.py`
    - 10 conditions: 8 embedding strategies with coupling + uncoupled baseline + differentiable (jointly trained)
    - 2 tasks: MNIST and topographic task
    - 3 seeds per condition (seeds: 42, 123, 456)
    - Metrics per condition: final test accuracy, steps to 95% of final accuracy, quality score, spatial coherence score, wall-clock time
    - Write results to `results/comparison_results.csv`
    - Generate summary visualization (quality vs performance scatter)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [ ]* 11.2 Write property tests for convergence detection and temporal degradation
    - **Property 10: Convergence detection** — converged=True iff max relative change in final 20% < 5%
    - **Property 15: Temporal quality degradation detection** — degraded=True iff min < 50% of initial
    - **Validates: Requirements 13.3, 17.3**

- [ ] 12. Implement analysis experiments
  - [ ] 12.1 Implement boundary condition test in `code/experiment/boundary.py`
    - Pearson correlation between quality scores and performance deltas across all methods
    - Scatter plot with regression line saved to `results/boundary_regression.png`
    - Record correlation coefficient and p-value in CSV
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [ ] 12.2 Implement convergence analysis in `code/experiment/convergence.py`
    - Record quality score every 50 position update steps during developmental embedding
    - Plot trajectory, report whether quality stabilizes (<5% relative change in final 20%)
    - Compare final developmental quality to best fixed embedding
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ] 12.3 Implement three-point validation experiment
    - Run adversarial, random, and best embedding with spatial coupling
    - Compute performance deltas relative to uncoupled baseline
    - Verify monotonicity: adversarial_delta < random_delta < best_delta
    - Generate three-point curve plot saved to `results/three_point_curve.png`
    - _Requirements: 15.4, 12.1_

  - [ ] 12.4 Implement temporal quality tracking experiment
    - Track quality score every 2 epochs for all fixed embedding methods
    - Compare against differentiable embedding (co-adapts)
    - Flag embeddings with >50% quality degradation
    - Save trajectories to `results/temporal_quality.csv` and plot to `results/temporal_quality_trajectories.png`
    - _Requirements: 17.1, 17.2, 17.3, 17.4_

  - [ ] 12.5 Implement spatial coherence test
    - Train with and without spatial coupling using a good embedding
    - Compare coherence scores between coupled and uncoupled conditions
    - Save results to `results/spatial_coherence.csv` and plot to `results/spatial_coherence_comparison.png`
    - _Requirements: 18.3, 18.4, 18.5_

  - [ ]* 12.6 Write property tests for three-point validation and spatial coherence
    - **Property 13: Spatial coherence is higher for coupled than uncoupled training** — statistical property validated empirically
    - **Property 14: Three-point validation monotonicity** — adversarial_delta < random_delta < best_delta
    - **Validates: Requirements 15.4, 18.5**

- [ ] 13. Wire everything together and generate summary
  - [ ] 13.1 Create experiment entry points and CLI
    - Create `code/experiment/__init__.py` exporting all experiment classes
    - Create a main runner script that orchestrates all experiments in sequence
    - Generate `results/summary.md` with key findings, observations, and implications for Step 02
    - _Requirements: 14.3_

  - [ ] 13.2 Create `code/embeddings/__init__.py` exporting all strategies
    - Export all 8 embedding classes from the package
    - Provide a `get_all_strategies()` convenience function
    - _Requirements: 11.6_

- [ ] 14. Final checkpoint - Ensure all tests pass and experiments run end-to-end
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the 15 correctness properties defined in the design document
- The implementation language is Python with PyTorch, NumPy, SciPy, scikit-learn, and Hypothesis
- The topographic task provides a benchmark where spatial structure should matter more than MNIST
- Experiments compose the components built in earlier tasks — infrastructure must be solid before running experiments
