# Requirements Document

## Introduction

This document specifies the requirements for implementing Step 01 (Spatial Embedding) of the glia-augmented neural network research plan. Step 01 is the universal dependency — it produces a spatial position array of shape `(N_weights, 3)` that all subsequent research steps consume. The implementation covers building a baseline MLP, implementing seven spatial embedding strategies (including differentiable positions), measuring embedding quality and spatial coherence, testing whether spatial learning rate coupling benefits from high-quality embeddings, and establishing the three-point validation curve (adversarial → random → good embedding) that distinguishes the strong spatial structure claim from the weak regularization claim.

Informed by Critical Review 3, this spec also includes: an adversarial embedding baseline, temporal quality tracking during training, a spatial coherence metric, differentiable positions as a learnable embedding strategy, and a spatially-structured benchmark task alongside MNIST.

The project uses a per-step directory structure where each research step gets its own directory containing documents, reasoning, code, data, and results, ensuring formal traceability of decisions and insights.

## Glossary

- **Embedding**: A mapping from network weight indices to 3D spatial coordinates, producing an array of shape `(N_weights, 3)`
- **MLP**: Multi-Layer Perceptron — a feedforward neural network with fully connected layers
- **Spatial_LR_Coupling**: A mechanism that averages each weight's learning rate with the learning rates of its k-nearest spatial neighbors
- **Embedding_Quality_Score**: The Pearson correlation between pairwise spatial distances and pairwise gradient correlations across weights; a good embedding has strong negative correlation (spatially close weights have correlated gradients)
- **KNN_Graph**: A k-nearest-neighbors graph built over the spatial positions of weights, used for spatial coupling operations
- **Spectral_Embedding**: An embedding derived from the eigenvectors of the graph Laplacian of the network's connectivity structure
- **Developmental_Embedding**: A co-evolving embedding that iteratively adjusts weight positions based on gradient correlation signals during training
- **Baseline_MLP**: A 2-hidden-layer MLP with 256 units per layer trained on MNIST classification
- **Step_Directory**: A self-contained directory for one research step, containing subdirectories for documents, code, data, and results
- **Gradient_Correlation**: The Pearson correlation between gradient vectors of two weights computed over a batch of training examples
- **Adversarial_Embedding**: A spatial embedding deliberately constructed to anti-correlate spatial proximity with gradient correlation, serving as a negative control
- **Differentiable_Embedding**: A learnable embedding where spatial positions are PyTorch parameters optimized jointly with the network via a spatial coherence loss term
- **Spatial_Coherence_Score**: A metric measuring whether spatially close weights develop similar PCA projections after training — tests whether spatial coupling produces spatially organized representations
- **Topographic_Task**: A benchmark task with inherent spatial structure (e.g., topographic sensor processing) where nearby inputs are related and spatial locality in weight space corresponds to functional similarity
- **Temporal_Quality_Tracking**: Measurement of embedding quality at multiple points during training to detect whether an initially good embedding degrades as the network learns

## Requirements

### Requirement 1: Project Directory Structure

**User Story:** As a researcher, I want each research step organized in its own directory with standardized subdirectories, so that documents, reasoning, code, data, and results are traceable and self-contained.

#### Acceptance Criteria

1. THE Step_Directory SHALL contain subdirectories named `docs/`, `code/`, `data/`, and `results/` for organizing step artifacts
2. THE Step_Directory SHALL be located at `steps/01-spatial-embedding/` relative to the project root
3. THE Step_Directory SHALL contain a `README.md` file summarizing the step's purpose, how to run experiments, and how to interpret results
4. WHEN a new research step is created, THE Step_Directory SHALL follow the naming convention `steps/NN-step-name/` where NN is the zero-padded step number

### Requirement 2: Baseline MLP Implementation

**User Story:** As a researcher, I want a small MLP trained on MNIST classification, so that I have a concrete network whose weights can be spatially embedded and whose gradients can be measured.

#### Acceptance Criteria

1. THE Baseline_MLP SHALL have 2 hidden layers with 256 units each and use ReLU activations
2. THE Baseline_MLP SHALL be trained on the MNIST dataset using PyTorch and the torchvision data loader
3. THE Baseline_MLP SHALL achieve at least 95% test accuracy when trained with the Adam optimizer as a validation that the network is functioning correctly
4. THE Baseline_MLP SHALL expose per-weight gradient tensors after each backward pass for use by embedding quality measurement
5. THE Baseline_MLP SHALL store trained model checkpoints in the `data/` subdirectory of the Step_Directory

### Requirement 3: Linear Embedding Strategy

**User Story:** As a researcher, I want a naive linear mapping from weight indices to 3D coordinates, so that I have a simple deterministic baseline embedding to compare against.

#### Acceptance Criteria

1. WHEN given a weight identified by layer index, source neuron index, and target neuron index, THE Linear_Embedding SHALL map it to a 3D coordinate using normalized index values as (layer/total_layers, source/max_neurons, target/max_neurons)
2. THE Linear_Embedding SHALL produce an output array of shape `(N_weights, 3)` with all coordinates in the range [0, 1]

### Requirement 4: Random Embedding Strategy

**User Story:** As a researcher, I want a random spatial embedding with a fixed seed, so that I have a control condition representing zero structural information.

#### Acceptance Criteria

1. THE Random_Embedding SHALL assign uniformly distributed random 3D coordinates to each weight using a configurable random seed
2. THE Random_Embedding SHALL produce an output array of shape `(N_weights, 3)` with all coordinates in the range [0, 1]
3. WHEN the same seed is provided, THE Random_Embedding SHALL produce identical coordinates across runs

### Requirement 5: Spectral Embedding Strategy

**User Story:** As a researcher, I want a topology-preserving embedding derived from the network's connectivity graph, so that weights connecting neurons with similar connectivity patterns end up spatially close.

#### Acceptance Criteria

1. THE Spectral_Embedding SHALL construct a graph Laplacian from the network's weight connectivity structure
2. THE Spectral_Embedding SHALL use the smallest non-trivial eigenvectors of the graph Laplacian as spatial coordinates
3. THE Spectral_Embedding SHALL produce an output array of shape `(N_weights, 3)` using the first 3 non-trivial eigenvectors
4. IF the graph Laplacian computation fails due to disconnected components, THEN THE Spectral_Embedding SHALL handle each connected component separately and combine the results

### Requirement 6: Correlation-Based Embedding Strategy

**User Story:** As a researcher, I want an embedding where weights with correlated activations are placed spatially close, so that I can test whether functional similarity predicts useful spatial proximity.

#### Acceptance Criteria

1. WHEN given a trained model and a data loader, THE Correlation_Embedding SHALL compute pairwise activation correlations across weights by running data through the network
2. THE Correlation_Embedding SHALL use multidimensional scaling (MDS) from scikit-learn to embed the correlation matrix into 3D coordinates
3. THE Correlation_Embedding SHALL produce an output array of shape `(N_weights, 3)`
4. THE Correlation_Embedding SHALL use a configurable number of data batches for computing correlations, defaulting to 10 batches

### Requirement 7: Layered-Clustered Embedding Strategy

**User Story:** As a researcher, I want an embedding that preserves layer structure on one axis while clustering within layers based on connectivity patterns, so that I can test a hybrid structural-functional approach.

#### Acceptance Criteria

1. THE Layered_Clustered_Embedding SHALL assign the x-coordinate based on normalized layer depth
2. THE Layered_Clustered_Embedding SHALL assign y and z coordinates using spectral embedding computed independently within each layer's weight matrix
3. THE Layered_Clustered_Embedding SHALL produce an output array of shape `(N_weights, 3)`

### Requirement 8: Developmental Embedding Strategy

**User Story:** As a researcher, I want a co-evolving embedding that adjusts weight positions based on gradient correlation signals during training, so that I can test whether the embedding can self-organize to match functional structure.

#### Acceptance Criteria

1. THE Developmental_Embedding SHALL initialize with random 3D positions for all weights
2. WHEN gradient correlations are computed between weight pairs, THE Developmental_Embedding SHALL apply attractive forces between positively correlated weights and repulsive forces between uncorrelated weights
3. THE Developmental_Embedding SHALL update positions iteratively over a configurable number of steps, defaulting to 1000 steps
4. THE Developmental_Embedding SHALL produce an output array of shape `(N_weights, 3)` after convergence
5. THE Developmental_Embedding SHALL record the Embedding_Quality_Score at each update step to track convergence

### Requirement 9: Embedding Quality Measurement

**User Story:** As a researcher, I want to quantify how well a spatial embedding predicts gradient similarity, so that I can objectively compare embedding strategies and validate the core hypothesis.

#### Acceptance Criteria

1. THE Quality_Measurement SHALL compute the Embedding_Quality_Score as the Pearson correlation between pairwise spatial distances and pairwise gradient correlations
2. WHEN computing gradient correlations, THE Quality_Measurement SHALL average over at least 50 training batches to obtain stable estimates
3. THE Quality_Measurement SHALL report the score with a 95% confidence interval computed via bootstrap resampling
4. THE Quality_Measurement SHALL store results in a CSV file in the `results/` subdirectory with columns for embedding method, quality score, confidence interval bounds, and computation time
5. IF the number of weight pairs exceeds 10 million, THEN THE Quality_Measurement SHALL use random subsampling of pairs to keep computation tractable while reporting the subsample size used

### Requirement 10: Spatial Learning Rate Coupling

**User Story:** As a researcher, I want to test whether averaging each weight's learning rate with its spatial neighbors' learning rates improves training, so that I can establish whether spatial proximity carries useful information for optimization.

#### Acceptance Criteria

1. THE Spatial_LR_Coupling SHALL build a KNN_Graph over the spatial positions with a configurable k parameter, defaulting to k=10
2. WHEN updating learning rates, THE Spatial_LR_Coupling SHALL compute each weight's effective learning rate as a weighted average of its own learning rate and the mean learning rate of its k-nearest spatial neighbors
3. THE Spatial_LR_Coupling SHALL support a configurable coupling strength parameter alpha in [0, 1] where 0 means no coupling and 1 means full neighbor averaging
4. THE Spatial_LR_Coupling SHALL be compatible with the Adam optimizer by modulating the per-parameter learning rate

### Requirement 11: Embedding Comparison Experiment

**User Story:** As a researcher, I want to run a controlled experiment comparing all six embedding strategies with and without spatial LR coupling, so that I can determine which embeddings produce meaningful spatial structure and whether that structure benefits optimization.

#### Acceptance Criteria

1. THE Comparison_Experiment SHALL train the Baseline_MLP with each of the 6 embedding strategies combined with Spatial_LR_Coupling enabled, plus a baseline with no spatial coupling (Adam only), for a total of 7 experimental conditions
2. THE Comparison_Experiment SHALL run each condition with at least 3 different random seeds and report mean and standard deviation of metrics
3. THE Comparison_Experiment SHALL measure and record: final test accuracy, steps to reach 95% of final accuracy, Embedding_Quality_Score, Spatial_Coherence_Score, and wall-clock training time for each condition
4. THE Comparison_Experiment SHALL store all results in a structured CSV file in the `results/` subdirectory
5. THE Comparison_Experiment SHALL generate a summary visualization showing embedding quality score versus downstream performance benefit as a scatter plot saved to the `results/` subdirectory
6. THE Comparison_Experiment SHALL include the Adversarial_Embedding and Differentiable_Embedding conditions in addition to the 6 fixed strategies, for a total of 9 spatially-coupled conditions plus 1 uncoupled baseline (10 conditions total)
7. THE Comparison_Experiment SHALL run on both MNIST and the Topographic_Task to test whether spatial mechanisms benefit differently on tasks with and without inherent spatial structure

### Requirement 12: Embedding Quality Predicts Downstream Benefit

**User Story:** As a researcher, I want to verify that embedding quality (gradient-distance correlation) predicts whether spatial LR coupling helps or hurts, so that I can establish the boundary condition for the entire glia-augmented framework.

#### Acceptance Criteria

1. THE Boundary_Condition_Test SHALL compute the Pearson correlation between Embedding_Quality_Score and the performance delta (spatially-coupled accuracy minus baseline accuracy) across all embedding methods
2. WHEN the Embedding_Quality_Score is below a threshold (to be determined empirically), THE Boundary_Condition_Test SHALL verify that Spatial_LR_Coupling does not improve performance relative to baseline
3. THE Boundary_Condition_Test SHALL produce a scatter plot of embedding quality versus performance delta with a fitted regression line, saved to the `results/` subdirectory
4. THE Boundary_Condition_Test SHALL record the correlation coefficient and p-value in the results CSV

### Requirement 13: Developmental Embedding Convergence Analysis

**User Story:** As a researcher, I want to track whether the developmental embedding converges to a stable, high-quality configuration, so that I can assess whether self-organizing embeddings are viable for the framework.

#### Acceptance Criteria

1. THE Convergence_Analysis SHALL record the Embedding_Quality_Score at regular intervals (every 50 position update steps) during developmental embedding evolution
2. THE Convergence_Analysis SHALL plot the quality score trajectory over update steps and save it to the `results/` subdirectory
3. THE Convergence_Analysis SHALL report whether the embedding quality stabilizes (defined as less than 5% relative change over the final 20% of update steps)
4. THE Convergence_Analysis SHALL compare the final developmental embedding quality to the best fixed embedding strategy

### Requirement 14: Reproducibility and Traceability

**User Story:** As a researcher, I want all experiments to be fully reproducible and all decisions traceable, so that insights can be referenced in subsequent research steps.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL log all hyperparameters, random seeds, library versions, and hardware information to a JSON metadata file in the `results/` subdirectory for each experiment run
2. THE Experiment_Runner SHALL set random seeds for Python, NumPy, and PyTorch at the start of each experiment to ensure deterministic execution
3. WHEN an experiment completes, THE Experiment_Runner SHALL generate a summary markdown file in the `results/` subdirectory documenting key findings, unexpected observations, and implications for subsequent steps
4. THE Step_Directory SHALL contain a `docs/decisions.md` file recording design decisions and their rationale as they are made during implementation

### Requirement 15: Adversarial Embedding Baseline

**User Story:** As a researcher, I want an embedding deliberately designed to anti-correlate spatial proximity with gradient correlation, so that I can establish the negative end of the three-point validation curve and confirm that bad spatial structure actively hurts performance.

#### Acceptance Criteria

1. THE Adversarial_Embedding SHALL compute gradient correlations from a partially-trained model and assign spatial positions that maximize spatial distance between highly-correlated weight pairs
2. THE Adversarial_Embedding SHALL produce an output array of shape `(N_weights, 3)` with all coordinates in the range [0, 1]
3. THE Adversarial_Embedding SHALL produce a negative Embedding_Quality_Score (positive correlation between spatial distance and gradient correlation, meaning correlated weights are far apart)
4. WHEN used with Spatial_LR_Coupling, THE Adversarial_Embedding SHALL be expected to degrade performance relative to the uncoupled baseline, confirming that spatial structure matters directionally

### Requirement 16: Differentiable Embedding Strategy

**User Story:** As a researcher, I want spatial positions that are learnable parameters optimized jointly with the network via a spatial coherence loss, so that I can test whether the embedding can co-adapt to the glial system without the chicken-and-egg problem of the developmental approach.

#### Acceptance Criteria

1. THE Differentiable_Embedding SHALL represent spatial positions as a PyTorch Parameter tensor of shape `(N_weights, 3)` that participates in gradient computation
2. THE Differentiable_Embedding SHALL add a spatial coherence loss term to the training objective: weights with high gradient correlation should be penalized for being spatially distant
3. THE Differentiable_Embedding SHALL use a configurable weighting parameter lambda_spatial to balance the spatial coherence loss against the task loss, defaulting to 0.01
4. THE Differentiable_Embedding SHALL normalize positions to [0, 1] after each optimization step via a clamping or sigmoid transformation
5. THE Differentiable_Embedding SHALL produce an output array of shape `(N_weights, 3)` at any point during training by reading the current parameter values

### Requirement 17: Embedding Quality Over Training Time

**User Story:** As a researcher, I want to measure embedding quality at multiple points during training, so that I can detect whether an initially good embedding degrades as the network's functional structure evolves.

#### Acceptance Criteria

1. THE Temporal_Quality_Tracking SHALL compute the Embedding_Quality_Score at configurable intervals during training, defaulting to every 2 epochs
2. THE Temporal_Quality_Tracking SHALL record the quality trajectory for each embedding method in a CSV file with columns for epoch, training step, and quality score
3. THE Temporal_Quality_Tracking SHALL flag any embedding whose quality score drops by more than 50% from its initial value during training
4. THE Temporal_Quality_Tracking SHALL compare the temporal stability of fixed embeddings (spectral, correlation) against the differentiable embedding to determine whether co-adaptation maintains quality

### Requirement 18: Spatial Coherence Metric

**User Story:** As a researcher, I want to measure whether training with spatial coupling produces spatially organized weight representations, so that I can verify the mechanism is working as intended (not just providing generic regularization).

#### Acceptance Criteria

1. THE Spatial_Coherence_Score SHALL compute the top-k principal components of the weight matrix after training (defaulting to k=10)
2. THE Spatial_Coherence_Score SHALL measure the Pearson correlation between pairwise spatial distances and pairwise similarities in PCA projection space (dot product of PC projections)
3. THE Spatial_Coherence_Score SHALL be computed for both spatially-coupled and uncoupled training conditions to establish whether spatial coupling induces spatial organization
4. THE Spatial_Coherence_Score SHALL be reported alongside the Embedding_Quality_Score in all results files
5. IF the Spatial_Coherence_Score is significantly higher for spatially-coupled training than uncoupled training, THEN the mechanism is confirmed to produce spatially organized representations (not just regularization)

### Requirement 19: Spatially-Structured Benchmark Task

**User Story:** As a researcher, I want a benchmark task with inherent spatial structure, so that I can test whether spatial embedding mechanisms provide greater benefit on tasks where the computational structure respects locality.

#### Acceptance Criteria

1. THE Topographic_Task SHALL process inputs from a simulated topographic sensor array where spatially adjacent sensors receive correlated signals
2. THE Topographic_Task SHALL be implementable as a classification or regression task using the same Baseline_MLP architecture (with adjusted input dimensions if needed)
3. THE Topographic_Task SHALL have a known ground-truth spatial structure that can be compared against the learned embedding
4. THE Topographic_Task SHALL be simple enough to train in comparable time to MNIST (not requiring hours of GPU time per condition)
5. WHEN comparing results across MNIST and the Topographic_Task, THE experiment SHALL report whether the performance benefit of spatial coupling is larger on the spatially-structured task (as the framework predicts)
