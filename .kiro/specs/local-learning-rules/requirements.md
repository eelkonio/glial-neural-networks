# Requirements Document

## Introduction

This document specifies the requirements for implementing Step 12 (Local Learning Rules) of the glia-augmented neural network research plan. Step 12 is the first step of Phase 2, which shifts from glia as modulators of backpropagation (Phase 1) to glia as constitutive components of the learning rule itself.

Phase 1 (Steps 01–01b) established that spatial LR coupling under backpropagation provides only weak regularization — the spatial structure hypothesis is not supported under backprop. The mechanism is a combination of noise reduction and regularization that does not depend on embedding quality. The go/no-go gate failed on all three mandatory criteria.

Phase 2 tests the prediction that spatial structure WILL matter under local learning rules because the glial "third factor" signal must be spatially local (astrocyte domains are ~50μm in biology). Step 12 implements local learning rules WITHOUT glia to establish baselines. Step 13 will add astrocyte gating to test whether glia make local rules competitive.

The implementation covers five learning rules: three core rules (three-factor rule, forward-forward algorithm, predictive coding) and two quick baselines (Hebbian, Oja's rule). STDP/spiking networks are deferred to a later step requiring Level 2 temporal simulation. All rules are tested on FashionMNIST using the same 4-layer MLP architecture from Phase 1 (784→128→128→128→128→10), with each layer learning independently using only local information.

The three-factor rule is the most important implementation — it is the substrate for Step 13's astrocyte gating. It uses a configurable third-factor interface with three placeholder modes in Step 12: random noise (lower bound — no useful signal), global reward (scalar reward based on loss decrease), and layer-wise error (local error signal per layer — approximates backprop). This interface is designed so Step 13 can plug in the astrocyte gate as a drop-in replacement.

## References

- Hebb, D.O. (1949). *The Organization of Behavior*. Wiley. — Foundation of Hebbian learning ("cells that fire together wire together").
- Oja, E. (1982). "Simplified neuron model as a principal component analyzer." *Journal of Mathematical Biology*, 15(3), 267–273. — Self-normalizing Hebbian rule that converges to the first principal component.
- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018). "Eligibility Traces and Plasticity on Behavioral Time Scales." *Frontiers in Neural Circuits*, 12, 53. — Formalization of three-factor learning rules with eligibility traces.
- Hinton, G. (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations." Technical report / NeurIPS 2022. — Layer-local learning via goodness maximization without backpropagation.
- Whittington, J.C.R. & Bogacz, R. (2017). "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity." *Neural Computation*, 29(5), 1229–1262. — Proof that predictive coding with inference iterations approximates backpropagation.
- Rao, R.P.N. & Ballard, D.H. (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature Neuroscience*, 2(1), 79–87. — Original predictive coding framework for cortical computation.

## Glossary

- **Local_Learning_Rule**: A weight update rule where the update for synapse w_ij depends only on information available at that synapse's physical location (pre-synaptic activity, post-synaptic activity, and optionally a broadcast signal), not on a per-weight error gradient computed by backpropagation
- **Three_Factor_Rule**: A learning rule where weight change equals the product of an eligibility trace (set by pre/post correlation) and a third factor signal; the third factor is a placeholder (random noise, global reward, or layer-wise error) in Step 12, to be replaced by astrocyte gating in Step 13
- **Forward_Forward_Algorithm**: Hinton's (2022) algorithm where each layer independently maximizes "goodness" (sum of squared activations) for positive data and minimizes it for negative data, requiring no backward pass through the full network
- **Predictive_Coding**: A learning rule where each layer predicts the activity of the layer below via top-down connections, and weight updates minimize local prediction error using only local activity and error signals
- **Hebbian_Rule**: The simplest local rule: weight change is proportional to the product of pre-synaptic and post-synaptic activity, with weight decay to prevent unbounded growth
- **Oja_Rule**: A normalized variant of the Hebbian rule that extracts the first principal component: Δw = η·post·(pre − post·w), which is self-stabilizing
- **Eligibility_Trace**: A decaying memory at each synapse that records recent pre/post co-activation, serving as a candidate signal for weight change that is only converted to an actual update when the third factor arrives
- **Third_Factor_Signal**: A broadcast or semi-local signal that gates whether eligibility traces convert to weight changes; in Step 12 this is a placeholder (random noise, global reward, or layer-wise error), in Step 13 it becomes the astrocyte D-serine gate
- **Third_Factor_Interface**: A pluggable interface that defines how the third factor signal is computed and delivered to the Three_Factor_Rule; designed so that Step 13 can replace the placeholder implementations with astrocyte gating without modifying the core learning rule
- **Layer_Wise_Error**: A local error signal computed per layer (e.g., difference between layer output and a local target derived from the label), providing an approximation of backprop's error signal using only locally available information
- **Goodness**: In the forward-forward algorithm, the sum of squared activations in a layer; positive data should produce high goodness, negative data should produce low goodness
- **Positive_Data**: In the forward-forward algorithm, real training examples with correct labels embedded in the input
- **Negative_Data**: In the forward-forward algorithm, corrupted examples with incorrect labels embedded in the input, generated to train layers to distinguish real from fake
- **Prediction_Error**: In predictive coding, the difference between bottom-up input to a layer and the top-down prediction from the layer above
- **Local_Layer**: A layer that updates its weights using only locally available information (its own input, output, and optionally a broadcast signal), with no gradient flowing backward through the full network
- **Backprop_Baseline**: The same architecture trained with standard backpropagation and Adam optimizer, serving as the upper-bound reference for local rule performance
- **Representation_Quality**: The accuracy of a linear classifier (linear probe) trained on frozen hidden layer activations, measuring how informative the learned representations are
- **Credit_Assignment_Gap**: The performance difference between a local rule and backpropagation, attributable to the local rule's inability to propagate error information to early layers
- **Spatial_Embedding_Quality**: The Pearson correlation between pairwise spatial distances and pairwise update-signal correlations (using local rule update signals instead of backprop gradients)
- **FashionMNIST**: The Fashion-MNIST dataset (10 clothing categories, 28×28 grayscale images), chosen as the primary benchmark because it does not saturate like MNIST and provides more discriminating accuracy differences between methods

## Requirements

### Requirement 1: Project Directory Structure

**User Story:** As a researcher, I want Step 12 organized in its own directory following the established convention, so that code, data, and results are self-contained and traceable.

#### Acceptance Criteria

1. THE Step_Directory SHALL be located at `steps/12-local-learning-rules/` relative to the project root
2. THE Step_Directory SHALL contain subdirectories named `docs/`, `code/`, `data/`, and `results/` for organizing step artifacts
3. THE Step_Directory SHALL contain a `README.md` file summarizing the step's purpose, how to run experiments, and how to interpret results
4. THE Step_Directory SHALL contain a `docs/decisions.md` file recording design decisions and their rationale

### Requirement 2: Local Layer Architecture

**User Story:** As a researcher, I want a 4-layer MLP where each layer can learn independently using only local information, so that I can test local learning rules on the same architecture used in Phase 1.

#### Acceptance Criteria

1. THE Local_Layer architecture SHALL use the same dimensions as Phase 1: 784→128→128→128→128→10 (4 hidden layers of 128 units plus output)
2. THE Local_Layer architecture SHALL support pluggable learning rules, allowing any Local_Learning_Rule to be attached to any layer without modifying the architecture
3. WHEN a Local_Learning_Rule is active, THE Local_Layer SHALL NOT propagate gradients backward through the full network (each layer's weight update uses only local signals)
4. THE Local_Layer architecture SHALL expose per-layer activations (pre-activation and post-activation) for use by learning rules that need them
5. THE Local_Layer architecture SHALL support a forward pass that produces class predictions, even when layers learn independently

### Requirement 3: Hebbian Learning Rule

**User Story:** As a researcher, I want a basic Hebbian learning rule implementation, so that I have the simplest possible local learning baseline.

#### Acceptance Criteria

1. THE Hebbian_Rule SHALL compute weight updates as Δw = η · pre · post, where pre is the pre-synaptic activation and post is the post-synaptic activation
2. THE Hebbian_Rule SHALL include weight decay (Δw_decay = −λ · w) to prevent unbounded weight growth
3. THE Hebbian_Rule SHALL use configurable learning rate η (default 0.01) and decay rate λ (default 0.001)
4. WHEN applied to a Local_Layer, THE Hebbian_Rule SHALL update weights using only the layer's input and output activations

### Requirement 4: Oja's Learning Rule

**User Story:** As a researcher, I want Oja's normalized Hebbian rule, so that I have a self-stabilizing local learning baseline that extracts principal components.

#### Acceptance Criteria

1. THE Oja_Rule SHALL compute weight updates as Δw = η · post · (pre − post · w), which is self-normalizing and converges to the first principal component direction
2. THE Oja_Rule SHALL use a configurable learning rate η (default 0.01)
3. WHEN applied to a Local_Layer, THE Oja_Rule SHALL update weights using only the layer's input, output, and current weight values
4. THE Oja_Rule SHALL maintain bounded weight norms without requiring explicit weight decay

### Requirement 5: Three-Factor Learning Rule

**User Story:** As a researcher, I want a three-factor learning rule with a configurable third-factor interface and placeholder implementations, so that I can establish baseline performance and Step 13 can plug in astrocyte gating without modifying the core rule.

#### Acceptance Criteria

1. THE Three_Factor_Rule SHALL maintain an Eligibility_Trace for each weight that accumulates pre/post correlation: e(t) = (1 − 1/τ) · e(t−1) + pre · post
2. THE Three_Factor_Rule SHALL compute weight updates as Δw = e · M · η, where e is the eligibility trace, M is the Third_Factor_Signal, and η is the learning rate
3. THE Three_Factor_Rule SHALL accept the Third_Factor_Signal through a pluggable Third_Factor_Interface, allowing new signal sources to be registered without modifying the core learning rule implementation
4. THE Three_Factor_Rule SHALL support three placeholder third-factor modes: random noise (sampled from N(0, σ²) per update), global reward (scalar signal based on batch accuracy improvement), and layer-wise error (local error signal per layer)
5. THE Three_Factor_Rule SHALL use configurable parameters: learning rate η (default 0.01), eligibility trace time constant τ (default 100 steps), and noise standard deviation σ (default 0.1)
6. THE Three_Factor_Rule SHALL decay the eligibility trace after it is consumed by a weight update, preventing stale traces from accumulating indefinitely
7. WHEN the third factor is global reward, THE Three_Factor_Rule SHALL compute reward as the change in batch accuracy relative to a running baseline (positive reward when accuracy improves, negative when it degrades)
8. WHEN the third factor is layer-wise error, THE Three_Factor_Rule SHALL compute a local error signal per layer by comparing the layer's output to a local target derived from the label (without backpropagating through the full network)
9. THE Third_Factor_Interface SHALL define a method that accepts layer activations, layer index, and optional global information (labels, loss) and returns a scalar or per-weight modulation signal

### Requirement 6: Forward-Forward Algorithm

**User Story:** As a researcher, I want an implementation of Hinton's forward-forward algorithm, so that I can test a modern local learning rule that is competitive with backpropagation on simple tasks.

#### Acceptance Criteria

1. THE Forward_Forward_Algorithm SHALL train each layer independently to maximize Goodness for Positive_Data and minimize Goodness for Negative_Data
2. THE Forward_Forward_Algorithm SHALL define Goodness as the sum of squared activations in a layer: G = Σ(h²)
3. THE Forward_Forward_Algorithm SHALL generate Negative_Data by replacing the correct label embedding in the input with a randomly chosen incorrect label
4. THE Forward_Forward_Algorithm SHALL embed labels into the first N pixels of the input image (where N equals the number of classes), setting the pixel at the label index to the maximum value
5. THE Forward_Forward_Algorithm SHALL use a per-layer loss: L = −log(σ(G_pos − θ)) − log(σ(θ − G_neg)), where θ is a configurable goodness threshold and σ is the sigmoid function
6. THE Forward_Forward_Algorithm SHALL normalize layer activations before passing them to the next layer (layer-norm or similar) to prevent goodness from trivially increasing through the network
7. THE Forward_Forward_Algorithm SHALL use configurable parameters: learning rate (default 0.03), goodness threshold θ (default computed as mean goodness over first batch), and number of negative samples per positive (default 1)
8. WHEN performing inference, THE Forward_Forward_Algorithm SHALL classify by finding the label embedding that produces the highest cumulative goodness across all layers

### Requirement 7: Predictive Coding Learning Rule

**User Story:** As a researcher, I want a predictive coding implementation with top-down prediction connections, so that I can test a biologically-motivated local rule that approximates backpropagation through local prediction error minimization.

#### Acceptance Criteria

1. THE Predictive_Coding rule SHALL add top-down prediction weights from each layer to the layer below, creating a generative model that predicts lower-layer activity from higher-layer representations
2. THE Predictive_Coding rule SHALL compute Prediction_Error at each layer as the difference between bottom-up input and top-down prediction: ε = input − W_predict · representation_above
3. THE Predictive_Coding rule SHALL update bottom-up weights to reduce prediction error using a local Hebbian-like rule: ΔW_up = η · ε · representation^T
4. THE Predictive_Coding rule SHALL update top-down prediction weights to improve predictions: ΔW_predict = η · ε · representation_above^T
5. THE Predictive_Coding rule SHALL iterate inference (updating representations to reduce prediction error) for a configurable number of steps before updating weights, defaulting to 20 inference iterations
6. THE Predictive_Coding rule SHALL use configurable parameters: learning rate η (default 0.01), inference step size (default 0.1), and number of inference iterations (default 20)
7. THE Predictive_Coding rule SHALL include a supervised signal at the top layer that biases the top-level representation toward the correct class

### Requirement 8: Backpropagation Baseline

**User Story:** As a researcher, I want a backpropagation baseline on the same architecture and dataset, so that I can quantify the Credit_Assignment_Gap for each local rule.

#### Acceptance Criteria

1. THE Backprop_Baseline SHALL use the identical architecture (784→128→128→128→128→10) trained with standard backpropagation and the Adam optimizer
2. THE Backprop_Baseline SHALL be trained on FashionMNIST with the same data augmentation, batch size, and epoch count as the local rule experiments
3. THE Backprop_Baseline SHALL achieve at least 88% test accuracy on FashionMNIST as validation that the architecture and training setup are correct
4. THE Backprop_Baseline SHALL record the same metrics as local rules (accuracy, convergence speed, weight norms, representation quality) for direct comparison

### Requirement 9: Performance Measurement and Comparison

**User Story:** As a researcher, I want comprehensive metrics comparing all learning rules, so that I can characterize the Credit_Assignment_Gap and identify where glia should help in Step 13.

#### Acceptance Criteria

1. THE Performance_Measurement SHALL record final test accuracy for each learning rule after a fixed number of training epochs (configurable, default 50 epochs)
2. THE Performance_Measurement SHALL record convergence speed as the number of epochs to reach 90% of final accuracy
3. THE Performance_Measurement SHALL record training stability as the standard deviation of test accuracy over the final 10 epochs
4. THE Performance_Measurement SHALL compute Representation_Quality via linear probe accuracy on frozen hidden layer activations for each layer independently
5. THE Performance_Measurement SHALL run each condition with at least 3 random seeds and report mean and standard deviation
6. THE Performance_Measurement SHALL store all results in a structured CSV file in the `results/` subdirectory with columns for rule name, seed, epoch, accuracy, loss, weight norm per layer, and representation quality per layer
7. THE Performance_Measurement SHALL generate a summary comparison table and visualization saved to the `results/` subdirectory

### Requirement 10: Deficiency Analysis

**User Story:** As a researcher, I want to characterize what each local rule LACKS compared to backpropagation, so that Step 13 knows exactly where astrocyte gating should provide benefit.

#### Acceptance Criteria

1. THE Deficiency_Analysis SHALL measure credit assignment reach by computing the correlation between each layer's weight update signal and the true gradient (from a parallel backprop computation) — this quantifies how much error information reaches early layers
2. THE Deficiency_Analysis SHALL measure weight stability by tracking the L2 norm of weights in each layer over training and flagging rules where norms grow unboundedly or oscillate
3. THE Deficiency_Analysis SHALL measure representation redundancy by computing the mean pairwise cosine similarity between hidden unit activation vectors within each layer — high similarity indicates redundant representations
4. THE Deficiency_Analysis SHALL measure inter-layer coordination by computing the mutual information (or a proxy such as CKA — Centered Kernel Alignment) between adjacent layer representations
5. THE Deficiency_Analysis SHALL produce a structured summary identifying for each rule: (a) which layers suffer most, (b) what type of deficiency dominates, and (c) what type of signal would address the deficiency
6. THE Deficiency_Analysis SHALL store results in `results/deficiency_analysis.md` with quantitative measurements and qualitative interpretation

### Requirement 11: Spatial Embedding Quality Under Local Rules

**User Story:** As a researcher, I want to check whether the spatial embedding remains meaningful when learning is local rather than backprop-based, so that I can validate the Phase 2 prediction that spatial structure will matter under local rules.

#### Acceptance Criteria

1. THE Spatial_Embedding_Quality check SHALL compute pairwise correlations between local weight update signals (instead of backprop gradients) across weights
2. THE Spatial_Embedding_Quality check SHALL measure the Pearson correlation between pairwise spatial distances (using the spectral embedding from Phase 1) and pairwise update-signal correlations
3. THE Spatial_Embedding_Quality check SHALL compare this correlation under local rules versus under backpropagation to determine whether spatial structure is more or less meaningful for local learning
4. THE Spatial_Embedding_Quality check SHALL report results for each learning rule separately, since different rules may produce different spatial correlation patterns
5. IF the spatial embedding quality is higher under local rules than under backpropagation, THEN this supports the Phase 2 prediction that spatial structure matters more for local learning

### Requirement 12: Reproducibility and Experiment Infrastructure

**User Story:** As a researcher, I want all experiments to be fully reproducible with logged metadata, so that results can be referenced by subsequent steps and the research is scientifically rigorous.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL log all hyperparameters, random seeds, library versions, and hardware information to a JSON metadata file in the `results/` subdirectory
2. THE Experiment_Runner SHALL set random seeds for Python, NumPy, and PyTorch at the start of each experiment
3. WHEN an experiment completes, THE Experiment_Runner SHALL generate a summary markdown file in the `results/` subdirectory documenting key findings and implications for Step 13
4. THE Experiment_Runner SHALL support running individual rules or all rules via command-line arguments
5. THE Experiment_Runner SHALL checkpoint model state periodically (every 10 epochs) to the `data/` subdirectory so that training can be resumed and intermediate states analyzed

### Requirement 13: FashionMNIST Data Pipeline

**User Story:** As a researcher, I want a data loading pipeline for FashionMNIST that supports both standard loading and the modified input format required by forward-forward, so that all rules use consistent data.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL load FashionMNIST via torchvision with standard train/test splits (60,000 train, 10,000 test)
2. THE Data_Pipeline SHALL normalize pixel values to [0, 1] range
3. THE Data_Pipeline SHALL support the forward-forward label embedding format where the first 10 pixels encode the label as a one-hot vector scaled to the image intensity range
4. THE Data_Pipeline SHALL use a configurable batch size (default 128) and support shuffling for training
5. THE Data_Pipeline SHALL be shared across all learning rules to ensure fair comparison

### Requirement 14: Infrastructure Reuse from Phase 1

**User Story:** As a researcher, I want Step 12 to reuse infrastructure from Step 01 where possible, so that I avoid duplicating code and maintain consistency with Phase 1 results.

#### Acceptance Criteria

1. THE Step_12 code SHALL reuse or adapt the model architecture definition from Step 01 (DeeperMLP with 784→128→128→128→128→10) rather than reimplementing it from scratch
2. THE Step_12 code SHALL reuse or adapt the data loading utilities from Step 01 for FashionMNIST
3. THE Step_12 code SHALL reuse or adapt the experiment runner pattern from Step 01 (seed management, metadata logging, results collection)
4. WHERE Step 01 infrastructure requires modification for local learning (disabling autograd, adding per-layer hooks), THE Step_12 code SHALL extend rather than replace the original modules
5. THE Step_12 code SHALL import shared utilities from a common location or copy them with attribution, maintaining traceability to the Phase 1 implementation

