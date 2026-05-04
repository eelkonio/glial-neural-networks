# Log 001: Project Setup and Baseline Implementation

## Date: 2025-01-XX

## Tasks Completed

### Task 1.1: Directory Structure
- Created `steps/12-local-learning-rules/` with all subdirectories
- Created `__init__.py` files for all Python packages
- Created `README.md` and `docs/decisions.md`

### Task 1.2: Data Pipeline (`code/data/fashion_mnist.py`)
- `get_fashion_mnist_loaders(batch_size=128)` — loads FashionMNIST
- `embed_label(x, labels, n_classes=10)` — embeds label in first 10 pixels
- `generate_negative(x, labels, n_classes=10)` — random wrong label
- `ForwardForwardDataAdapter` — wraps loader to yield (x_pos, x_neg, labels)

### Task 2.1: Base Protocols (`code/rules/base.py`)
- `LayerState` dataclass with all local information
- `LocalLearningRule` protocol (name, compute_update, reset)
- `ThirdFactorInterface` protocol (name, compute_signal)

### Task 2.2: LocalLayer (`code/network/local_layer.py`)
- Single linear layer exposing pre/post activations
- Detaches output in local mode
- ReLU activation (configurable, off for output layer)

### Task 2.3: LocalMLP (`code/network/local_mlp.py`)
- Architecture: 784→128→128→128→128→10
- `forward(x, detach=True)` — standard forward with optional detachment
- `forward_with_states(x)` — collects per-layer LayerState
- `get_layer_activations(x)` — returns activation tensors

### Task 3.1: HebbianRule (`code/rules/hebbian.py`)
- Δw = η · mean_over_batch(outer(post, pre)) − λ · weights
- lr=0.01, decay=0.001, weight clip at norm > 100

### Task 3.2: OjaRule (`code/rules/oja.py`)
- Δw = η · mean_over_batch(post · (pre − post · w))
- lr=0.01, self-normalizing

### Task 4: Checkpoint
- All 39 tests pass
- Verification script confirms end-to-end operation

## Test Results

```
39 passed in 2.31s
```

All tests cover:
- Data pipeline: embed_label, generate_negative, ForwardForwardDataAdapter
- Network: LocalLayer (shapes, detach, activation), LocalMLP (shapes, states, locality)
- Rules: HebbianRule (formula, decay, explosion guard), OjaRule (formula, self-normalizing)

## Verification Script Output

```
[1] FashionMNIST: 469 train batches, 79 test batches, pixels in [0, 1]
[2] FF adapter: correct label embedding verified
[3] LocalMLP: 5 layers with correct shapes (784→128→128→128→128→10)
[4] Forward pass: produces (128, 10) logits
[5] HebbianRule: valid updates for all layers
[6] OjaRule: valid updates for all layers
[7] Gradient locality: output detached in local mode, gradients flow in backprop mode
```

## Configuration

- Python 3.12.13, PyTorch (latest), MPS GPU (M4 Pro)
- pytest paths configured in pyproject.toml
- conftest.py handles sys.path for step 12 imports
