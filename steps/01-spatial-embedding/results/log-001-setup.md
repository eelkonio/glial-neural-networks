# Execution Log 001: Project Setup and Tasks 1-5

## Task 1: Project Structure and Dependencies

**Date**: 2026-05-03  
**Status**: Complete

### Actions
- Created directory structure: `steps/01-spatial-embedding/` with `docs/`, `code/`, `data/`, `results/`
- Created `pyproject.toml` with Python 3.12, PyTorch, NumPy, SciPy, scikit-learn, matplotlib, pandas, hypothesis, pytest
- Created Python 3.12 venv at `.venv/`
- Installed all dependencies via `pip install -e ".[dev]"`
- Verified MPS GPU support: PyTorch 2.11.0, MPS available

### Observations
- Build backend initially set incorrectly (`setuptools.backends._legacy`), fixed to `setuptools.build_meta`
- Setuptools auto-discovery found multiple top-level packages; fixed with explicit `[tool.setuptools.packages.find]`

---

## Task 2: Baseline MLP and Data Loading

**Date**: 2026-05-03  
**Status**: Complete

### Actions
- Implemented `BaselineMLP` in `code/model.py` (784→256→256→10, ReLU)
- Implemented MNIST loading in `code/data.py`
- Implemented topographic task in `code/topographic_task.py` (16×16 sensor grid)
- Trained baseline MLP for 5 epochs on MPS

### Results
- Weight count: 268,800 (not 203,264 as design doc stated — arithmetic error in doc)
- Epoch 1: 95.58% test accuracy
- Epoch 5: 97.64% test accuracy
- Checkpoint saved to `data/baseline_mlp.pt`
- **PASS**: ≥95% accuracy requirement met

### Observations
- MPS device works correctly for training
- The design doc had a typo: 784×256 + 256×256 + 256×10 = 268,800, not 203,264

---

## Task 3: Embedding Protocol and Simple Embeddings

**Date**: 2026-05-03  
**Status**: Complete

### Actions
- Defined `EmbeddingStrategy` protocol in `code/embeddings/base.py`
- Implemented `LinearEmbedding` in `code/embeddings/linear.py`
- Implemented `RandomEmbedding` in `code/embeddings/random.py`
- Wrote property tests (Properties 1-3) in `code/tests/test_embedding_properties.py`

### Test Results
- All 3 property tests pass (28.73s runtime)
- Property 1: Shape (268800, 3) and range [0, 1] verified
- Property 2: Determinism verified
- Property 3: Linear formula verified

---

## Task 4: Graph-Based Embeddings

**Date**: 2026-05-03  
**Status**: Complete

### Actions
- Implemented `SpectralEmbedding` in `code/embeddings/spectral.py`
- Implemented `LayeredClusteredEmbedding` in `code/embeddings/layered_clustered.py`
- Added Property 4 test (layered-clustered x-coordinate)

### Design Notes
- Spectral embedding operates at neuron level (1306 neurons), not weight level
- Uses midpoint interpolation for weight positions from source/target neurons
- Deterministic via fixed random state (v0) and sign convention on eigenvectors
- LayeredClusteredEmbedding uses SVD within each layer for y/z structure

### Test Results
- All 4 property tests pass (74s total runtime)
- Spectral embedding is slower due to eigenvector computation on 1306×1306 matrix

---

## Task 5: Data-Dependent Embeddings

**Date**: 2026-05-03  
**Status**: Complete

### Actions
- Implemented `CorrelationEmbedding` in `code/embeddings/correlation.py`
- Implemented `DevelopmentalEmbedding` in `code/embeddings/developmental.py`
- Wrote Property 5 test (force direction) in `code/tests/test_developmental_force.py`

### Design Notes
- CorrelationEmbedding subsamples to 5000 weights for MDS tractability
- Interpolates remaining positions via nearest-neighbor in correlation space
- DevelopmentalEmbedding uses vectorized force computation for efficiency
- Force capped at min(|corr|*dist, max_force, 0.49*dist) to prevent overshooting
- `compute_force()` exported as standalone function for testability

### Test Results
- All 6 tests pass (4 property + 2 force direction)
- Force direction property verified with 200 examples per test
