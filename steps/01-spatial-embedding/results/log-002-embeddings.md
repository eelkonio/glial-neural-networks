# Execution Log 002: Embedding Strategies (Tasks 3-7)

## Task 6: Adversarial and Differentiable Embeddings

**Date**: 2026-05-03  
**Status**: Complete

### Actions
- Implemented `AdversarialEmbedding` in `code/embeddings/adversarial.py`
- Implemented `DifferentiableEmbedding` in `code/embeddings/differentiable.py`
- Wrote Property 11 and 12 tests in `code/tests/test_adversarial_differentiable.py`

### Design Notes
- Adversarial uses MDS on |correlation| (not 1-|correlation|) so correlated weights are far apart
- Differentiable uses sigmoid(param) to guarantee [0,1] range mathematically
- Initial differentiable range is [0.12, 0.88] due to sigmoid of uniform[-2, 2]

---

## Task 7: Checkpoint — All Embeddings Verified

**Date**: 2026-05-03  
**Status**: Complete

### Verification Script
`code/scripts/verify_all_embeddings.py`

### Results

| Embedding | Status | Time | Range |
|-----------|--------|------|-------|
| Linear | PASS | 0.21s | [0.0000, 0.9987] |
| Random | PASS | 0.00s | [0.0000, 1.0000] |
| Spectral | PASS | 0.64s | [0.0000, 1.0000] |
| LayeredClustered | PASS | 0.29s | [0.0000, 1.0000] |
| Correlation | PASS | 0.26s | [0.0000, 1.0000] |
| Developmental | PASS | 0.04s | [0.0000, 1.0000] |
| Adversarial | PASS | 0.38s | [0.0000, 1.0000] |
| Differentiable | PASS | 0.00s | [0.1192, 0.8808] |

### Observations
- sklearn MDS raises FutureWarning about `dissimilarity` parameter deprecation (will change in sklearn 1.10). Non-blocking.
- Differentiable embedding has narrower range [0.12, 0.88] due to sigmoid initialization. This is expected and correct.
- All embeddings produce shape (268800, 3) as required.
- Spectral is the slowest (0.64s) due to eigenvector computation on 1306-node graph.

### All Tests Passing
9 test functions across 3 test files, all passing.
