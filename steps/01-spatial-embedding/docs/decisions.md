# Design Decisions Log

This document records design decisions and their rationale as they are made during implementation of Step 01 (Spatial Embedding).

---

## Decision 001: Weight-level embedding for Step 01

**Date**: 2026-05-03  
**Context**: Spatial coordinates can be assigned at the neuron level or the weight level.  
**Decision**: Use weight-level embedding (one position per weight parameter) for Step 01.  
**Rationale**: Weight-level is the most general — different weights on the same neuron can land in different astrocyte domains (biologically accurate). For Level 2+ (signal traversal), we'll refactor to neuron-level positions with derived weight positions.  
**Implications**: The positions array has shape (N_weights, 3) ≈ (203264, 3) for our MLP. Pairwise operations require subsampling.

## Decision 002: Python 3.12 + venv + pyproject.toml

**Date**: 2026-05-03  
**Context**: Need a reproducible Python environment for the research code.  
**Decision**: Use Python 3.12 with a standard venv and pyproject.toml for dependency management.  
**Rationale**: Simpler than poetry for a research project. Avoids lock file overhead. Python 3.12 is stable and well-supported by all dependencies.  
**Implications**: Use `python3.12 -m venv` for environment creation.

## Decision 003: MPS (Metal Performance Shaders) for GPU acceleration

**Date**: 2026-05-03  
**Context**: Running on MacBook Pro M4 Pro with 24GB unified memory.  
**Decision**: Use `torch.device("mps")` for GPU-accelerated training.  
**Rationale**: MPS provides significant speedup for PyTorch operations on Apple Silicon. Unified memory means no CPU-GPU transfer overhead.  
**Implications**: Some PyTorch operations may not be supported on MPS; fall back to CPU for those. Test MPS compatibility early.
