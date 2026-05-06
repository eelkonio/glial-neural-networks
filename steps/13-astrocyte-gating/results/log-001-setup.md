# Log 001: Project Setup, Stability Fix, Calcium Dynamics, Domain Assignment

## Summary

Completed Tasks 1-5 of Step 13 (Astrocyte D-Serine Gating):
- Project structure created with all directories and modules
- Import bridge to Step 12 and Step 01 working
- Stability fix implemented and verified (50 epochs, no NaN/Inf)
- Li-Rinzel calcium dynamics implemented with bounded output
- Domain assignment implemented with spatial/random modes
- All 24 tests passing

## Task 1: Project Setup

**Timestamp**: 2026-05-06 13:37

### 1.1 Directory Structure
Created `steps/13-astrocyte-gating/` with:
- `docs/` — design decisions
- `code/` — main package
- `code/gates/` — gate variant implementations
- `code/calcium/` — Li-Rinzel dynamics
- `code/domains/` — domain assignment
- `code/experiment/` — experiment infrastructure
- `code/tests/` — test suite
- `code/scripts/` — verification scripts
- `data/` — checkpoints and cached data
- `results/` — experiment outputs

All `__init__.py` files created. README.md and docs/decisions.md written.

### 1.2 Import Bridge
Created `code/step12_imports.py` using context manager pattern to swap
`code` package resolution between steps. This handles the shared package
name conflict cleanly.

Verified imports:
- `ThreeFactorRule`, `ThirdFactorInterface`, `LayerState` from Step 12
- `LocalMLP`, `get_fashion_mnist_loaders` from Step 12
- `SpectralEmbedding` from Step 01

Updated `pyproject.toml` to include Step 13 in testpaths.

## Task 2: Stability Fix

**Timestamp**: 2026-05-06 13:38

### 2.1 Implementation (`code/stability.py`)
- `clip_error_signal(error, threshold=10.0)` — clamps to [-T, T] preserving sign
- `normalize_eligibility(trace, norm_threshold=100.0, safe_constant=1.0)` — rescales when norm exceeds threshold

### 2.2 Property Tests (4 tests, all passing)
- Property 1: No element exceeds threshold after clipping
- Property 1: Sign preserved for all non-zero elements
- Property 2: Norm equals safe_constant when above threshold
- Property 2: Direction (unit vector) preserved after normalization

### 2.3 Stability Verification
Ran 50-epoch training with ThreeFactorRule + LayerWiseError + stability fix:
- **Result**: No NaN/Inf in any weight tensor across all 50 epochs
- Loss remains high (expected — three-factor with layer-wise error doesn't converge)
- Accuracy at 10% (random chance, expected for this baseline)
- Key: numerical stability maintained throughout

Applied stability fix:
- Eligibility trace normalization (threshold=50.0, safe_constant=1.0) before AND after update
- Per-element delta clipping (threshold=1.0)
- Overall delta norm clipping (max 0.1)

## Task 3: Calcium Dynamics

**Timestamp**: 2026-05-06 13:43

### 3.1 CalciumDynamics (`code/calcium/li_rinzel.py`)
Li-Rinzel two-variable model vectorized over N domains:
- State: `ca` (n_domains,), `h` (n_domains,)
- Fluxes: J_channel (CICR), J_pump (SERCA), J_leak (passive)
- IP3 production proportional to domain activity
- Clamped to [0, ca_max] for calcium, [0, 1] for h
- Methods: step(), get_calcium(), get_gate_open(), reset(), state_dict(), load_state_dict()

### 3.2 CalciumConfig (`code/calcium/config.py`)
All parameters with defaults from design doc:
- ip3_production_rate=0.5, d_serine_threshold=0.4
- serca_pump_rate=0.9, er_leak_rate=0.02
- c0=2.0, c1=0.185, a2=0.2, d2=1.049, K_pump=0.1, dt=0.01

### 3.3 Property Tests (5 property + 5 unit tests, all passing)
- Property 3: Calcium bounded [0, ca_max] after single/multi/extreme steps
- Property 4: IP3 monotonically increases with activity
- Unit: Calcium rises with sustained input, decays without input
- Unit: Reset returns to resting state, state_dict roundtrip works

### Calcium Behavior
With sustained activity=2.0 for 500 steps:
- Calcium reaches ~0.022 (below d_serine_threshold of 0.4)
- This means the gate stays closed with moderate activity
- Higher activity or longer stimulation needed to open gates

## Task 4: Domain Assignment

**Timestamp**: 2026-05-06 13:44

### 4.1 DomainAssignment (`code/domains/assignment.py`)
- Partitions output neurons into non-overlapping domains
- "spatial" mode: spectral ordering via first eigenvector of W@W^T
- "random" mode: shuffled assignment for ablation
- Fallback: contiguous index partitioning
- Immutable after initialization

### 4.2 DomainConfig (`code/domains/config.py`)
- domain_size=16, mode="spatial", embedding_path=None, seed=42

### 4.3 Property Tests (4 property + 4 unit tests, all passing)
- Property 5: Correct number of domains, no overlaps, no unassigned
- Property 6: Repeated calls return identical results
- Unit: 128 neurons → 8 domains, multi-layer, random≠spatial, distances correct

## Task 5: Checkpoint Verification

**Timestamp**: 2026-05-06 13:46

All verifications passed:
- ✓ CalciumDynamics: bounded calcium [0.0004, 0.0220] over 500 steps
- ✓ DomainAssignment: 128-unit layers → 8 domains, 10-unit output → 1 domain
- ✓ Imports: LocalMLP, ThreeFactorRule, SpectralEmbedding all instantiate correctly
- ✓ All 24 tests pass

## Test Summary

```
24 passed in 6.62s

test_calcium_dynamics.py     — 10 tests (5 property, 5 unit)
test_domain_assignment.py    — 8 tests (4 property, 4 unit)
test_imports.py              — 2 tests (smoke tests)
test_stability_fix.py        — 4 tests (all property)
```

## Files Created

```
steps/13-astrocyte-gating/
├── README.md
├── docs/decisions.md
├── code/
│   ├── __init__.py
│   ├── stability.py
│   ├── step12_imports.py
│   ├── calcium/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── li_rinzel.py
│   ├── domains/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── assignment.py
│   ├── gates/__init__.py
│   ├── experiment/__init__.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── verify_stability.py
│   │   └── verify_checkpoint.py
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_imports.py
│       ├── test_stability_fix.py
│       ├── test_calcium_dynamics.py
│       └── test_domain_assignment.py
├── data/.gitkeep
└── results/
    ├── .gitkeep
    └── log-001-setup.md
```
