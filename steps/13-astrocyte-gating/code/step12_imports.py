"""Import bridge for Step 12 and Step 01 modules.

Since all steps use 'code' as their package name, this module provides
clean access to Step 12 and Step 01 classes by temporarily swapping
which 'code' package is active on sys.path.

Usage:
    from code.step12_imports import (
        ThreeFactorRule,
        ThirdFactorInterface,
        LayerState,
        LocalMLP,
        get_fashion_mnist_loaders,
        SpectralEmbedding,
    )
"""

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

_steps_dir = Path(__file__).parent.parent.parent
_step12_dir = str(_steps_dir / "12-local-learning-rules")
_step01_dir = str(_steps_dir / "01-spatial-embedding")
_step13_dir = str(_steps_dir / "13-astrocyte-gating")


@contextmanager
def _step_context(step_dir: str):
    """Temporarily make a different step's 'code' package active."""
    # Save current state
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "code" or key.startswith("code."):
            saved_modules[key] = sys.modules.pop(key)

    # Put the target step at front of path
    sys.path.insert(0, step_dir)
    try:
        yield
    finally:
        # Remove target step from path
        if step_dir in sys.path:
            sys.path.remove(step_dir)
        # Clear any modules loaded from target step
        for key in list(sys.modules.keys()):
            if key == "code" or key.startswith("code."):
                del sys.modules[key]
        # Restore original modules
        sys.modules.update(saved_modules)


def _import_from_step12(module_path: str, attr: str):
    """Import an attribute from a Step 12 module."""
    with _step_context(_step12_dir):
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)


def _import_from_step01(module_path: str, attr: str):
    """Import an attribute from a Step 01 module."""
    with _step_context(_step01_dir):
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)


# --- Lazy-loaded Step 12 classes ---

_cache = {}


def _get_cached(key, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


# Pre-load all needed classes at module level for clean import syntax
ThreeFactorRule = _import_from_step12("code.rules.three_factor", "ThreeFactorRule")
ThirdFactorInterface = _import_from_step12("code.rules.base", "ThirdFactorInterface")
LayerState = _import_from_step12("code.rules.base", "LayerState")
LocalMLP = _import_from_step12("code.network.local_mlp", "LocalMLP")
get_fashion_mnist_loaders = _import_from_step12(
    "code.data.fashion_mnist", "get_fashion_mnist_loaders"
)
LayerWiseErrorThirdFactor = _import_from_step12(
    "code.rules.three_factor", "LayerWiseErrorThirdFactor"
)
RandomNoiseThirdFactor = _import_from_step12(
    "code.rules.three_factor", "RandomNoiseThirdFactor"
)
GlobalRewardThirdFactor = _import_from_step12(
    "code.rules.three_factor", "GlobalRewardThirdFactor"
)

# --- Step 01 classes ---
SpectralEmbedding = _import_from_step01("code.embeddings.spectral", "SpectralEmbedding")
