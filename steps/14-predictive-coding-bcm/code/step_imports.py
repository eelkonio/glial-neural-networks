"""Import bridge for Step 12, Step 13, and Step 12b modules.

Since all steps use 'code' as their package name, this module provides
clean access to Step 12, Step 13, and Step 12b classes by temporarily
swapping which 'code' package is active on sys.path.

Usage:
    from code.step_imports import (
        LocalMLP,
        LayerState,
        LocalLearningRule,
        get_fashion_mnist_loaders,
        ThreeFactorRule,
        GlobalRewardThirdFactor,
        CalciumDynamics,
        CalciumConfig,
        DomainAssignment,
        DomainConfig,
        BCMDirectedRule,
        BCMConfig,
    )
"""

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

_steps_dir = Path(__file__).parent.parent.parent
_step12_dir = str(_steps_dir / "12-local-learning-rules")
_step13_dir = str(_steps_dir / "13-astrocyte-gating")
_step12b_dir = str(_steps_dir / "12b-bcm-directed")


@contextmanager
def _step_context(step_dir: str):
    """Temporarily make a different step's 'code' package active."""
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "code" or key.startswith("code."):
            saved_modules[key] = sys.modules.pop(key)
    sys.path.insert(0, step_dir)
    try:
        yield
    finally:
        if step_dir in sys.path:
            sys.path.remove(step_dir)
        for key in list(sys.modules.keys()):
            if key == "code" or key.startswith("code."):
                del sys.modules[key]
        sys.modules.update(saved_modules)


def _import_from_step12(module_path: str, attr: str):
    """Import an attribute from a Step 12 module."""
    with _step_context(_step12_dir):
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)


def _import_from_step13(module_path: str, attr: str):
    """Import an attribute from a Step 13 module."""
    with _step_context(_step13_dir):
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)


def _import_from_step12b(module_path: str, attr: str):
    """Import an attribute from a Step 12b module."""
    with _step_context(_step12b_dir):
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)


# --- Step 12 classes ---
LocalMLP = _import_from_step12("code.network.local_mlp", "LocalMLP")
LayerState = _import_from_step12("code.rules.base", "LayerState")
LocalLearningRule = _import_from_step12("code.rules.base", "LocalLearningRule")
get_fashion_mnist_loaders = _import_from_step12(
    "code.data.fashion_mnist", "get_fashion_mnist_loaders"
)
ThreeFactorRule = _import_from_step12("code.rules.three_factor", "ThreeFactorRule")
GlobalRewardThirdFactor = _import_from_step12(
    "code.rules.three_factor", "GlobalRewardThirdFactor"
)

# --- Step 13 classes ---
CalciumDynamics = _import_from_step13("code.calcium.li_rinzel", "CalciumDynamics")
CalciumConfig = _import_from_step13("code.calcium.config", "CalciumConfig")
DomainAssignment = _import_from_step13("code.domains.assignment", "DomainAssignment")
DomainConfig = _import_from_step13("code.domains.config", "DomainConfig")

# --- Step 12b classes ---
BCMDirectedRule = _import_from_step12b("code.bcm_rule", "BCMDirectedRule")
BCMConfig = _import_from_step12b("code.bcm_config", "BCMConfig")
