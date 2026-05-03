"""Reproducibility utilities for experiment infrastructure.

Provides seed management, hardware info collection, library version
tracking, and git hash retrieval for experiment metadata logging.
"""

import os
import platform
import random
import subprocess

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value to use across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set MPS seed if available
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.use_deterministic_algorithms(False)


def get_hardware_info() -> dict:
    """Collect hardware information for experiment metadata.

    Returns:
        Dictionary with GPU, CPU, and memory information.
    """
    info: dict = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # GPU info
    if torch.backends.mps.is_available():
        info["device_type"] = "mps"
        info["gpu_name"] = "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
        )
    else:
        info["device_type"] = "cpu"
        info["gpu_name"] = "None"

    return info


def get_library_versions() -> dict:
    """Collect versions of key libraries used in experiments.

    Returns:
        Dictionary mapping library names to version strings.
    """
    versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except ImportError:
        versions["scipy"] = "not installed"

    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except ImportError:
        versions["sklearn"] = "not installed"

    try:
        import hypothesis

        versions["hypothesis"] = hypothesis.__version__
    except ImportError:
        versions["hypothesis"] = "not installed"

    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except ImportError:
        versions["matplotlib"] = "not installed"

    try:
        import pandas

        versions["pandas"] = pandas.__version__
    except ImportError:
        versions["pandas"] = "not installed"

    return versions


def get_git_hash() -> str | None:
    """Get the current git commit hash.

    Returns:
        Short git hash string, or None if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None
