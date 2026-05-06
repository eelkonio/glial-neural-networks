"""Checkpoint verification script for Tasks 1-4.

Verifies:
1. CalciumDynamics produces bounded calcium with sustained input
2. DomainAssignment partitions 128-unit layers into 8 domains
3. Imports from Step 12 and Step 01 work

Usage:
    .venv/bin/python steps/13-astrocyte-gating/code/scripts/verify_checkpoint.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add Step 13 to path
step13_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, step13_dir)

import torch

from code.calcium.li_rinzel import CalciumDynamics
from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment
from code.domains.config import DomainConfig
from code.step12_imports import (
    ThreeFactorRule,
    ThirdFactorInterface,
    LocalMLP,
    get_fashion_mnist_loaders,
    SpectralEmbedding,
)


def verify_calcium_bounded():
    """Verify CalciumDynamics produces bounded calcium with sustained input."""
    print("\n--- Verifying CalciumDynamics bounded output ---")

    config = CalciumConfig()
    dynamics = CalciumDynamics(n_domains=8, config=config)

    # Sustained moderate activity
    activities = torch.ones(8) * 2.0

    max_ca = 0.0
    min_ca = float('inf')

    for step in range(500):
        ca = dynamics.step(activities)
        max_ca = max(max_ca, ca.max().item())
        min_ca = min(min_ca, ca.min().item())

        # Check bounds at every step
        assert (ca >= 0.0).all(), f"Calcium went negative at step {step}"
        assert (ca <= config.ca_max).all(), f"Calcium exceeded max at step {step}"
        assert (dynamics.h >= 0.0).all(), f"h went negative at step {step}"
        assert (dynamics.h <= 1.0).all(), f"h exceeded 1.0 at step {step}"

    final_ca = dynamics.get_calcium()
    gate_open = dynamics.get_gate_open()

    print(f"  500 steps with activity=2.0:")
    print(f"  Final calcium range: [{final_ca.min().item():.4f}, {final_ca.max().item():.4f}]")
    print(f"  Overall min/max: [{min_ca:.4f}, {max_ca:.4f}]")
    print(f"  Gate open: {gate_open.sum().item()}/{len(gate_open)} domains")
    print(f"  PASSED: All calcium values bounded in [0, {config.ca_max}]")
    return True


def verify_domain_assignment():
    """Verify DomainAssignment partitions 128-unit layers into 8 domains."""
    print("\n--- Verifying DomainAssignment ---")

    config = DomainConfig(domain_size=16, mode="spatial")
    # 4 hidden layers of 128 + output layer of 10
    layer_sizes = [(784, 128), (128, 128), (128, 128), (128, 128), (128, 10)]
    weights = [torch.randn(out, inp) for inp, out in layer_sizes]

    assignment = DomainAssignment(
        layer_sizes=layer_sizes,
        config=config,
        weight_matrices=weights,
    )

    print(f"  Layer sizes: {layer_sizes}")
    print(f"  Domains per layer: {assignment.n_domains_per_layer}")
    print(f"  Total domains: {assignment.total_domains}")

    # Verify 128-unit layers have 8 domains
    for i in range(4):
        n_domains = assignment.n_domains_per_layer[i]
        assert n_domains == 8, f"Layer {i}: expected 8 domains, got {n_domains}"

        domains = assignment.get_domain_indices(i)
        all_neurons = [n for d in domains for n in d]
        assert len(all_neurons) == 128, f"Layer {i}: not all neurons assigned"
        assert len(set(all_neurons)) == 128, f"Layer {i}: duplicate assignments"

    # Output layer: ceil(10/16) = 1 domain
    assert assignment.n_domains_per_layer[4] == 1

    print(f"  PASSED: 128-unit layers correctly partitioned into 8 domains")
    return True


def verify_imports():
    """Verify imports from Step 12 and Step 01 work."""
    print("\n--- Verifying imports ---")

    # Step 12
    model = LocalMLP()
    assert model.hidden_size == 128
    print(f"  LocalMLP: {model.input_size}→{model.hidden_size}→{model.n_classes}")

    rule = ThreeFactorRule(lr=0.01, tau=100.0)
    assert rule.name == "three_factor"
    print(f"  ThreeFactorRule: lr={rule.lr}, tau={rule.tau}")

    # Step 01
    se = SpectralEmbedding()
    assert se.name == "spectral"
    print(f"  SpectralEmbedding: name={se.name}")

    print(f"  PASSED: All imports work correctly")
    return True


if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Starting checkpoint verification (Tasks 1-4)")

    results = []
    results.append(("Calcium bounded", verify_calcium_bounded()))
    results.append(("Domain assignment", verify_domain_assignment()))
    results.append(("Imports", verify_imports()))

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Checkpoint verification complete")
    print(f"\n{'='*50}")
    print("SUMMARY:")
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
    print(f"{'='*50}")

    all_passed = all(p for _, p in results)
    sys.exit(0 if all_passed else 1)
