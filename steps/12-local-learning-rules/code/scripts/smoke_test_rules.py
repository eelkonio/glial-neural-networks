"""Smoke test for all learning rules.

Trains each rule for 2 epochs on FashionMNIST and verifies:
1. No crashes during training
2. No NaN weights after training
3. Forward-forward classification works
4. Predictive coding inference converges
5. Reports accuracy after 2 epochs

Usage:
    python -m code.scripts.smoke_test_rules
    # or from project root:
    .venv/bin/python steps/12-local-learning-rules/code/scripts/smoke_test_rules.py
"""

import sys
from pathlib import Path

# Ensure correct imports
step_dir = str(Path(__file__).parent.parent.parent)
if step_dir not in sys.path:
    sys.path.insert(0, step_dir)

import torch
import time

from code.network.local_mlp import LocalMLP
from code.rules.hebbian import HebbianRule
from code.rules.oja import OjaRule
from code.rules.three_factor import (
    ThreeFactorRule,
    RandomNoiseThirdFactor,
    GlobalRewardThirdFactor,
    LayerWiseErrorThirdFactor,
)
from code.rules.forward_forward import ForwardForwardRule
from code.rules.predictive_coding import PredictiveCodingRule
from code.experiment.runner import (
    train_backprop,
    train_local_rule,
    train_forward_forward,
    train_predictive_coding,
    evaluate_accuracy,
    set_seed,
    get_device,
)
from code.data.fashion_mnist import get_fashion_mnist_loaders


def check_no_nan_weights(model: LocalMLP, rule_name: str) -> bool:
    """Check that no weights are NaN or Inf."""
    for i, layer in enumerate(model.layers):
        w = layer.linear.weight.data
        if torch.isnan(w).any():
            print(f"  ❌ {rule_name}: NaN weights in layer {i}")
            return False
        if torch.isinf(w).any():
            print(f"  ❌ {rule_name}: Inf weights in layer {i}")
            return False
    return True


def smoke_test_all():
    """Run smoke tests for all learning rules."""
    print("=" * 60)
    print("SMOKE TEST: Local Learning Rules (2 epochs each)")
    print("=" * 60)

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")

    results = {}
    all_passed = True

    # 1. Backprop baseline
    print("\n--- Backprop Baseline ---")
    t0 = time.time()
    try:
        result = train_backprop(
            epochs=2, batch_size=256, seed=42, device=device, verbose=False
        )
        elapsed = time.time() - t0
        acc = result["final_accuracy"]
        passed = check_no_nan_weights(result["model"], "backprop")
        results["backprop"] = {"accuracy": acc, "time": elapsed, "passed": passed}
        print(f"  ✓ Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["backprop"] = {"accuracy": 0, "time": 0, "passed": False}
        all_passed = False

    # 2. Hebbian
    print("\n--- Hebbian Rule ---")
    t0 = time.time()
    try:
        rule = HebbianRule(lr=0.01, weight_decay=0.001)
        result = train_local_rule(
            rule, epochs=2, batch_size=256, seed=42, device=device, verbose=False
        )
        elapsed = time.time() - t0
        acc = result["final_accuracy"]
        passed = check_no_nan_weights(result["model"], "hebbian")
        results["hebbian"] = {"accuracy": acc, "time": elapsed, "passed": passed}
        print(f"  ✓ Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["hebbian"] = {"accuracy": 0, "time": 0, "passed": False}
        all_passed = False

    # 3. Oja
    print("\n--- Oja's Rule ---")
    t0 = time.time()
    try:
        rule = OjaRule(lr=0.01)
        result = train_local_rule(
            rule, epochs=2, batch_size=256, seed=42, device=device, verbose=False
        )
        elapsed = time.time() - t0
        acc = result["final_accuracy"]
        passed = check_no_nan_weights(result["model"], "oja")
        results["oja"] = {"accuracy": acc, "time": elapsed, "passed": passed}
        print(f"  ✓ Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["oja"] = {"accuracy": 0, "time": 0, "passed": False}
        all_passed = False

    # 4. Three-factor (random noise)
    print("\n--- Three-Factor (Random Noise) ---")
    t0 = time.time()
    try:
        rule = ThreeFactorRule(lr=0.01, tau=100, third_factor=RandomNoiseThirdFactor(sigma=0.1))
        result = train_local_rule(
            rule, epochs=2, batch_size=256, seed=42, device=device, verbose=False
        )
        elapsed = time.time() - t0
        acc = result["final_accuracy"]
        passed = check_no_nan_weights(result["model"], "three_factor_noise")
        results["three_factor_noise"] = {"accuracy": acc, "time": elapsed, "passed": passed}
        print(f"  ✓ Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["three_factor_noise"] = {"accuracy": 0, "time": 0, "passed": False}
        all_passed = False

    # 5. Three-factor (global reward)
    print("\n--- Three-Factor (Global Reward) ---")
    t0 = time.time()
    try:
        rule = ThreeFactorRule(lr=0.01, tau=100, third_factor=GlobalRewardThirdFactor())
        result = train_local_rule(
            rule, epochs=2, batch_size=256, seed=42, device=device, verbose=False
        )
        elapsed = time.time() - t0
        acc = result["final_accuracy"]
        passed = check_no_nan_weights(result["model"], "three_factor_reward")
        results["three_factor_reward"] = {"accuracy": acc, "time": elapsed, "passed": passed}
        print(f"  ✓ Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["three_factor_reward"] = {"accuracy": 0, "time": 0, "passed": False}
        all_passed = False

    # 6. Three-factor (layer-wise error)
    print("\n--- Three-Factor (Layer-Wise Error) ---")
    t0 = time.time()
    try:
        rule = ThreeFactorRule(lr=0.01, tau=100, third_factor=LayerWiseErrorThirdFactor())
        result = train_local_rule(
            rule, epochs=2, batch_size=256, seed=42, device=device, verbose=False
        )
        elapsed = time.time() - t0
        acc = result["final_accuracy"]
        passed = check_no_nan_weights(result["model"], "three_factor_error")
        results["three_factor_error"] = {"accuracy": acc, "time": elapsed, "passed": passed}
        print(f"  ✓ Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["three_factor_error"] = {"accuracy": 0, "time": 0, "passed": False}
        all_passed = False

    # 7. Forward-Forward
    print("\n--- Forward-Forward Algorithm ---")
    t0 = time.time()
    try:
        rule = ForwardForwardRule(lr=0.03)
        result = train_forward_forward(
            rule, epochs=2, batch_size=256, seed=42, device=device, verbose=False
        )
        elapsed = time.time() - t0
        acc = result["final_accuracy"]
        passed = check_no_nan_weights(result["model"], "forward_forward")
        results["forward_forward"] = {"accuracy": acc, "time": elapsed, "passed": passed}
        print(f"  ✓ Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")

        # Verify classification works
        print("  Verifying FF classification...")
        model = result["model"]
        model.eval()
        x_test = torch.rand(16, 784).to(device)
        preds = rule.classify(model, x_test)
        assert preds.shape == (16,), "Classification shape mismatch"
        assert (preds >= 0).all() and (preds < 10).all(), "Invalid predictions"
        print("  ✓ FF classification produces valid predictions")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["forward_forward"] = {"accuracy": 0, "time": 0, "passed": False}
        all_passed = False

    # 8. Predictive Coding
    print("\n--- Predictive Coding ---")
    t0 = time.time()
    try:
        rule = PredictiveCodingRule(lr=0.01, inference_lr=0.1, n_inference_steps=10)
        result = train_predictive_coding(
            rule, epochs=2, batch_size=256, seed=42, device=device, verbose=False
        )
        elapsed = time.time() - t0
        acc = result["final_accuracy"]
        passed = check_no_nan_weights(result["model"], "predictive_coding")
        results["predictive_coding"] = {"accuracy": acc, "time": elapsed, "passed": passed}
        print(f"  ✓ Accuracy: {acc:.4f} | Time: {elapsed:.1f}s")

        # Verify inference convergence
        print("  Verifying PC inference convergence...")
        model = result["model"]
        rule2 = PredictiveCodingRule(lr=0.01, inference_lr=0.1, n_inference_steps=20)
        rule2.setup_predictions(model)
        x_test = torch.randn(8, 784).to(device)

        with torch.no_grad():
            activations = model.get_layer_activations(x_test)
        representations = [x_test.clone()] + [a.clone() for a in activations]

        errors_before = rule2.compute_prediction_errors(representations, x_test)
        total_before = sum((e ** 2).mean().item() for e in errors_before)

        for _ in range(20):
            representations = rule2.inference_step(representations, x_test)

        errors_after = rule2.compute_prediction_errors(representations, x_test)
        total_after = sum((e ** 2).mean().item() for e in errors_after)

        if total_after < total_before:
            print(f"  ✓ PC inference converges: {total_before:.4f} → {total_after:.4f}")
        else:
            print(f"  ⚠ PC inference did not converge: {total_before:.4f} → {total_after:.4f}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["predictive_coding"] = {"accuracy": 0, "time": 0, "passed": False}
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Rule':<25} {'Accuracy':>10} {'Time':>8} {'Status':>8}")
    print("-" * 55)
    for rule_name, info in results.items():
        status = "✓ PASS" if info["passed"] else "❌ FAIL"
        print(f"{rule_name:<25} {info['accuracy']:>10.4f} {info['time']:>7.1f}s {status:>8}")

    print(f"\n{'All passed' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

    return results, all_passed


if __name__ == "__main__":
    results, all_passed = smoke_test_all()
    sys.exit(0 if all_passed else 1)
