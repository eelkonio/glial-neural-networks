#!/usr/bin/env python3
"""Quick smoke test for Predictive Coding + BCM learning rule.

Runs 5 epochs with the predictive_bcm_full condition (1 seed)
to verify the rule produces signed updates, prediction errors decrease,
and accuracy is above chance.

Usage:
    cd steps/14-predictive-coding-bcm
    python -m code.scripts.run_quick
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

_step14_dir = str(Path(__file__).parent.parent.parent)
if _step14_dir not in sys.path:
    sys.path.insert(0, _step14_dir)

import torch
from code.step_imports import LocalMLP, get_fashion_mnist_loaders, DomainConfig, CalciumConfig
from code.predictive_bcm_config import PredictiveBCMConfig
from code.training import setup_predictive_bcm_rule, train_epoch_predictive, evaluate


def main():
    print("=" * 60)
    print("STEP 14: PREDICTIVE CODING + BCM — QUICK EXPERIMENT")
    print("=" * 60)

    start_time = time.time()
    start_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"Start: {start_timestamp}")

    # Configuration
    seed = 42
    n_epochs = 5
    batch_size = 128
    device = "cpu"

    torch.manual_seed(seed)

    # Setup model
    model = LocalMLP()
    layer_sizes = [(784, 128), (128, 128), (128, 128), (128, 128), (128, 10)]

    # Predictive BCM full condition
    config = PredictiveBCMConfig(
        lr=0.01, lr_pred=0.01,  # Lower prediction LR to prevent explosion
        use_d_serine=True, use_competition=True, use_domain_modulation=True,
        clip_pred_delta=0.1,  # Tighter clipping on prediction weight updates
    )
    domain_config = DomainConfig(domain_size=16)
    calcium_config = CalciumConfig()

    rule = setup_predictive_bcm_rule(config, domain_config, calcium_config, layer_sizes, device)

    # Load data
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    # Training loop
    epoch_results = []
    signed_updates_verified = False

    for epoch in range(n_epochs):
        # Train
        train_result = train_epoch_predictive(model, rule, train_loader, device)

        # Evaluate
        metrics = evaluate(model, test_loader, device)

        # Check for signed updates (on first epoch)
        if epoch == 0:
            x_sample = next(iter(train_loader))[0][:32].view(32, -1)
            states = model.forward_with_states(x_sample)
            deltas = rule.compute_all_updates(states)
            for delta in deltas:
                pos = (delta > 0).sum().item()
                neg = (delta < 0).sum().item()
                if pos > 0 and neg > 0:
                    signed_updates_verified = True
                    break

        # Check for NaN
        has_nan = any(
            torch.isnan(layer.weight.data).any() or torch.isinf(layer.weight.data).any()
            for layer in model.layers
        )

        result = {
            "epoch": epoch,
            "train_loss": train_result["train_loss"],
            "test_accuracy": metrics["test_accuracy"],
            "test_loss": metrics["test_loss"],
            "prediction_errors": train_result["prediction_errors"],
            "has_nan": has_nan,
        }
        epoch_results.append(result)

        pred_err_str = ", ".join(
            f"L{k.split('_')[1]}={v:.4f}"
            for k, v in train_result["prediction_errors"].items()
        )
        print(f"  Epoch {epoch}: loss={train_result['train_loss']:.4f}, "
              f"acc={metrics['test_accuracy']:.4f}, pred_err=[{pred_err_str}], nan={has_nan}")

        # Reset rule between epochs
        rule.reset()

    end_time = time.time()
    end_timestamp = datetime.now(timezone.utc).isoformat()
    duration = end_time - start_time

    # Final accuracy
    final_accuracy = epoch_results[-1]["test_accuracy"]

    # Check prediction error trend
    first_errors = epoch_results[0]["prediction_errors"]
    last_errors = epoch_results[-1]["prediction_errors"]
    pred_errors_decreased = False
    if first_errors and last_errors:
        first_mean = sum(first_errors.values()) / len(first_errors)
        last_mean = sum(last_errors.values()) / len(last_errors)
        pred_errors_decreased = last_mean < first_mean

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Signed updates: {signed_updates_verified}")
    print(f"Prediction errors decreased: {pred_errors_decreased}")
    print(f"Duration: {duration:.1f}s")
    print(f"End: {end_timestamp}")

    # Verify requirements
    if signed_updates_verified:
        print("✓ Signed updates verified")
    else:
        print("⚠ No signed updates detected")

    if final_accuracy > 0.10:
        print(f"✓ Accuracy > 10% ({final_accuracy*100:.2f}%)")
    else:
        print(f"⚠ Accuracy ≤ 10% ({final_accuracy*100:.2f}%) — may need more epochs or tuning")

    if pred_errors_decreased:
        print("✓ Prediction errors decreased over training")
    else:
        print("⚠ Prediction errors did not decrease (may need more epochs)")

    # Compare against forward-forward baseline
    ff_baseline = 0.165  # 16.5% from forward-forward
    if final_accuracy > ff_baseline:
        print(f"✓ Above forward-forward baseline ({ff_baseline*100:.1f}%)")
    else:
        print(f"⚠ Below forward-forward baseline ({ff_baseline*100:.1f}%) — expected with only 5 epochs")

    # Save results
    results = {
        "experiment": "predictive_bcm_quick",
        "condition": "predictive_bcm_full",
        "seed": seed,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "duration_seconds": duration,
        "final_accuracy": final_accuracy,
        "signed_updates_verified": signed_updates_verified,
        "prediction_errors_decreased": pred_errors_decreased,
        "epoch_results": epoch_results,
        "config": {
            "predictive_bcm": {
                "lr": config.lr,
                "lr_pred": config.lr_pred,
                "theta_decay": config.theta_decay,
                "theta_init": config.theta_init,
                "d_serine_boost": config.d_serine_boost,
                "competition_strength": config.competition_strength,
                "clip_delta": config.clip_delta,
                "clip_pred_delta": config.clip_pred_delta,
                "combination_mode": config.combination_mode,
                "use_d_serine": config.use_d_serine,
                "use_competition": config.use_competition,
                "use_domain_modulation": config.use_domain_modulation,
                "granularity": config.granularity,
            },
            "domain": {"domain_size": domain_config.domain_size, "mode": domain_config.mode},
        },
    }

    results_dir = Path(_step14_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "quick_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
