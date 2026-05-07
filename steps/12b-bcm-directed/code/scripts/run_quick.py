#!/usr/bin/env python3
"""Quick smoke test for BCM-directed learning rule.

Runs 5 epochs with the full BCM condition (D-serine + competition)
to verify the rule produces signed updates and achieves >10% accuracy.

Usage:
    cd steps/12b-bcm-directed
    python -m code.scripts.run_quick
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure step 12b code is importable
_step12b_dir = str(Path(__file__).parent.parent.parent)
if _step12b_dir not in sys.path:
    sys.path.insert(0, _step12b_dir)

import torch
from code.step_imports import LocalMLP, get_fashion_mnist_loaders, DomainConfig, CalciumConfig
from code.bcm_config import BCMConfig
from code.training import setup_bcm_rule, train_epoch, evaluate


def main():
    print("=" * 60)
    print("STEP 12b: BCM-DIRECTED QUICK EXPERIMENT")
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

    # BCM full condition: D-serine + competition
    bcm_config = BCMConfig(
        lr=0.01,
        theta_decay=0.99,
        theta_init=0.1,
        d_serine_boost=1.0,
        competition_strength=1.0,
        clip_delta=1.0,
        use_d_serine=True,
        use_competition=True,
    )
    domain_config = DomainConfig(domain_size=16)
    calcium_config = CalciumConfig()

    rule = setup_bcm_rule(bcm_config, domain_config, calcium_config, layer_sizes, device)

    # Load data
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    # Training loop
    epoch_results = []
    signed_updates_verified = False

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, rule, train_loader, device)

        # Evaluate
        metrics = evaluate(model, test_loader, device)

        # Check for signed updates (on first epoch)
        if epoch == 0:
            # Run one more forward pass to check delta signs
            x_sample = next(iter(train_loader))[0][:32].view(32, -1)
            states = model.forward_with_states(x_sample)
            for state in states:
                delta = rule.compute_update(state)
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
            "train_loss": train_loss,
            "test_accuracy": metrics["test_accuracy"],
            "test_loss": metrics["test_loss"],
            "has_nan": has_nan,
        }
        epoch_results.append(result)

        print(f"  Epoch {epoch}: loss={train_loss:.4f}, "
              f"acc={metrics['test_accuracy']:.4f}, nan={has_nan}")

        # Reset rule between epochs
        rule.reset()

    end_time = time.time()
    end_timestamp = datetime.now(timezone.utc).isoformat()
    duration = end_time - start_time

    # Final accuracy
    final_accuracy = epoch_results[-1]["test_accuracy"]

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Signed updates: {signed_updates_verified}")
    print(f"Duration: {duration:.1f}s")
    print(f"End: {end_timestamp}")

    # Verify requirements
    assert signed_updates_verified, "FAIL: No signed updates detected!"
    print(f"✓ Signed updates verified")

    if final_accuracy > 0.10:
        print(f"✓ Accuracy > 10% ({final_accuracy*100:.2f}%)")
    else:
        print(f"⚠ Accuracy ≤ 10% ({final_accuracy*100:.2f}%) — may need more epochs or tuning")

    # Save results
    results = {
        "experiment": "bcm_directed_quick",
        "condition": "bcm_full",
        "seed": seed,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "duration_seconds": duration,
        "final_accuracy": final_accuracy,
        "signed_updates_verified": signed_updates_verified,
        "epoch_results": epoch_results,
        "config": {
            "bcm": {
                "lr": bcm_config.lr,
                "theta_decay": bcm_config.theta_decay,
                "theta_init": bcm_config.theta_init,
                "d_serine_boost": bcm_config.d_serine_boost,
                "competition_strength": bcm_config.competition_strength,
                "clip_delta": bcm_config.clip_delta,
                "use_d_serine": bcm_config.use_d_serine,
                "use_competition": bcm_config.use_competition,
            },
            "domain": {"domain_size": domain_config.domain_size, "mode": domain_config.mode},
        },
    }

    results_dir = Path(_step12b_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "quick_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
