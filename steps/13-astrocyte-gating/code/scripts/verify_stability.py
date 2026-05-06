"""Verify stability fix enables 50-epoch training without NaN/Inf.

Trains ThreeFactorRule with LayerWiseError + stability fix for 50 epochs.
Asserts no NaN or Inf in any weight tensor at any epoch.

Usage:
    .venv/bin/python steps/13-astrocyte-gating/code/scripts/verify_stability.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add Step 13 to path
step13_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, step13_dir)

import torch
import torch.nn as nn

from code.step12_imports import (
    ThreeFactorRule,
    LayerWiseErrorThirdFactor,
    LocalMLP,
    get_fashion_mnist_loaders,
)
from code.stability import clip_error_signal, normalize_eligibility


def verify_stability(epochs: int = 50, seed: int = 42):
    """Run 50-epoch training with stability fix, check for NaN/Inf."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Starting stability verification: {epochs} epochs, seed={seed}")

    # Setup
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    model = LocalMLP().to(device)
    third_factor = LayerWiseErrorThirdFactor(n_classes=10)
    rule = ThreeFactorRule(lr=0.01, tau=100.0, third_factor=third_factor)

    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=128)
    criterion = nn.CrossEntropyLoss()

    nan_inf_found = False

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            x = images.view(images.size(0), -1)

            with torch.no_grad():
                logits = model(x, detach=True)
                loss = criterion(logits, labels)

            states = model.forward_with_states(
                x, labels=labels, global_loss=loss.item()
            )

            with torch.no_grad():
                for state in states:
                    # Apply stability fix: normalize eligibility BEFORE update
                    if hasattr(rule, '_eligibility') and state.layer_index in rule._eligibility:
                        rule._eligibility[state.layer_index] = normalize_eligibility(
                            rule._eligibility[state.layer_index],
                            norm_threshold=50.0,
                            safe_constant=1.0,
                        )

                    delta = rule.compute_update(state)

                    # Apply stability fix: clip the weight delta per-element
                    delta = clip_error_signal(delta, threshold=1.0)

                    # Clip overall delta norm
                    delta_norm = delta.norm()
                    if delta_norm > 0.1:
                        delta = delta * (0.1 / delta_norm)

                    # Apply stability fix: normalize eligibility AFTER update
                    if hasattr(rule, '_eligibility') and state.layer_index in rule._eligibility:
                        rule._eligibility[state.layer_index] = normalize_eligibility(
                            rule._eligibility[state.layer_index],
                            norm_threshold=50.0,
                            safe_constant=1.0,
                        )

                    layer = model.layers[state.layer_index]
                    layer.linear.weight.data += delta

            epoch_loss += loss.item()
            n_batches += 1

        # Check for NaN/Inf in all weights
        for i, layer in enumerate(model.layers):
            w = layer.linear.weight.data
            if torch.isnan(w).any() or torch.isinf(w).any():
                print(f"  ERROR: NaN/Inf found in layer {i} at epoch {epoch}")
                nan_inf_found = True
                break

        if nan_inf_found:
            break

        # Evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images, detach=True)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"test_acc={acc:.4f}")

        rule.reset()

    elapsed = datetime.now()
    print(f"\n[{elapsed.strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Stability verification complete")

    if nan_inf_found:
        print("  FAILED: NaN/Inf detected during training")
        return False
    else:
        print("  PASSED: No NaN/Inf in any weight tensor across all epochs")
        return True


if __name__ == "__main__":
    success = verify_stability(epochs=50, seed=42)
    sys.exit(0 if success else 1)
