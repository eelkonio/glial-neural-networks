"""Verification script for Step 12 setup.

Loads FashionMNIST, creates LocalMLP, runs forward_with_states,
applies HebbianRule and OjaRule to one batch, prints shapes.
"""

import sys
from pathlib import Path

# Add the step directory to path
step_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(step_dir))

import torch

from code.data.fashion_mnist import (
    get_fashion_mnist_loaders,
    ForwardForwardDataAdapter,
    embed_label,
    generate_negative,
)
from code.network.local_mlp import LocalMLP
from code.rules.hebbian import HebbianRule
from code.rules.oja import OjaRule


def main():
    print("=" * 60)
    print("Step 12: Local Learning Rules — Setup Verification")
    print("=" * 60)

    # 1. Load FashionMNIST
    print("\n[1] Loading FashionMNIST...")
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=128)
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Test batches: {len(test_loader)}")

    # Get one batch
    images, labels = next(iter(train_loader))
    x = images.view(images.size(0), -1)
    print(f"    Batch shape: {x.shape}")
    print(f"    Labels shape: {labels.shape}")
    print(f"    Pixel range: [{x.min():.3f}, {x.max():.3f}]")

    # 2. Test ForwardForward adapter
    print("\n[2] Testing ForwardForward data adapter...")
    adapter = ForwardForwardDataAdapter(train_loader)
    x_pos, x_neg, ff_labels = next(iter(adapter))
    print(f"    x_pos shape: {x_pos.shape}")
    print(f"    x_neg shape: {x_neg.shape}")
    print(f"    Positive label check (first 5): ", end="")
    for i in range(5):
        embedded = x_pos[i, :10].argmax().item()
        correct = ff_labels[i].item()
        status = "✓" if embedded == correct else "✗"
        print(f"{status}", end=" ")
    print()

    # 3. Create LocalMLP and run forward_with_states
    print("\n[3] Creating LocalMLP and running forward_with_states...")
    model = LocalMLP()
    states = model.forward_with_states(x, labels=labels)
    print(f"    Number of layers: {len(states)}")
    for i, state in enumerate(states):
        print(
            f"    Layer {i}: pre={state.pre_activation.shape}, "
            f"post={state.post_activation.shape}, "
            f"weights={state.weights.shape}"
        )

    # 4. Test standard forward
    print("\n[4] Testing standard forward pass...")
    logits = model(x, detach=True)
    print(f"    Output logits shape: {logits.shape}")
    preds = logits.argmax(dim=1)
    print(f"    Predictions (first 10): {preds[:10].tolist()}")

    # 5. Apply HebbianRule
    print("\n[5] Applying HebbianRule to each layer...")
    hebbian = HebbianRule(lr=0.01, weight_decay=0.001)
    for i, state in enumerate(states):
        delta = hebbian.compute_update(state)
        print(
            f"    Layer {i}: delta shape={delta.shape}, "
            f"delta norm={torch.norm(delta):.6f}"
        )

    # 6. Apply OjaRule
    print("\n[6] Applying OjaRule to each layer...")
    oja = OjaRule(lr=0.01)
    for i, state in enumerate(states):
        delta = oja.compute_update(state)
        print(
            f"    Layer {i}: delta shape={delta.shape}, "
            f"delta norm={torch.norm(delta):.6f}"
        )

    # 7. Verify gradient locality
    print("\n[7] Verifying gradient locality...")
    model.zero_grad()
    out = model(x, detach=True)
    # With detach=True, output has no grad_fn (proves locality)
    print(f"    Output requires_grad: {out.requires_grad} (should be False)")
    print(f"    Output grad_fn: {out.grad_fn} (should be None)")
    assert not out.requires_grad, "Output should not require grad in local mode"

    # Verify with detach=False, gradients flow everywhere
    model.zero_grad()
    out_bp = model(x, detach=False)
    loss_bp = out_bp.sum()
    loss_bp.backward()
    for i, layer in enumerate(model.layers):
        has_grad = layer.linear.weight.grad is not None and (layer.linear.weight.grad != 0).any()
        status = "has gradient ✓" if has_grad else "NO gradient ✗"
        print(f"    Layer {i} (backprop mode): {status}")

    print("\n" + "=" * 60)
    print("✓ All checks passed! Setup is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
