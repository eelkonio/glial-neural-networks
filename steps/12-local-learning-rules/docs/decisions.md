# Design Decisions — Step 12: Local Learning Rules

## Decision 1: Detached Forward Pass for Locality

**Context:** Local learning rules must not use backpropagation through the full network.

**Decision:** Use `tensor.detach()` between layers during forward pass. Each layer sees its input as a fixed tensor with no gradient history.

**Rationale:** This is the simplest way to enforce locality in PyTorch without modifying autograd internals. The detachment happens at the layer boundary, so per-layer gradients (for forward-forward) still work.

## Decision 2: Protocol-Based Rule Interface

**Context:** Need to support 5+ learning rules with a common interface.

**Decision:** Use Python `Protocol` (structural typing) rather than abstract base classes.

**Rationale:** Protocols allow duck typing — any object with the right methods works. This is more Pythonic and avoids inheritance hierarchies. Step 13's astrocyte gate just needs to implement `compute_signal()`.

## Decision 3: FashionMNIST Over MNIST

**Context:** Need a benchmark that discriminates between methods.

**Decision:** Use FashionMNIST (10 clothing categories) instead of MNIST (digits).

**Rationale:** MNIST saturates too easily — even simple methods get >95%. FashionMNIST provides more headroom to see differences between rules (backprop ~89%, local rules ~60-80%).

## Decision 4: Same Architecture as Phase 1

**Context:** Need to compare with Phase 1 spatial embedding results.

**Decision:** Use 784→128→128→128→128→10 (same as Phase 1's DeeperMLP).

**Rationale:** Enables direct comparison of spatial embedding quality under local vs global learning. The 4 hidden layers provide enough depth to test credit assignment reach.

## Decision 5: Three-Factor Rule as Primary Target

**Context:** Step 13 will add astrocyte gating to one of these rules.

**Decision:** Prioritize three-factor rule implementation and make its third-factor interface maximally pluggable.

**Rationale:** The three-factor rule naturally decomposes into (eligibility trace) × (modulation signal). The astrocyte gate IS the modulation signal. Other rules (FF, PC) have their own internal mechanisms that are harder to augment.
