# Design Decisions — Step 12b: BCM-Directed Substrate

## Decision 1: Synapse Calcium as |post| Mean

We use `mean(|post_activation|)` over the batch as the per-neuron "synapse calcium" proxy. This is simpler than computing actual NMDA-receptor calcium influx but captures the key biological insight: more active neurons have higher calcium.

## Decision 2: Domain-Level Theta (Not Per-Neuron)

Theta is tracked per astrocyte domain rather than per neuron. This matches biology (astrocytes integrate over their territory) and reduces state from O(neurons) to O(domains).

## Decision 3: Reuse CalciumDynamics from Step 13

Rather than implementing a new calcium model, we reuse the Li-Rinzel dynamics from Step 13. The D-serine gate threshold is already implemented there.

## Decision 4: Competition Strength as Interpolation

`competition_strength` interpolates between no competition (0) and full zero-centering (1). This enables smooth ablation studies.
