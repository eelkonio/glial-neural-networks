# Deficiency Analysis Report

## Summary

This report characterizes what each local learning rule lacks compared to backpropagation, identifying specific deficiencies that the astrocyte gate (Step 13) should address.

## forward_forward

### Credit Assignment Reach

- Layer 0: 0.000
- Layer 1: 0.000
- Layer 2: 0.000
- Layer 3: 0.000
- Layer 4: 0.000

### Representation Redundancy
- Layer 0: 0.000
- Layer 1: 0.004
- Layer 2: 0.000
- Layer 3: 0.003
- Layer 4: -0.067

### Inter-Layer Coordination (CKA)
- Layers 0-1: 0.176
- Layers 1-2: 0.853
- Layers 2-3: 0.736
- Layers 3-4: 0.995

### Assessment
- **Dominant deficiency**: credit_assignment
- **Recommended intervention**: Third-factor signal carrying error information to early layers

## hebbian

### Credit Assignment Reach

- Layer 0: 0.051
- Layer 1: 0.005
- Layer 2: -0.019
- Layer 3: -0.312
- Layer 4: 0.031

### Representation Redundancy
- Layer 0: 0.504
- Layer 1: 0.609
- Layer 2: 0.561
- Layer 3: 0.597
- Layer 4: -0.111

### Inter-Layer Coordination (CKA)
- Layers 0-1: 1.000
- Layers 1-2: 1.000
- Layers 2-3: 0.000
- Layers 3-4: nan

### Assessment
- **Dominant deficiency**: credit_assignment
- **Recommended intervention**: Third-factor signal carrying error information to early layers

## oja

### Credit Assignment Reach

- Layer 0: 0.540
- Layer 1: -0.615
- Layer 2: 0.655
- Layer 3: -0.464
- Layer 4: 0.652

### Representation Redundancy
- Layer 0: 0.530
- Layer 1: 0.074
- Layer 2: 0.287
- Layer 3: 0.316
- Layer 4: 0.014

### Inter-Layer Coordination (CKA)
- Layers 0-1: 0.990
- Layers 1-2: 0.999
- Layers 2-3: 0.999
- Layers 3-4: 0.997

### Assessment
- **Dominant deficiency**: coordination
- **Recommended intervention**: Inter-layer synchronization signal

## predictive_coding

### Credit Assignment Reach

- Layer 0: 0.000
- Layer 1: 0.000
- Layer 2: 0.000
- Layer 3: 0.000
- Layer 4: 0.000

### Representation Redundancy
- Layer 0: 0.263
- Layer 1: 0.498
- Layer 2: 0.336
- Layer 3: 0.532
- Layer 4: 0.279

### Inter-Layer Coordination (CKA)
- Layers 0-1: 0.881
- Layers 1-2: 0.959
- Layers 2-3: 0.912
- Layers 3-4: 1.000

### Assessment
- **Dominant deficiency**: credit_assignment
- **Recommended intervention**: Third-factor signal carrying error information to early layers

## three_factor_error

### Credit Assignment Reach

- Layer 0: 0.000
- Layer 1: 0.000
- Layer 2: 0.000
- Layer 3: 0.000
- Layer 4: 0.000

### Representation Redundancy
- Layer 0: 0.398
- Layer 1: 0.264
- Layer 2: 0.360
- Layer 3: 0.323
- Layer 4: nan

### Inter-Layer Coordination (CKA)
- Layers 0-1: 1.000
- Layers 1-2: nan
- Layers 2-3: nan
- Layers 3-4: nan

### Assessment
- **Dominant deficiency**: credit_assignment
- **Recommended intervention**: Third-factor signal carrying error information to early layers

## three_factor_random

### Credit Assignment Reach

- Layer 0: 0.000
- Layer 1: 0.000
- Layer 2: 0.000
- Layer 3: 0.000
- Layer 4: -0.775

### Representation Redundancy
- Layer 0: 0.043
- Layer 1: 0.000
- Layer 2: 0.000
- Layer 3: 0.225
- Layer 4: -0.111

### Inter-Layer Coordination (CKA)
- Layers 0-1: 1.000
- Layers 1-2: 0.000
- Layers 2-3: 0.000
- Layers 3-4: 0.000

### Assessment
- **Dominant deficiency**: credit_assignment
- **Recommended intervention**: Third-factor signal carrying error information to early layers

## three_factor_reward

### Credit Assignment Reach

- Layer 0: 0.000
- Layer 1: 0.000
- Layer 2: 0.000
- Layer 3: 0.000
- Layer 4: 0.000

### Representation Redundancy
- Layer 0: 0.208
- Layer 1: 0.254
- Layer 2: 0.297
- Layer 3: 0.220
- Layer 4: -0.068

### Inter-Layer Coordination (CKA)
- Layers 0-1: 0.881
- Layers 1-2: 0.907
- Layers 2-3: 0.850
- Layers 3-4: 0.678

### Assessment
- **Dominant deficiency**: credit_assignment
- **Recommended intervention**: Third-factor signal carrying error information to early layers
