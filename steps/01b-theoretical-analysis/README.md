# Step 01b: Theoretical Analysis — Mechanism Discrimination

## Purpose

Formally characterize WHY spatial LR coupling provides a small benefit
under backpropagation. Step 01 v2 showed the benefit is embedding-independent
(pure regularization). This step quantifies that finding and compares against
known methods (dropout, weight decay, KFAC) to close the loop.

## Experiments

1. **Mechanism discrimination** (7 conditions): spatial coupling vs dropout vs
   weight decay vs KFAC, at multiple batch sizes
2. **Fisher information structure**: does any embedding predict Fisher structure?
3. **Batch size sweep**: does coupling benefit vanish at large batch sizes?

## How to Run

```bash
source .venv/bin/activate
python steps/01b-theoretical-analysis/run_01b.py
```

## Expected Runtime

~60-90 minutes on MPS
