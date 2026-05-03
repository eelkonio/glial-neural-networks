# Step 01 v2: Spatial Embedding Experiments (Harder Conditions)

## Purpose

Re-run of Step 01 experiments with harder conditions after v1 produced
null results (all quality scores indistinguishable from zero).

## Changes from v1

| Parameter | v1 | v2 | Rationale |
|-----------|----|----|-----------|
| Task | MNIST (97.9% in 10 epochs) | FashionMNIST (~89% in 50 epochs) | Harder task, more room for coupling to help |
| Epochs | 10 | 50 | Longer training for spatial dynamics to differentiate |
| Gradient batches | 5 | 50 | More stable gradient correlation estimates |
| MDS subsample | 500 | 5000 | Better representation of weight space |
| Architecture | 784→256→256→10 | 784→128→128→128→128→10 | Deeper, narrower, harder optimization |
| Quality max_pairs | 100K | 1M | More pairs for reliable quality measurement |

Note: Originally planned CIFAR-10 but the Toronto download server returned 503.
FashionMNIST is a suitable alternative: same image size, harder than MNIST,
reliable download, and the MLP achieves ~89% (not saturated).

## How to Run

```bash
source .venv/bin/activate
python steps/01-spatial-embedding-v2/run_v2.py
```

## Expected Runtime

~1-2 hours (50 epochs × 10 conditions × 3 seeds on MPS)
