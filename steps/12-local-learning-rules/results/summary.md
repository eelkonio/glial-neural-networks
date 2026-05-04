# Local Learning Rules — Quick Experiment Summary

**Configuration**: 10 epochs, seed=[42], batch_size=128

## Final Test Accuracy

| Rule | Accuracy |
|------|----------|
| backprop | 0.8772 |
| forward_forward | 0.1739 |
| hebbian | 0.1000 |
| oja | 0.0930 |
| predictive_coding | 0.1000 |
| three_factor_error | 0.1000 |
| three_factor_random | 0.1000 |
| three_factor_reward | 0.0999 |

## Verification Status

- [ ] Backprop achieves >85% in 10 epochs
- [ ] All output files generated
- [ ] No training errors

*This is a quick verification run. See run_full_experiment.py for the full 50-epoch run.*