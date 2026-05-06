# Tasks: BCM-Directed Substrate (Step 12b)

## Task 1: Project Setup

- [ ] 1.1 Create directory structure: `steps/12b-bcm-directed/{README.md, docs/, code/, data/, results/}`
- [ ] 1.2 Create `code/__init__.py` and `code/tests/__init__.py`
- [ ] 1.3 Create `code/step_imports.py` with imports from Step 12 (LocalMLP, LayerState, LocalLearningRule, get_fashion_mnist_loaders) and Step 13 (CalciumDynamics, CalciumConfig, DomainAssignment, DomainConfig)
- [ ] 1.4 Create `code/bcm_config.py` with BCMConfig dataclass (lr, theta_decay, theta_init, d_serine_boost, competition_strength, clip_delta, use_d_serine, use_competition)
- [ ] 1.5 Verify imports work by running a smoke test that instantiates LocalMLP and DomainAssignment

## Task 2: Implement BCMDirectedRule

- [ ] 2.1 Create `code/bcm_rule.py` with BCMDirectedRule class implementing LocalLearningRule protocol
- [ ] 2.2 Implement `_compute_synapse_calcium(state)` → batch-mean of |post_activation|
- [ ] 2.3 Implement `_apply_d_serine_boost(calcium, layer_index)` → amplify calcium in open-gate domains
- [ ] 2.4 Implement `_update_theta(domain_activities, layer_index)` → EMA update of sliding threshold
- [ ] 2.5 Implement `_apply_heterosynaptic_competition(direction, layer_index)` → zero-center within domains
- [ ] 2.6 Implement `compute_update(state)` → full algorithm (steps 1-9 from design pseudocode)
- [ ] 2.7 Implement `reset()` → clear theta state
- [ ] 2.8 Verify BCMDirectedRule produces signed weight deltas with a manual test (print pos/neg counts)

## Task 3: Property-Based Tests

- [ ] 3.1 Create `code/tests/conftest.py` with Hypothesis strategies (layer_states, domain_configs, bcm_configs, activity_tensors)
- [ ] 3.2 Property 1: BCM Direction is Signed — for varied activations and non-degenerate theta, direction contains both positive and negative values
- [ ] 3.3 Property 2: Theta Slides Toward Mean — verify EMA formula for random activity sequences; verify convergence for constant activity
- [ ] 3.4 Property 3: D-Serine Amplifies Calcium — open-gate neurons get amplified by (1 + boost), closed-gate neurons unchanged
- [ ] 3.5 Property 4: Heterosynaptic Zero-Centering — after competition with strength=1.0, mean direction per domain ≈ 0
- [ ] 3.6 Property 5: Domain Partition Completeness — all neurons assigned, no overlaps, correct domain count
- [ ] 3.7 Property 6: Output Shape Matches Weight Shape — compute_update returns (out_features, in_features)
- [ ] 3.8 Property 7: Delta Norm Bounded — Frobenius norm ≤ clip_delta for all inputs including extreme values
- [ ] 3.9 Property 8: Competition Preserves Relative Order — neuron ordering within domain unchanged after zero-centering
- [ ] 3.10 Property 9: Ablation Independence — with both flags False, output independent of calcium dynamics state
- [ ] 3.11 Property 10: Calcium Dynamics Bounded — calcium ∈ [0, ca_max] and h ∈ [0, 1] after any step sequence

## Task 4: Training Loop

- [ ] 4.1 Create `code/training.py` with `train_epoch(model, rule, train_loader, device)` → mean loss
- [ ] 4.2 Create `evaluate(model, test_loader, device)` → accuracy, loss
- [ ] 4.3 Create helper `setup_bcm_rule(bcm_config, domain_config, calcium_config, layer_sizes)` → configured BCMDirectedRule
- [ ] 4.4 Verify training loop runs 1 epoch without error or NaN

## Task 5: Quick Experiment (Smoke Test)

- [ ] 5.1 Create `code/scripts/run_quick.py` — 5 epochs, bcm_full condition, 1 seed
- [ ] 5.2 Add timestamp logging (start time, end time, duration)
- [ ] 5.3 Verify produces signed updates (both positive and negative weight deltas)
- [ ] 5.4 Verify achieves >10% accuracy after 5 epochs
- [ ] 5.5 Save quick results to `results/quick_results.json`

## Task 6: Full Experiment

- [ ] 6.1 Create `code/experiment.py` with ExperimentCondition definitions for all 5 conditions (bcm_no_astrocyte, bcm_d_serine, bcm_full, three_factor_reward, backprop)
- [ ] 6.2 Create `code/scripts/run_experiment.py` — 50 epochs × 3 seeds × 5 conditions with timestamp logging
- [ ] 6.3 Implement per-epoch metric recording (accuracy, loss) for each condition/seed
- [ ] 6.4 Save full results to `results/full_results.json` with timestamps
- [ ] 6.5 Expected runtime: ~2 hours total

## Task 7: Results Analysis and Summary

- [ ] 7.1 Create `code/scripts/analyze_results.py` — load results, compute mean±std across seeds
- [ ] 7.2 Generate `results/summary.md` with table of final accuracies per condition
- [ ] 7.3 Include ablation analysis: quantify contribution of D-serine and competition components
- [ ] 7.4 Compare against Step 12 baseline (three_factor_reward ~10%) and backprop upper bound
- [ ] 7.5 Document key finding: does BCM direction solve the "always positive eligibility" problem from Step 12?
