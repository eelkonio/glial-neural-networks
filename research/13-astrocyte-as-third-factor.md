# Step 13: Astrocyte D-Serine Gating as the Third Factor

```
SIMULATION FIDELITY: Level 1-2 (Transitional)
SIGNAL MODEL: Instantaneous (rate-based) or Temporal (spiking variant)
NETWORK STATE DURING INFERENCE: Evolving (astrocyte calcium changes during processing)
GLIAL INTERACTION WITH SIGNALS: At endpoints (astrocyte gates whether plasticity occurs at synapse)
NOTE: This is the first step where glial state affects the learning rule IN REAL TIME
      rather than between training steps. The astrocyte gate opens/closes based on
      ongoing activity, meaning the network state is no longer static during learning.
      At Level 2+, the gate also affects signal transmission (not just plasticity),
      making it a true real-time modulator of both learning and inference.
```

## The Claim Being Tested

When the astrocyte provides the gating signal (third factor) in a three-factor learning rule, it transforms a weak local rule into a competitive learning algorithm. The astrocyte's calcium-dependent D-serine release determines WHICH synapses are allowed to learn at any given moment, creating spatially and temporally structured plasticity that solves the credit assignment problem locally.

## Why This Matters

This is the most biologically faithful experiment in the entire plan. In biology, astrocytic D-serine is literally required for NMDA-dependent LTP. This step tests whether that biological necessity translates into computational necessity — whether the astrocyte gate is what makes local learning work.

## Experiment 13.1: Implement Astrocyte-Gated Three-Factor Learning

### The Learning Rule

```python
class AstrocyteGatedSynapse:
    """A synapse whose plasticity is gated by astrocyte D-serine release."""
    
    def __init__(self):
        self.weight = 0.0
        self.eligibility = 0.0
        self.tau_elig = 50  # Eligibility trace time constant (steps)
        
    def update(self, pre_activity, post_activity, astrocyte_gate):
        """
        Three-factor update:
          Factor 1: pre-synaptic activity
          Factor 2: post-synaptic activity  
          Factor 3: astrocyte D-serine gate (0 to 1)
        
        The eligibility trace captures Hebbian coincidence.
        The astrocyte gate determines if the trace converts to a weight change.
        """
        # Update eligibility trace (Factors 1 & 2)
        hebbian = pre_activity * post_activity
        self.eligibility = (1 - 1/self.tau_elig) * self.eligibility + hebbian
        
        # Apply gate (Factor 3): only learn if astrocyte permits
        delta_w = self.eligibility * astrocyte_gate * LEARNING_RATE
        
        # Weight update
        self.weight += delta_w
        
        # Decay eligibility after use
        self.eligibility *= (1 - astrocyte_gate * 0.5)
        
        return delta_w
```

### Astrocyte Gate Logic

```python
class AstrocytePlasticityGate:
    """Astrocyte that gates plasticity via D-serine-like signal."""
    
    def __init__(self, domain_synapses, calcium_threshold=0.4):
        self.domain = domain_synapses  # Which synapses this astrocyte controls
        self.ca = 0.1                  # Internal calcium
        self.threshold = calcium_threshold
        self.d_serine = 0.0            # D-serine output (the gate signal)
        
    def sense_activity(self, domain_activations):
        """Detect neural activity in domain (neurotransmitter spillover)."""
        # Calcium rises with activity (glutamate uptake triggers IP3 -> Ca release)
        activity_level = np.mean(np.abs(domain_activations))
        self.ca += 0.1 * (activity_level - self.ca)  # Slow integration
        
    def compute_gate(self):
        """Release D-serine when calcium exceeds threshold."""
        # Threshold-based release (biological: vesicular release is threshold-dependent)
        if self.ca > self.threshold:
            # Graded release above threshold
            self.d_serine = min(1.0, (self.ca - self.threshold) / 0.5)
        else:
            # Below threshold: no D-serine, no plasticity
            self.d_serine *= 0.9  # Slow decay of existing D-serine
        
        return self.d_serine
    
    def get_gate_for_synapses(self):
        """Return gate value for all synapses in domain."""
        gate = self.compute_gate()
        # All synapses in domain get the same gate (domain-level control)
        return {syn: gate for syn in self.domain}
```

### Key Properties of This System

1. **Plasticity is spatially gated**: Only synapses in active astrocyte domains can learn
2. **Plasticity is temporally gated**: Only when calcium exceeds threshold (after sustained activity)
3. **The gate is domain-wide**: All synapses in one astrocyte's territory share the same gate
4. **The gate integrates over time**: Brief activity doesn't trigger learning; sustained activity does
5. **No global error signal**: The astrocyte doesn't know the loss — it only knows local activity

## Experiment 13.2: Does the Gate Solve Credit Assignment?

### The Question

The credit assignment problem: how does a synapse in layer 1 know whether to strengthen or weaken when the error is at the output? Backprop solves this by propagating error backward. Can astrocyte gating solve it differently?

### Hypothesis

The astrocyte gate doesn't solve credit assignment directly. Instead, it solves a different problem: **which synapses should be plastic right now?** By restricting plasticity to synapses in active domains (where the network is currently processing relevant information), it prevents catastrophic interference and focuses learning on the right subset of weights. Combined with a simple Hebbian rule, this may be sufficient.

### Protocol

1. Train a 3-layer network on MNIST with three-factor rule + astrocyte gating
2. Measure: which synapses are allowed to learn at each step? (gate > 0)
3. Compare: does the gate correlate with "useful" weight updates? (updates that reduce loss)
4. Ablation: what happens with random gating? (same sparsity but random selection)

### Expected Result

Astrocyte gating should select synapses for plasticity better than random selection, because it selects based on activity (which correlates with relevance to the current input).

## Experiment 13.3: Astrocyte Gate + Reward Signal

### Implementation

Combine astrocyte gating with a global reward signal (dopamine analog):

```python
def three_factor_with_reward(synapse, pre, post, astrocyte_gate, reward):
    """
    Full three-factor rule:
    - Eligibility set by pre/post correlation (Hebbian)
    - Gated by astrocyte (spatial/temporal selection)
    - Modulated by reward (global success signal)
    """
    # Update eligibility
    synapse.eligibility += pre * post
    synapse.eligibility *= 0.95  # Decay
    
    # Weight change = eligibility * gate * reward
    delta_w = synapse.eligibility * astrocyte_gate * reward * LR
    synapse.weight += delta_w
```

### The Question

Does adding astrocyte gating to a reward-modulated rule improve over reward alone?

### Protocol

Compare:
1. Three-factor with random gate + reward
2. Three-factor with astrocyte gate + reward
3. Three-factor with astrocyte gate + no reward (just gate)
4. Three-factor with astrocyte gate + reward (full system)

### Expected Result

The full system (gate + reward) should outperform either alone:
- Reward alone: all synapses update on reward, causing interference
- Gate alone: synapses update based on activity, but without direction
- Gate + reward: only active, relevant synapses update, and in the right direction

## Experiment 13.4: Comparison to Backprop + Astrocyte Modulation

### The Critical Comparison

This directly tests the central prediction: **are astrocytes MORE beneficial under local rules than under backprop?**

### Protocol

```
Condition A: Backprop, no astrocytes           (Phase 1 baseline)
Condition B: Backprop + astrocyte modulation   (Phase 1, Step 03 result)
Condition C: Three-factor, no astrocytes       (Step 12 baseline)
Condition D: Three-factor + astrocyte gating   (THIS experiment)

Compute:
  Benefit under backprop = B - A
  Benefit under local rule = D - C
  
  If (D - C) > (B - A): astrocytes are MORE beneficial under local rules
  If (D - C) < (B - A): astrocytes are MORE beneficial under backprop
  If (D - C) ≈ (B - A): benefit is learning-rule-independent
```

### Expected Result

Prediction: (D - C) > (B - A). Astrocytes should help local rules more because:
- Under backprop, credit assignment is already solved globally — astrocytes just optimize
- Under local rules, credit assignment is unsolved — astrocyte gating provides essential structure

## Experiment 13.5: Calcium Dynamics Matter

### The Question

Does the specific nonlinear calcium dynamics of the astrocyte (threshold, oscillations, hysteresis) matter, or would a simple activity-threshold gate work equally well?

### Comparison

1. **Full calcium dynamics**: Li-Rinzel model with ER stores, CICR, pumps
2. **Simple threshold**: gate = 1 if activity > threshold, else 0
3. **Linear filter**: gate = exponential moving average of activity
4. **Random gate**: gate = random (same average sparsity)

### Expected Result

If calcium dynamics matter, the full model should outperform the simple threshold. The nonlinear features that might help:
- **Hysteresis**: once activated, stays active even if activity briefly dips (persistence)
- **Oscillations**: periodic gating creates natural learning/consolidation cycles
- **Threshold**: prevents learning from noise (only sustained activity triggers plasticity)

## Success Criteria

- Astrocyte-gated three-factor rule achieves >5% accuracy improvement over ungated three-factor
- Astrocyte gating selects synapses for plasticity better than random (correlation with useful updates > 0.3)
- Benefit under local rules exceeds benefit under backprop (the central prediction)
- Full calcium dynamics outperform simple threshold gate

## Deliverables

- `src/astrocyte_gate.py`: AstrocytePlasticityGate implementation
- `src/gated_three_factor.py`: Three-factor rule with astrocyte gating
- `experiments/gate_credit_assignment.py`: Credit assignment analysis
- `experiments/gate_vs_backprop_benefit.py`: The critical comparison
- `experiments/calcium_dynamics_ablation.py`: Does calcium complexity matter?
- `results/gating_benefit.csv`: Quantified benefit of astrocyte gating
- `results/central_prediction_test.png`: Bar chart comparing benefit under backprop vs. local rules

## Estimated Timeline

5-6 weeks. This is the most important experiment in Phase 2.
