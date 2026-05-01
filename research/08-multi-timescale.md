# Step 08: Multi-Clock Training (Fast Neural, Slow Glial)

```
SIMULATION FIDELITY: Level 1 (Null-Space)
SIGNAL MODEL: Instantaneous
NETWORK STATE DURING INFERENCE: Static (glial state updates between forward passes, not during)
GLIAL INTERACTION WITH SIGNALS: Learning-only
NOTE: At Level 2+, the "fast clock" becomes the actual simulation timestep (~0.1ms),
      and glial clocks are defined relative to it. The multi-timescale structure becomes
      physically meaningful rather than just a scheduling choice. Signals propagating
      during a glial update cycle encounter a changing environment mid-flight.
```

## The Claim Being Tested

Operating the neural network and glial system on different timescales (fast neural updates, slow glial modulation) creates a coupled dynamical system with better convergence, stability, and generalization properties than either timescale alone. The slow glial clock provides temporal smoothing that prevents the neural network from overfitting to noise in individual batches.

## Why This Matters

The first critical review identified the multi-clock paradigm as "the clearest formalization of the paradigm shift." This experiment tests whether the timescale separation is computationally beneficial or just adds latency.

## Experiment 8.1: Timescale Ratio Sweep

### Implementation

```python
class MultiClockTrainer:
    """Trains neural network with glial system on separate clocks."""
    
    def __init__(self, model, astrocyte_network, microglia_pool, 
                 glial_update_ratio=100):
        self.model = model
        self.astrocytes = astrocyte_network
        self.microglia = microglia_pool
        self.glial_ratio = glial_update_ratio  # 1 glial step per N neural steps
        self.neural_step = 0
        
    def train_step(self, batch):
        """One training step with multi-clock dynamics."""
        # FAST CLOCK: Neural network update (every step)
        loss = self.model.forward(batch)
        gradients = self.model.backward(loss)
        
        # Apply current glial modulation to gradients
        modulated_gradients = gradients * self.astrocytes.get_lr_modulation()
        modulated_gradients *= self.model.get_pruning_mask()  # From microglia
        
        self.model.update_weights(modulated_gradients)
        self.neural_step += 1
        
        # SLOW CLOCK: Glial update (every glial_ratio steps)
        if self.neural_step % self.glial_ratio == 0:
            # Astrocyte update
            activations = self.model.get_activation_stats()
            grad_stats = self.model.get_gradient_stats()
            self.astrocytes.step(activations, grad_stats)
            
            # Microglia update (even slower: every 5 glial steps)
            if self.neural_step % (self.glial_ratio * 5) == 0:
                weight_stats = self.model.compute_weight_stats()
                self.microglia.step(weight_stats)
        
        return loss
```

### Protocol

Sweep glial_update_ratio over [1, 5, 10, 50, 100, 500, 1000]:
- ratio = 1: Glial system updates every neural step (no timescale separation)
- ratio = 100: Glial system updates every 100 neural steps (moderate separation)
- ratio = 1000: Glial system updates every 1000 neural steps (extreme separation)

For each ratio, measure:
- Final test accuracy
- Convergence speed (steps to 95% of final accuracy)
- Training stability (loss variance over last 1000 steps)
- Glial state smoothness (how much does modulation change between glial updates?)
- Computational overhead (wall-clock time per epoch)

### Expected Result

- ratio = 1: Glial system is too reactive, tracks noise, adds overhead without benefit
- ratio = 100-500: Sweet spot where glial system provides stable modulation
- ratio = 1000+: Glial system is too slow, cannot adapt to changing network needs

## Experiment 8.2: What the Slow Clock Provides

### The Question

Is the benefit of the slow clock just temporal averaging (which a simple exponential moving average could provide), or does the nonlinear calcium dynamics add something?

### Comparison

1. **No slow modulation**: Standard Adam (baseline)
2. **EMA-based modulation**: Exponential moving average of gradient statistics, applied as LR multiplier (same temporal smoothing, no calcium dynamics)
3. **Astrocyte slow clock**: Full calcium dynamics with timescale separation
4. **Astrocyte fast clock (ratio=1)**: Calcium dynamics but updated every step

### Expected Result

If calcium dynamics add value beyond simple averaging:
- Astrocyte slow clock > EMA-based modulation (at same effective timescale)
- The difference should be attributable to nonlinear features: oscillations, threshold effects, hysteresis

## Experiment 8.3: Stability Under Perturbation

### The Question

Does the slow glial clock make the system more robust to sudden perturbations (learning rate spikes, data distribution shifts, gradient explosions)?

### Protocol

Train network normally for 5000 steps, then introduce perturbation:

**Perturbation types**:
- Learning rate spike (10x for 100 steps)
- Distribution shift (switch from CIFAR-10 to CIFAR-10-C corrupted)
- Gradient noise injection (add Gaussian noise to gradients for 200 steps)
- Weight perturbation (add random noise to 10% of weights)

**Measure recovery**:
- Steps to return to pre-perturbation performance
- Maximum performance drop during perturbation
- Does the system recover fully or settle at a lower level?

### Expected Result

The slow glial clock should act as a stabilizer:
- It doesn't react to the perturbation immediately (inertia)
- It maintains the pre-perturbation modulation pattern during the disturbance
- This prevents the neural network from making large, noise-driven weight changes
- Recovery should be faster with glial stabilization

## Experiment 8.4: Three-Clock System (Neural + Astrocyte + Microglia)

### Implementation

Test the full three-timescale system:

```
Fast clock (every step):     Neural weight updates
Medium clock (every ~100):   Astrocyte modulation updates
Slow clock (every ~500):     Microglia pruning decisions
```

### The Question

Does adding the third (microglia) timescale provide benefit beyond two timescales?

### Protocol

Compare:
1. One clock: Neural only (Adam)
2. Two clocks: Neural + Astrocyte
3. Three clocks: Neural + Astrocyte + Microglia
4. Three clocks with interaction: Astrocytes signal microglia, microglia signal astrocytes

### Measurement

- Performance improvement from each additional clock
- Whether the clocks interact beneficially (synergy > sum of parts)
- Computational overhead of each additional clock

## Experiment 8.5: Adaptive Clock Rates

### The Question

Should the glial update rate be fixed, or should it adapt to network state?

### Implementation

```python
class AdaptiveClockTrainer(MultiClockTrainer):
    """Adjusts glial update frequency based on network state."""
    
    def compute_update_urgency(self):
        """How urgently does the glial system need to update?"""
        # High urgency: loss is changing rapidly, activations are unusual
        loss_change_rate = abs(self.recent_losses[-1] - self.recent_losses[-10])
        activation_anomaly = self.model.compute_activation_anomaly_score()
        
        urgency = 0.5 * loss_change_rate + 0.5 * activation_anomaly
        return urgency
    
    def get_current_ratio(self):
        """Adaptive ratio: update more frequently when urgency is high."""
        urgency = self.compute_update_urgency()
        # Map urgency [0, 1] to ratio [10, 500]
        ratio = int(500 * (1 - urgency) + 10 * urgency)
        return max(10, min(500, ratio))
```

### Expected Result

Adaptive clocking should:
- Update frequently during early training (high urgency, rapid changes)
- Update infrequently during stable training (low urgency, save compute)
- Spike update frequency after perturbations (detected urgency)

## Success Criteria

- Optimal timescale ratio exists in intermediate range (not 1, not infinity)
- Astrocyte dynamics provide benefit beyond simple temporal averaging (EMA)
- Slow clock improves robustness to perturbations (faster recovery)
- Three-clock system outperforms two-clock system
- Adaptive clocking matches or exceeds fixed-ratio performance with less compute

## Deliverables

- `src/multi_clock_trainer.py`: Multi-timescale training loop
- `src/adaptive_clock.py`: Adaptive update frequency
- `experiments/timescale_sweep.py`: Ratio sweep experiment
- `experiments/perturbation_robustness.py`: Stability under perturbation
- `experiments/three_clock.py`: Full three-timescale system
- `results/timescale_vs_performance.png`: Performance as function of ratio
- `results/perturbation_recovery.png`: Recovery curves with/without glial stabilization

## Estimated Timeline

3 weeks. Builds on all previous infrastructure.
