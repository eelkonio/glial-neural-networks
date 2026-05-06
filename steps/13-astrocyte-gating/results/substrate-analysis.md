# Substrate Analysis: How Biology Gets Directional Learning from Local Information

## The Biological Answer

In biology, the **direction** of synaptic change (LTP vs LTD — strengthen vs weaken) is determined by a purely local signal: **the level of postsynaptic calcium at the synapse itself.**

This is the key insight we're missing in our implementation.

### The Calcium-Direction Rule (BCM Theory)

The biological learning rule is NOT `Δw = pre × post × gate`. It's:

```
IF postsynaptic_calcium > θ_high:  → LTP (strengthen)
IF θ_low < calcium < θ_high:      → LTD (weaken)  
IF calcium < θ_low:               → no change

Where θ is a sliding threshold that depends on recent activity history.
```

This is the BCM (Bienenstock-Cooper-Munro) theory, confirmed experimentally. Three calcium zones determine direction:
- **High calcium** (strong NMDA activation) → LTP
- **Moderate calcium** (weak NMDA activation) → LTD
- **Low calcium** (below threshold) → no change

The direction comes from the LEVEL of calcium, not from an external error signal. And the calcium level is determined by purely local factors:
- How strongly the presynaptic neuron fired (glutamate release)
- How depolarized the postsynaptic membrane is (Mg²⁺ block removal)
- Whether the astrocyte released D-serine (NMDA co-agonist)

### Where the Astrocyte Fits

The astrocyte doesn't provide direction directly. It **shifts the threshold** and **gates whether the calcium signal can reach the levels needed for LTP**:

1. **D-serine release** → enables full NMDA receptor opening → allows HIGH calcium → LTP possible
2. **Without D-serine** → NMDA partially blocked → only MODERATE calcium → LTD or no change
3. **Astrocyte calcium waves** → coordinate which domains have D-serine available → spatial structure of plasticity

So the biological system is:
```
Direction = f(postsynaptic_calcium_level)
Postsynaptic_calcium = f(pre_activity, post_depolarization, NMDA_state)
NMDA_state = f(glutamate, D-serine_from_astrocyte)
D-serine = f(astrocyte_calcium > threshold)
```

The astrocyte doesn't tell the synapse WHICH direction to change. It determines WHETHER the synapse CAN reach the high-calcium state needed for LTP. Without the astrocyte, synapses are biased toward LTD or no change.

### The Geometric/Chemical Boundary

The spatial structure enters because:
- Each astrocyte domain (~50μm, covering ~100-1000 synapses) shares a single D-serine availability state
- All synapses within a domain are either "LTP-enabled" or "LTP-disabled" simultaneously
- This creates **spatially coherent plasticity regions** — groups of synapses that can strengthen together
- The domain boundary IS the chemical boundary (D-serine diffusion range)

### Heterosynaptic Plasticity: The Lateral Signal

Within an astrocyte domain, there's another mechanism: **heterosynaptic plasticity**. When one synapse is strongly potentiated (high calcium → LTP), nearby synapses within the same domain are DEPRESSED (LTD). This is mediated by:
- The astrocyte detecting the strong potentiation event
- Releasing ATP/adenosine that depresses neighboring synapses
- Creating a contrast: one synapse strengthens, neighbors weaken

This provides a form of **local competition** — within a domain, synapses compete for potentiation. The "winner" (most active synapse) gets LTP, the "losers" get LTD. This is directional and local.

---

## What This Means for Our Implementation

### The Problem with Our Current Approach

Our three-factor rule uses: `Δw = eligibility × gate × lr`

Where eligibility = `pre × post` (always positive for ReLU networks).

This means:
- The update is always in the SAME direction (positive, strengthening)
- The gate can only modulate magnitude (0 to 1)
- There's no mechanism for LTD (weakening)
- There's no competition between synapses

### The Biologically Faithful Fix

Replace the three-factor rule with a **calcium-level-based rule**:

```python
def compute_update(pre, post, astrocyte_d_serine, domain_mean_activity):
    # 1. Compute "postsynaptic calcium" analog
    #    High when both pre and post are active AND D-serine is available
    synapse_calcium = pre * post * (1.0 + astrocyte_d_serine)
    
    # 2. BCM-like direction from calcium level
    #    θ = sliding threshold based on recent domain activity
    theta = domain_mean_activity  # BCM sliding threshold
    
    # 3. Direction determined by calcium relative to threshold
    #    Above theta → LTP (positive update)
    #    Below theta but above zero → LTD (negative update)
    direction = synapse_calcium - theta
    
    # 4. Magnitude from eligibility trace (temporal integration)
    delta_w = direction * eligibility_decay_factor
    
    return delta_w
```

Key differences from our current approach:
1. **Direction comes from calcium level vs threshold** — not from an external signal
2. **The threshold slides** based on domain mean activity (BCM homeostasis)
3. **The astrocyte gates LTP** by controlling D-serine (amplifies calcium)
4. **LTD is natural** — synapses below threshold weaken automatically
5. **Competition emerges** — within a domain, only the most active synapses exceed threshold

### The Heterosynaptic Competition Layer

Add lateral inhibition within domains:

```python
def heterosynaptic_competition(domain_updates, domain_indices):
    # Within each domain, normalize updates so they sum to ~zero
    # This creates competition: some strengthen, others weaken
    for domain in domain_indices:
        domain_mean = updates[domain].mean()
        updates[domain] -= domain_mean  # Zero-center within domain
    return updates
```

This ensures that within each astrocyte domain:
- The most active synapses get positive updates (LTP)
- The least active synapses get negative updates (LTD)
- The domain as a whole doesn't drift (homeostasis)

---

## Proposed Implementation: Step 12b (Directed Substrate)

A new three-factor variant that implements the biological calcium-direction rule:

```
Δw = BCM_direction(synapse_calcium, theta) × astrocyte_gate × eligibility_magnitude

Where:
  synapse_calcium = pre × post × (1 + D_serine_available)
  theta = EMA(domain_mean_activity)  [sliding threshold]
  BCM_direction = synapse_calcium - theta  [signed!]
  astrocyte_gate = calcium_dynamics > d_serine_threshold  [from Step 13]
  eligibility_magnitude = |eligibility_trace|  [unsigned, just magnitude]
```

This gives us:
- **Direction from local information** (calcium level vs threshold)
- **Geometric bounding** (domain-level threshold, domain-level D-serine)
- **Competition** (heterosynaptic normalization within domains)
- **Astrocyte role** (gates whether LTP is possible, sets domain context)

---

## Why This Should Work

1. **Direction is local**: No external error signal needed. The calcium level at each synapse determines its own direction.
2. **Spatial structure matters**: The domain threshold (θ) is shared within a domain, creating coordinated plasticity.
3. **The astrocyte is constitutive**: Without D-serine, synapses can't reach high calcium → no LTP → only LTD. The astrocyte literally enables learning.
4. **Competition solves credit assignment locally**: Within a domain, the most relevant synapses (highest calcium) get LTP, others get LTD. This is a local form of winner-take-all that doesn't need global error.
5. **The sliding threshold provides homeostasis**: Prevents runaway potentiation (the BCM mechanism).

---

## References

- BCM theory: Bienenstock, Cooper & Munro (1982). "Theory for the development of neuron selectivity."
- Calcium zones: "Three Ca²⁺ levels affect plasticity differently: the LTP zone, the LTD zone and no man's land" (PMC2278561)
- Astrocyte D-serine gating: Henneberger et al. (2010). "Long-term potentiation depends on release of D-serine from astrocytes." Nature 463.
- Heterosynaptic plasticity: Gordon et al. (2009). "Astrocyte mediated distributed plasticity at hypothalamic glutamate synapses." Neuron 64(3).
- Calcium stores and polarity: Bhatt et al. (2000). "Calcium stores regulate the polarity and input specificity of synaptic modification." Nature 408.

Content was rephrased for compliance with licensing restrictions.
