# Critical Review 3: Incorporation Assessment

This document tracks which points from Critical Review 3 were incorporated into the research plan, requirements, and design documents, and where each change was made.

---

## "Risks Worth Taking Seriously" Section

| Point | Status | Where Incorporated |
|-------|--------|-------------------|
| Step 01 is harder than it appears / deserves more than one step | ✅ Incorporated | Step 01 expanded from 3 to 5 experiments, 2 new embedding methods added (adversarial, differentiable), topographic task added. Scope is now a mini-phase. |
| Phase 1 may show weak glial benefits by design | ✅ Acknowledged | No spec change needed — strategic/psychological point. Research plan already notes Phase 2 is worth pursuing regardless. |
| Benchmark selection disfavors the framework | ✅ Incorporated | Topographic sensor task added (Req 19 in requirements.md, TopographicTask class in design.md, added to research/01-spatial-embedding.md). Guiding principle #10 added to overview. |
| Turing instability is a real engineering risk / go/no-go should be explicit | ✅ Incorporated | Explicit go/no-go gate added to research/04-turing-stability.md: if stable regions cannot be found, Steps 02-03 must be redesigned. |
| Timeline implies team or AI-assisted execution | ✅ Acknowledged | Not actionable for spec. Already addressed by parallelization analysis in overview. |

---

## "Structural Gaps" Section

| Gap | Status | Where Incorporated |
|-----|--------|-------------------|
| **Gap 1: Permutation control missing everywhere** | ✅ Incorporated | Added to: Step 01 (adversarial embedding, Req 15), Step 02 (permuted embedding condition 5), Step 03 (Experiment 3.7), guiding principle #7 in overview. Applies to all Steps 02-08. |
| **Gap 2: No mechanistic theory for why spatial coupling helps backprop** | ✅ Incorporated | New document created: research/01b-theoretical-analysis.md. KFAC baseline added to Step 02. Fisher information analysis formalized. |
| **Gap 3: Benchmark choice disfavors framework** | ✅ Incorporated | Topographic task added (see above). |
| **Gap 4: Embedding is fixed during training** | ✅ Incorporated | Temporal quality tracking (Req 17, Experiment 1.4 in Step 01), differentiable embedding (Req 16) as co-adaptive solution. |
| **Gap 5: Astrocyte domain assignment underspecified** | ✅ Incorporated | Experiment 3.6 added to research/03-astrocyte-domains.md (domain alignment with functional modules via gradient clustering). |
| **Gap 6: Steps 05-06 have unspecified core metrics** | ✅ Incorporated | Concrete metric definitions and validation experiment added to research/05-microglia-agents.md. |

---

## "Outside-the-Box Suggestions" Section

| Suggestion | Status | Where Incorporated |
|------------|--------|-------------------|
| **A: Differentiable positions** | ✅ Incorporated | Req 16 in requirements.md, DifferentiableEmbedding class in design.md, Method H in research/01-spatial-embedding.md. |
| **B: Bayesian microglia** | ✅ Incorporated | Added as alternative approach in research/05-microglia-agents.md (Experiment 5.6: Bayesian evidence accumulation). |
| **C: Glial field as implicit meta-learner** | ✅ Incorporated | Added as Experiment 2.5 in research/02-modulation-field.md (meta-learner variant). |
| **D: Spatial coherence as primary outcome metric** | ✅ Incorporated | Req 18 in requirements.md, SpatialCoherence class in design.md, Experiment 1.5 in research/01-spatial-embedding.md. |
| **E: Adversarial embedding baseline** | ✅ Incorporated | Req 15 in requirements.md, AdversarialEmbedding class in design.md, Method G in research/01-spatial-embedding.md. Three-point validation curve formalized. |
| **F: Recast modulation field as structured preconditioning / KFAC baseline** | ✅ Incorporated | KFAC added as condition 6 in Step 02 measurement protocol. Theoretical analysis in research/01b-theoretical-analysis.md covers Fisher information connection. |

---

## "Structural Recommendation" Section

| Point | Status | Where Incorporated |
|-------|--------|-------------------|
| **Step 01b: three-experiment mini-phase as gate before Step 02** | ✅ Incorporated | Formalized as research/01b-theoretical-analysis.md with explicit go/no-go criteria. The three experiments (permutation baseline, spatial coherence, benchmark selection) are in Step 01 but the gate is now explicit: Step 02 does not proceed until Step 01 confirms the three-point curve is monotonic. |

---

## Go/No-Go Gates Added

1. **Step 01 → Step 02 gate**: Three-point validation curve must be monotonic (adversarial hurts, random neutral, good helps). If not, the spatial structure hypothesis is not supported and the modulation field approach needs rethinking.

2. **Step 04 → continued use of modulation field**: If Turing stability analysis cannot identify reliable safe operating regions, Steps 02-03 must be redesigned with explicit damping/clamping mechanisms, not just parameter-tuned.

---

## New Documents Created

- `research/01b-theoretical-analysis.md` — Theoretical analysis of why spatial LR coupling might help backpropagation (Fisher information, preconditioning, regularization framing). Includes go/no-go gate criteria for Step 01 → Step 02 transition.
- `critical-reviews/critical-review-03-accepted.md` — This document

## Documents Modified

- `.kiro/specs/spatial-embedding-experiments/requirements.md` — Requirements 15-19 added, Requirement 11 expanded
- `.kiro/specs/spatial-embedding-experiments/design.md` — New classes (AdversarialEmbedding, DifferentiableEmbedding, SpatialCoherence, TemporalQualityTracker, TopographicTask), new data models, properties 11-15, file outputs updated
- `research/00-research-plan-overview.md` — Guiding principles 7-10, "Amendments from Critical Review 3" section, technology stack updated, dependency graph updated to include Step 01b, Step 01b added to Phase 1 table
- `research/01-spatial-embedding.md` — Methods G (adversarial) and H (differentiable), Experiments 1.4 (temporal quality) and 1.5 (spatial coherence), expanded measurements/deliverables/risks/success criteria, topographic task added
- `research/02-modulation-field.md` — Permuted embedding control (condition 5), KFAC baseline (condition 6), Experiment 2.5 (meta-learner variant), expanded success criteria, expanded Critical Reviews section
- `research/03-astrocyte-domains.md` — Experiments 3.6 (domain alignment) and 3.7 (permuted embedding control), expanded success criteria
- `research/04-turing-stability.md` — Explicit go/no-go gate for modulation field viability with pass/conditional pass/fail criteria and fallback options
- `research/05-microglia-agents.md` — Concrete metric definitions (redundancy_score, is_unique_path) with validation protocol (Experiment 5.0), Experiment 5.6 (Bayesian evidence accumulation), Experiment 5.7 (permuted embedding control), expanded success criteria
