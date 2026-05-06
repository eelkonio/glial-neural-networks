---
inclusion: auto
---

# Experiment Summary Convention

All experiment summary files (`results/summary.md`) MUST begin with a "General" section that provides high-level insights before the detailed data.

## Required Structure

```markdown
# [Step Name] Experiment Summary

## General

### [Step Name] Results & Insights

[2-3 sentences summarizing the key finding in plain language]

**Key insight**: [The single most important takeaway — what did we learn?]

**[Additional insight label]**: [Any critical finding that affects downstream steps]

**Outcome**: [What decision was made based on these results? What's next?]

---

[... rest of the detailed summary with data tables, plots, etc.]
```

## Purpose

The "General" section serves as a quick-reference for anyone (including future AI agents) reading the summary. It should be understandable without reading the full detailed analysis below it.

## What to Include in "Key insight"

- What mechanism was identified or ruled out
- What hypothesis was confirmed or refuted
- What unexpected finding changes the approach
- What the results mean for the NEXT step (not just this one)

## What to Include in "Outcome"

- Whether a go/no-go gate passed or failed
- What the recommended next action is
- Whether the approach needs redesign

## Applies To

- All `results/summary.md` files in any step directory
- All `results/full-experiment-analysis.md` files
- Any document that reports experimental results
