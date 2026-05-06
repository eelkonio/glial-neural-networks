---
inclusion: auto
---

# Experiment Script Convention: Timestamps

All experiment scripts MUST print timestamps between training runs so that elapsed time per condition is visible in the log output.

## Required Pattern

```python
from datetime import datetime

# Before each condition:
print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting: {rule_name} seed={seed}")

# After each condition:
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed: {rule_name} seed={seed} ({elapsed:.1f}s)")
```

## Applies To

- All `run_full_experiment.py` scripts
- All `run_quick_experiment.py` scripts
- Any script that trains multiple conditions sequentially

## Rationale

Makes it easy to see how long each condition took when reviewing log files after the fact, without needing to parse wall_clock_seconds from CSV output.
