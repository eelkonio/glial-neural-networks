"""Pytest configuration for Step 13 tests.

Ensures the correct 'code' package is importable by inserting
the step directory at the front of sys.path.

When running Step 13 tests specifically:
  .venv/bin/pytest steps/13-astrocyte-gating/code/tests/ -v
"""

import sys
from pathlib import Path

# Insert step 13 directory at the front of sys.path so that
# 'import code.xxx' resolves to steps/13-astrocyte-gating/code/xxx
_step13_dir = str(Path(__file__).parent.parent.parent)

# Put Step 13 at front of path
if _step13_dir in sys.path:
    sys.path.remove(_step13_dir)
sys.path.insert(0, _step13_dir)
