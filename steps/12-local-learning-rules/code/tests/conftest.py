"""Pytest configuration for Step 12 tests.

Ensures the correct 'code' package is importable by inserting
the step directory at the front of sys.path.
"""

import sys
from pathlib import Path

# Insert step 12 directory at the front of sys.path so that
# 'import code.xxx' resolves to steps/12-local-learning-rules/code/xxx
step_dir = str(Path(__file__).parent.parent.parent)
if step_dir not in sys.path:
    sys.path.insert(0, step_dir)
