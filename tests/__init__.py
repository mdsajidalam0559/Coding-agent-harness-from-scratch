"""Test suites, one per day. Run from the repo root: python -m tests.run_all, or python -m tests.test_day7."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:  # the packages (tools, core, ...) live in the repo root
    sys.path.insert(0, REPO)
