"""Test suites, one per day. Run from the repo root: python -m tests.run_all, or python -m tests.test_day7."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:  # the packages (tools, core, ...) live in the repo root
    sys.path.insert(0, REPO)

# Every test fakes the HTTP calls, but the agents refuse to start without a key. Dummy keys make the tests
# run without a .env (e.g. in CI), and since load_dotenv never overrides a variable that is already set,
# a forgotten mock fails with 401 instead of spending real quota.
for _key in ("OPENROUTER_KEY", "GROQAPI_KEY", "GEMINIAPI_KEY", "ANTHROPIC_API_KEY"):
    os.environ.setdefault(_key, "test-dummy-key")
