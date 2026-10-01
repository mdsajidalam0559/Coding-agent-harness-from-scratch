"""Day 11: an independent evaluator agent.

Models grade their own work too generously, so a second agent with a fresh context and a skeptical
prompt checks the generator's work by running it. It can read and run anything but has no edit tools;
anything it changes through bash is reverted afterwards. Its verdict is structured (submit_verdict)
and its issues are sent back to the generator as feedback.
"""
import difflib
import os
import subprocess

from tools import tool

EVALUATOR_TOOLS = ["read_file", "list_dir", "search", "bash", "submit_verdict"]
EVALUATOR_MAX_STEPS = 20
MAX_DIFF_CHARS = 15_000
SKIP = {".git", ".agent", "__pycache__", ".pytest_cache", ".venv", "node_modules"}

SYSTEM_PROMPT = """You are a skeptical QA engineer. Another AI agent says it has finished a task. Your job is to find out whether that is true by checking the work yourself. Do not trust its report or its tests: models routinely claim success for code that does not work.

How to review:
1. Read the changed code.
2. Run the project's tests.
3. Check EACH requirement independently by running the code: small `python -c` commands or scratch scripts that exercise it, including the edge cases the tests might miss (empty or invalid input, boundaries, error handling). For command-line programs, run the actual commands.
4. If a linter is available (e.g. `python -m pyflakes`), run it on the changed files; `python -m py_compile <file>` at least checks syntax.
5. Put scratch files under /tmp only. Do not modify the project's files: you have no edit tools, and anything you change through bash is reverted.

Then call submit_verdict:
- "pass" only if every requirement works.
- "fail" otherwise, with one issue per real problem: what is wrong and the evidence (the exact command you ran, what it printed, and what was expected).
Only report problems you actually reproduced. No style nitpicks, no speculation."""

TASK_PROMPT = """The task the other agent was given:
{spec}

The other agent's report of what it did:
{claim}

Its changes:
{diff}
{tests}
Review the work now."""

FEEDBACK_PROMPT = """An independent reviewer checked your work and found problems:

{issues}

Fix every one of them. Run the tests, and re-check each point yourself. {claim_again}Then reply with an updated summary."""

VERDICT = {}


@tool(
    "Submit your final verdict on the other agent's work. Call it exactly once, at the end of your review.",
    {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "summary": {"type": "string", "description": "One or two sentences on what you checked and found"},
        "issues": {"type": "array", "description": "Required for 'fail': each reproduced problem", "items": {
            "type": "object",
            "properties": {"problem": {"type": "string", "description": "What is wrong"},
                           "evidence": {"type": "string", "description": "Command run, actual output, expected output"}},
            "required": ["problem", "evidence"]}},
    },
    required=["verdict", "summary"],
)
def submit_verdict(verdict, summary, issues=None):
    if verdict not in ("pass", "fail"):
        return "Error: verdict must be 'pass' or 'fail'."
    issues = [i for i in (issues or []) if isinstance(i, dict) and str(i.get("problem", "")).strip()]
    if verdict == "fail" and not issues:
        return "Error: a 'fail' verdict needs at least one issue with its evidence."
    VERDICT.update(verdict=verdict, summary=str(summary), issues=issues)
    return "Verdict recorded. Reply with one line to finish."


def snapshot(workspace):
    """Contents of every project file, to undo anything the evaluator changes."""
    files = {}
    for root, dirs, names in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                files[os.path.relpath(path, workspace)] = f.read()
    return files


def restore(workspace, before):
    """Put the project back as it was; returns the paths that had to be reverted."""
    after = snapshot(workspace)
    reverted = []
    for rel, content in before.items():
        if after.get(rel) != content:
            path = os.path.join(workspace, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(content)
            reverted.append(rel)
    for rel in after.keys() - before.keys():
        os.remove(os.path.join(workspace, rel))
        reverted.append(rel)
    return sorted(reverted)


def git_diff(workspace):
    """Uncommitted changes, including new files (the evaluator must see what was added)."""
    run = lambda *args: subprocess.run(["git", *args], cwd=workspace, capture_output=True, text=True).stdout
    untracked = [f for f in run("ls-files", "--others", "--exclude-standard").splitlines() if f]
    diff = run("diff", "HEAD")
    for rel in untracked:
        try:
            with open(os.path.join(workspace, rel)) as f:
                lines = f.read().splitlines(keepends=True)
        except (UnicodeDecodeError, OSError):
            continue
        diff += "".join(difflib.unified_diff([], lines, "/dev/null", f"b/{rel}"))
    return diff


def dir_diff(original, workspace):
    """Diff of workspace against a pristine copy (for eval tasks, which are not git repos)."""
    old, new = snapshot(original), snapshot(workspace)
    out = []
    for rel in sorted(old.keys() | new.keys()):
        if old.get(rel) == new.get(rel):
            continue
        try:
            a = (old.get(rel) or b"").decode().splitlines(keepends=True)
            b = (new.get(rel) or b"").decode().splitlines(keepends=True)
        except UnicodeDecodeError:
            out.append(f"Binary file {rel} changed\n")
            continue
        out.append("".join(difflib.unified_diff(a, b, f"a/{rel}" if rel in old else "/dev/null",
                                                f"b/{rel}" if rel in new else "/dev/null")))
    return "".join(out)


def format_issues(verdict):
    return "\n".join(f"{n}. {i['problem']}\n   Evidence: {i.get('evidence', '').strip()}"
                     for n, i in enumerate(verdict["issues"], 1))


def run_evaluator(model, workspace, spec, claim, diff, test_command=None, quiet=True, log_file=None, sandbox=None):
    """Review the work in `workspace` with a fresh evaluator agent on `model` (any ModelAdapter).

    The evaluator is its own core Agent: fresh context, read/search/run tools only, its own record of read
    files. It logs into `log_file` (the generator's transcript, so its cost is counted with the run).
    Returns {"verdict": "pass" | "fail" | "none", "summary", "issues", "reverted"}.
    """
    from core.agent import Agent, AgentConfig
    from safety.permissions import Policy

    VERDICT.clear()
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n[... diff truncated; {len(diff) - MAX_DIFF_CHARS} more characters: read the files ...]"
    tests = f"\nThe project's test command: `{test_command}`\n" if test_command else ""
    config = AgentConfig(name="evaluator", tools=EVALUATOR_TOOLS, system_prompt=SYSTEM_PROMPT,
                         max_steps=EVALUATOR_MAX_STEPS, log_label="evaluator")
    evaluator = Agent(config, model, workspace, Policy(workspace, "auto", ask=None),
                      log_file=log_file or os.path.join(workspace, ".agent", "evaluator.jsonl"), sandbox=sandbox)
    before = snapshot(workspace)
    evaluator.log("evaluator_start", {"spec": spec[:2000]})
    try:
        evaluator.run(TASK_PROMPT.format(spec=spec, claim=claim or "(no report)", diff=diff or "(no changes found)",
                                         tests=tests))
    finally:
        reverted = restore(workspace, before)
    result = {"verdict": VERDICT.get("verdict", "none"), "summary": VERDICT.get("summary", ""),
              "issues": VERDICT.get("issues", []), "reverted": reverted}
    evaluator.log("evaluator_verdict", result)
    if not quiet:
        print(f"  🔎 evaluator: {result['verdict']}. {result['summary'][:200]}")
        for issue in result["issues"]:
            print(f"     - {issue['problem'][:160]}")
    return result
