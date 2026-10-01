"""Day 11: evaluator agent. Run: python -m tests.test_day11

The central test plays the checkpoint scenario with a scripted model: the generator writes a `mul` that
only works for the numbers in its own test and claims it is done; the harness's tests pass; the
evaluator catches the bug, its findings go back to the generator, which fixes it; a second review passes.
"""
import json
import os
import subprocess
import tempfile
from unittest import mock

import requests

from days import day_10_agent as agent
from longrun import evaluator, features, session
from tests.test_day7 import check, response, sse, text_chunks, tool_chunks
from tools import execute_tool


def fake_model(script):
    script, seen = list(script), []

    def post(*args, **kwargs):
        seen.append(json.loads(json.dumps(kwargs["json"])))
        return response(sse(script.pop(0)))
    return post, seen


def calls(*items, prefix="c"):
    return tool_chunks([(f"{prefix}{n}", name, args) for n, (name, args) in enumerate(items)])


def verdict(v, summary, issues=()):
    return calls(("submit_verdict", {"verdict": v, "summary": summary,
                                     "issues": [{"problem": p, "evidence": e} for p, e in issues]}), prefix="v")


def test_verdict_tool():
    check("a 'fail' needs at least one reproduced issue",
          "needs at least one issue" in execute_tool("submit_verdict", {"verdict": "fail", "summary": "bad"}))
    evaluator.VERDICT.clear()
    execute_tool("submit_verdict", {"verdict": "pass", "summary": "all good"})
    check("a verdict is recorded", evaluator.VERDICT["verdict"] == "pass")


def test_snapshot_and_diffs():
    ws = tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, "pkg"))
    for rel, text in {"a.py": "x = 1\n", "pkg/b.py": "y = 2\n"}.items():
        with open(os.path.join(ws, rel), "w") as f:
            f.write(text)
    before = evaluator.snapshot(ws)
    with open(os.path.join(ws, "a.py"), "w") as f:
        f.write("x = 999\n")
    os.remove(os.path.join(ws, "pkg/b.py"))
    open(os.path.join(ws, "scratch.py"), "w").close()
    reverted = evaluator.restore(ws, before)
    check("changes made during the review are undone", reverted == ["a.py", "pkg/b.py", "scratch.py"]
          and open(os.path.join(ws, "a.py")).read() == "x = 1\n" and not os.path.exists(os.path.join(ws, "scratch.py")))

    session.ensure_repo(ws)
    session.commit(ws, "base")
    with open(os.path.join(ws, "a.py"), "a") as f:
        f.write("z = 3\n")
    with open(os.path.join(ws, "new.py"), "w") as f:
        f.write("def f():\n    pass\n")
    diff = evaluator.git_diff(ws)
    check("the review diff includes edits and brand-new files", "+z = 3" in diff and "b/new.py" in diff and "+def f():" in diff)

    original = tempfile.mkdtemp()
    with open(os.path.join(original, "a.py"), "w") as f:
        f.write("x = 1\n")
    diff = evaluator.dir_diff(original, ws)
    check("dir_diff compares against a pristine copy", " x = 1\n+z = 3" in diff
          and "--- /dev/null\n+++ b/new.py" in diff and "+++ b/pkg/b.py" in diff, diff)


def project_with_feature():
    ws = tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, session.LOG_DIR))
    os.makedirs(os.path.join(ws, "tests"))
    open(os.path.join(ws, "tests", "__init__.py"), "w").close()
    session.ensure_repo(ws)
    features.save(ws, {"goal": "calculator", "test_command": "python3 -m unittest", "features": [
        {"id": 1, "title": "mul", "description": "calc.mul(a, b) returns a * b for any integers",
         "acceptance": ["mul(3, 4) == 12", "mul(-2, 5) == -10", "mul(0, 7) == 0"],
         "status": "pending", "attempts": 0, "notes": []}]})
    session.commit(ws, "seed")
    return ws


MUL_TEST = "import unittest\nfrom calc import mul\n\n\nclass T(unittest.TestCase):\n    def test_mul(self):\n        self.assertEqual(mul(3, 4), 12)\n"


def test_evaluator_catches_false_claim_and_generator_fixes_it():
    ws = project_with_feature()
    script = [
        # generator: a mul that only works for its own test, claims done
        calls(("write_file", {"path": "calc.py", "content": "def mul(a, b):\n    return 12\n"}),
              ("write_file", {"path": "tests/test_mul.py", "content": MUL_TEST})),
        calls(("update_feature", {"id": 1, "status": "done", "note": "mul works for all integers"})),
        text_chunks("Implemented mul; all tests pass."),
        # evaluator round 1: checks an acceptance case the test skipped, tries to tamper, fails the work
        calls(("bash", {"command": "python3 -c 'from calc import mul; print(mul(-2, 5))'"}), prefix="e"),
        calls(("bash", {"command": "echo 'def mul(a, b): return a * b' > calc.py"}), prefix="e2"),
        verdict("fail", "mul ignores its arguments",
                [("mul(-2, 5) returns 12 instead of -10", "python3 -c 'from calc import mul; print(mul(-2, 5))' printed 12")]),
        text_chunks("Failed: mul returns a constant."),
        # generator gets the feedback in the same context and fixes it
        calls(("read_file", {"path": "calc.py"})),
        calls(("str_replace", {"path": "calc.py", "old_str": "    return 12", "new_str": "    return a * b"})),
        calls(("update_feature", {"id": 1, "status": "done", "note": "fixed: mul now multiplies"})),
        text_chunks("Fixed mul to return a * b."),
        # evaluator round 2: passes
        calls(("bash", {"command": "python3 -c 'from calc import mul; print(mul(-2, 5), mul(0, 7))'"}), prefix="f"),
        verdict("pass", "all acceptance checks verified"),
        text_chunks("Passed."),
    ]
    post, seen = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        outcome = session.work_session(ws, 1, quiet=True)

    check("the evaluator's rejection sent the work back, and the second review passed",
          outcome["reviews"] == ["fail", "pass"] and outcome["status"] == "done", outcome)
    reviews = [p for p in seen if p["messages"][0]["content"].startswith("You are a skeptical QA engineer")]
    first_review = reviews[0]
    check("the evaluator starts with a fresh context (its own system prompt + the task)", len(first_review["messages"]) == 2)
    check("the evaluator sees the spec, the claim and the diff",
          all(s in first_review["messages"][1]["content"] for s in ("mul(-2, 5) == -10", "mul works for all integers", "+    return 12")))
    offered = [t["function"]["name"] for t in first_review["tools"]]
    check("the evaluator has no edit tools", "write_file" not in offered and "str_replace" not in offered
          and "submit_verdict" in offered, offered)
    feedback_request = seen[len(seen) - 1 - [i for i, p in enumerate(reversed(seen))
                                             if "An independent reviewer" in json.dumps(p["messages"][-1:])][0]]
    check("the generator gets the findings with the evidence, in its own ongoing context",
          "mul(-2, 5) returns 12 instead of -10" in feedback_request["messages"][-1]["content"]
          and any("Implemented mul" in (m.get("content") or "") for m in feedback_request["messages"]))
    check("the evaluator's attempt to fix the code itself was reverted before the generator continued",
          open(os.path.join(ws, "calc.py")).read() == "def mul(a, b):\n    return a * b\n")
    progress = open(os.path.join(ws, "progress.md")).read()
    check("progress.md records both review rounds", "Evaluator round 1: fail" in progress
          and "mul(-2, 5) returns 12 instead of -10" in progress and "Evaluator round 2: pass" in progress)


def test_persistent_problems_mark_the_feature_failing():
    ws = project_with_feature()
    fail = verdict("fail", "still wrong", [("mul(-2, 5) is 12", "printed 12")])
    script = [calls(("write_file", {"path": "calc.py", "content": "def mul(a, b):\n    return 12\n"}),
                    ("write_file", {"path": "tests/test_mul.py", "content": MUL_TEST})),
              calls(("update_feature", {"id": 1, "status": "done", "note": "done"})), text_chunks("done")]
    for _ in range(session.MAX_REVIEW_ROUNDS):  # review fails, generator claims again without fixing
        script += [fail, text_chunks("fail"), calls(("update_feature", {"id": 1, "status": "done", "note": "really"})),
                   text_chunks("it works")]
    script += [fail, text_chunks("fail")]
    post, _ = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        outcome = session.work_session(ws, 1, quiet=True)
    check(f"after {session.MAX_REVIEW_ROUNDS} feedback rounds the feature is marked failing, not done",
          outcome["status"] == "failing" and outcome["reviews"] == ["fail"] * (session.MAX_REVIEW_ROUNDS + 1))
    check("the evaluator's findings reach the next session through the notes",
          "evaluator still found problems" in features.load(ws)["features"][0]["notes"][-1])


def test_no_verdict_does_not_block():
    ws = project_with_feature()
    script = [calls(("write_file", {"path": "calc.py", "content": "def mul(a, b):\n    return a * b\n"}),
                    ("write_file", {"path": "tests/test_mul.py", "content": MUL_TEST})),
              calls(("update_feature", {"id": 1, "status": "done", "note": "done"})), text_chunks("done"),
              text_chunks("I looked around but forgot to submit a verdict.")]
    post, _ = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        outcome = session.work_session(ws, 1, quiet=True)
    check("an evaluator that gives no verdict does not block a feature whose tests pass",
          outcome["status"] == "done" and outcome["reviews"] == ["none"])


def test_runner_review_loop():
    from evals import runner
    task = next(t for t in runner.load_tasks() if t["name"] == "pager-off-by-one")
    half_fixed = open(os.path.join(task["dir"], "repo", "pager.py")).read().replace("page * per_page", "(page - 1) * per_page")
    fixed = open(os.path.join(task["dir"], "solution", "pager.py")).read()
    script = [
        calls(("read_file", {"path": "pager.py"})),
        calls(("write_file", {"path": "pager.py", "content": half_fixed})),
        text_chunks("Fixed the first-page bug."),
        verdict("fail", "page_count is still wrong", [("page_count(10, 3) returns 3, expected 4", "python3 -c ... printed 3")]),
        text_chunks("fail"),
        calls(("write_file", {"path": "pager.py", "content": fixed})),
        text_chunks("Fixed page_count too."),
        verdict("pass", "both bugs fixed"),
        text_chunks("pass"),
    ]
    post, _ = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        r = runner.run_trial(agent, task, sandbox_mode="off", use_evaluator=True)
    check("in the eval runner, the evaluator turns a half fix into a pass",
          r["passed"] and r["evaluator_verdicts"] == ["fail", "pass"], (r["evaluator_verdicts"], r["grade_output"][-300:]))


if __name__ == "__main__":
    home = os.getcwd()
    agent.PROVIDER = "openrouter"  # requests are faked
    session.USE_EVALUATOR = True
    try:
        test_verdict_tool()
        test_snapshot_and_diffs()
        test_evaluator_catches_false_claim_and_generator_fixes_it()
        test_persistent_problems_mark_the_feature_failing()
        test_no_verdict_does_not_block()
        test_runner_review_loop()
    finally:
        os.chdir(home)
    print("\nDay 11 tests passed.")
