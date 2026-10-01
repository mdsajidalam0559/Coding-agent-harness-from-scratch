"""Day 10: todo tool, feature list, multi-session harness. Run: python -m tests.test_day10

The harness test builds a 2-feature project over 4 sessions with a scripted model. In session 2 the
model claims a feature is done but its code is wrong: the harness's own test run must catch that, and
session 3 must resume the same feature from the files alone.
"""
import json
import os
import subprocess
import tempfile
from unittest import mock

import requests

from days import day_10_agent as agent
from longrun import features, session
from safety.permissions import Policy
from tests.test_day7 import check, response, sse, text_chunks, tool_chunks
from tools import execute_tool, todo


def fake_model(script):
    script, seen = list(script), []

    def post(*args, **kwargs):
        seen.append(json.loads(json.dumps(kwargs["json"])))
        return response(sse(script.pop(0)))
    return post, seen


def calls(*items):
    return tool_chunks([(f"c{n}", name, args) for n, (name, args) in enumerate(items)])


def test_todo():
    todo.reset()
    out = execute_tool("todo_write", {"items": [{"content": "read code", "status": "completed"},
                                                {"content": "fix bug", "status": "in_progress"},
                                                {"content": "run tests", "status": "pending"}]})
    check("todo list renders with progress", out.startswith("Todo list (1/3 done)") and "[>] fix bug" in out, out)
    out = execute_tool("todo_write", {"items": [{"content": "a", "status": "in_progress"},
                                                {"content": "b", "status": "in_progress"}]})
    check("only one item may be in progress", "keep exactly one in progress" in out)
    check("the previous list is kept after a rejected update", len(todo.current()) == 3)
    check("bad statuses are rejected", "use one of" in execute_tool("todo_write", {"items": [{"content": "x", "status": "doing"}]}))


def test_feature_list_rules():
    ws = tempfile.mkdtemp()
    features.STATE.update(workspace=ws, plan={"goal": "g", "test_command": None, "features": []},
                          claims={}, allow_create=False)
    check("the feature list cannot be recreated in a work session",
          "only be created by the initializer" in execute_tool("create_feature_list", {"test_command": "x", "features": []}))
    features.STATE["allow_create"] = True
    out = execute_tool("create_feature_list", {"test_command": "python -m unittest", "features": [
        {"title": "A", "description": "do a", "acceptance": ["a works"]},
        {"title": "B", "description": "do b", "acceptance": ["b works"]}]})
    check("create_feature_list saves numbered pending features", "Saved 2 features" in out
          and [f["status"] for f in features.load(ws)["features"]] == ["pending", "pending"])
    check("update_feature rejects unknown ids", "no feature with id 9" in
          execute_tool("update_feature", {"id": 9, "status": "done", "note": ""}))
    execute_tool("update_feature", {"id": 2, "status": "done", "note": "ok"})
    check("update_feature only records a claim", features.STATE["claims"][2]["status"] == "done"
          and features.load(ws)["features"][1]["status"] == "pending")

    plan = {"features": [{"id": 1, "status": "done"}, {"id": 2, "status": "pending"}, {"id": 3, "status": "failing"}]}
    check("unfinished work is picked before new features", features.pick_next(plan)["id"] == 3)
    plan["features"][2]["status"] = "blocked"
    check("then the next pending feature", features.pick_next(plan)["id"] == 2)

    p = Policy(ws, mode="auto", ask=None, protected=("feature_list.json", "progress.md"))
    check("the agent cannot edit harness-owned files", not p.check("write_file", {"path": "feature_list.json", "content": "{}"})[0])
    check("...but can read them", p.check("read_file", {"path": "feature_list.json"})[0])


CALC_TEST = """import unittest
from calc import {name}


class Test(unittest.TestCase):
    def test_{name}(self):
        self.assertEqual({name}(3, 4), {expected})
"""


def test_multi_session_build():
    ws = tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, session.LOG_DIR))
    session.ensure_repo(ws)
    script = [
        # session 0: initializer sets up tests/ and plans two features
        calls(("write_file", {"path": "tests/__init__.py", "content": ""}),
              ("create_feature_list", {"test_command": "python3 -m unittest", "features": [
                  {"title": "add", "description": "calc.add(a, b)", "acceptance": ["add(3, 4) == 7"]},
                  {"title": "mul", "description": "calc.mul(a, b)", "acceptance": ["mul(3, 4) == 12"]}]})),
        text_chunks("Set up tests/ and planned 2 features."),
        # session 1: feature 1, correct
        calls(("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}),
              ("write_file", {"path": "tests/test_add.py", "content": CALC_TEST.format(name="add", expected=7)})),
        calls(("update_feature", {"id": 1, "status": "done", "note": "add works"})),
        text_chunks("Implemented add with a test."),
        # session 2: feature 2, claims done but mul is wrong
        calls(("read_file", {"path": "calc.py"})),
        calls(("str_replace", {"path": "calc.py", "old_str": "    return a + b\n",
                               "new_str": "    return a + b\n\n\ndef mul(a, b):\n    return a + b\n"}),
              ("write_file", {"path": "tests/test_mul.py", "content": CALC_TEST.format(name="mul", expected=12)})),
        calls(("update_feature", {"id": 2, "status": "done", "note": "mul works"})),
        text_chunks("Implemented mul."),
        # session 3: resumes feature 2 from the files and fixes it
        calls(("read_file", {"path": "calc.py"})),
        calls(("str_replace", {"path": "calc.py", "old_str": "def mul(a, b):\n    return a + b",
                               "new_str": "def mul(a, b):\n    return a * b"})),
        calls(("update_feature", {"id": 2, "status": "done", "note": "fixed mul"})),
        text_chunks("Fixed mul: it was adding."),
    ]
    post, seen = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        session.initialize(ws, "A tiny calculator library", quiet=True)
        outcomes = [session.work_session(ws, n, quiet=True) for n in (1, 2, 3)]
        check("nothing is left after the last feature", session.work_session(ws, 4, quiet=True) is None)

    check("session 1 finished feature 1", outcomes[0]["feature"] == 1 and outcomes[0]["status"] == "done")
    check("a false 'done' claim is caught by the harness's own test run",
          outcomes[1]["feature"] == 2 and outcomes[1]["status"] == "failing" and not outcomes[1]["tests_pass"])
    check("the next session resumes the unfinished feature", outcomes[2]["feature"] == 2 and outcomes[2]["status"] == "done")

    first_requests = [p for p in seen if p["messages"][-1]["role"] == "user" and len(p["messages"]) == 2]
    check("every session starts with a fresh context (system prompt + one message)", len(first_requests) == 4,
          [len(p["messages"]) for p in seen])
    resume = first_requests[3]["messages"][1]["content"]
    check("the resumed session learns about the failure only from files",
          "attempted before" in resume and "FAILING" in resume and "feature 2: mul" in resume)
    check("the resumed session is told the suite starts red", "Test status at the start of this session: FAILING" in resume)
    check("sessions see progress.md and git history", "Session 2" in resume and "session 2: feature 2 mul -> failing" in resume)
    tools_in_session = [t["function"]["name"] for t in first_requests[1]["tools"]]
    check("work sessions get update_feature but not create_feature_list",
          "update_feature" in tools_in_session and "create_feature_list" not in tools_in_session)

    plan = features.load(ws)
    check("feature_list.json records the final state", [f["status"] for f in plan["features"]] == ["done", "done"]
          and plan["features"][1]["attempts"] == 2)
    log = subprocess.run(["git", "log", "--oneline"], cwd=ws, capture_output=True, text=True).stdout.splitlines()
    check("one commit per session", len(log) == 4, log)
    progress = open(os.path.join(ws, "progress.md")).read()
    check("progress.md has an entry per session", all(f"## Session {n}" in progress for n in range(4)))
    check("transcripts are kept out of git", ".agent/" in open(os.path.join(ws, ".gitignore")).read()
          and not any(".agent" in line for line in subprocess.run(["git", "ls-files"], cwd=ws, capture_output=True,
                                                                     text=True).stdout.splitlines()))


def test_blocked_after_max_attempts():
    ws = tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, session.LOG_DIR))
    session.ensure_repo(ws)
    features.save(ws, {"goal": "g", "test_command": "true", "features": [
        {"id": 1, "title": "hard", "description": "d", "acceptance": ["a"], "status": "pending", "attempts": 0, "notes": []}]})
    post, _ = fake_model([text_chunks("I could not figure it out.")] * session.MAX_ATTEMPTS)
    with mock.patch.object(requests, "post", side_effect=post):
        statuses = [session.work_session(ws, n, quiet=True)["status"] for n in range(1, session.MAX_ATTEMPTS + 1)]
    check(f"a feature with no progress is blocked after {session.MAX_ATTEMPTS} sessions",
          statuses == ["in_progress"] * (session.MAX_ATTEMPTS - 1) + ["blocked"], statuses)


def test_parse_test_runs():
    ok = session.parse_test_run("exit code: 0\n--- stderr ---\n...\n------\nRan 4 tests in 0.01s\n\nOK")
    check("unittest pass is parsed", ok["passed"] and ok["count"] == 4)
    bad = session.parse_test_run("exit code: 1\n--- stderr ---\nFAIL: test_mul (tests.test_mul.Test)\n"
                                 "ERROR: test_div (tests.test_div.Test)\nRan 5 tests in 0.1s\nFAILED (failures=1, errors=1)")
    check("unittest failures are named", not bad["passed"] and bad["failing"] ==
          ["test_div (tests.test_div.Test)", "test_mul (tests.test_mul.Test)"], bad["failing"])
    empty = session.parse_test_run("exit code: 5\n--- stderr ---\nRan 0 tests in 0.000s\n\nNO TESTS RAN")
    check("'no tests ran' (exit 5) is its own case", empty["no_tests"] and not empty["passed"]
          and session.describe(empty) == "no tests yet")
    py = session.parse_test_run("exit code: 1\n--- stdout ---\nFAILED tests/test_a.py::test_x - assert 1 == 2\n"
                                "==== 1 failed, 3 passed in 0.2s ====")
    check("pytest output is understood too", py["count"] == 4 and py["failing"] == ["tests/test_a.py::test_x"])
    check("a timeout is reported as such", "TIMED OUT" in session.describe(session.parse_test_run("Command timed out")))


def new_project(ws=None):
    ws = ws or tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, session.LOG_DIR), exist_ok=True)
    session.ensure_repo(ws)
    return ws


def test_initializer_repairs_bad_test_command():
    ws = new_project()
    plan_args = [{"title": "a", "description": "d", "acceptance": ["x"]}]
    script = [calls(("create_feature_list", {"test_command": "python3 -m no_such_test_runner", "features": plan_args})),
              text_chunks("planned"),
              # repair session: fixes the command
              calls(("create_feature_list", {"test_command": "python3 -m unittest", "features": plan_args})),
              text_chunks("fixed the test command")]
    post, seen = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        session.initialize(ws, "g", quiet=True)
    check("a broken test command triggers one repair session", len([p for p in seen if len(p["messages"]) == 2]) == 2)
    check("the repaired command is saved", features.load(ws)["test_command"] == "python3 -m unittest")

    ws = new_project()
    post, _ = fake_model([script[0], script[1], text_chunks("could not fix it")])
    with mock.patch.object(requests, "post", side_effect=post):
        try:
            session.initialize(ws, "g", quiet=True)
            stopped = False
        except SystemExit as e:
            stopped = "still does not work" in str(e)
    check("if the repair fails too, the run stops with a clear message", stopped)


def seeded_project(test_files):
    ws = new_project()
    os.makedirs(os.path.join(ws, "tests"))
    open(os.path.join(ws, "tests", "__init__.py"), "w").close()
    for name, content in test_files.items():
        with open(os.path.join(ws, name), "w") as f:
            f.write(content)
    features.save(ws, {"goal": "g", "test_command": "python3 -m unittest", "features": [
        {"id": 1, "title": "add", "description": "d", "acceptance": ["a"], "status": "done", "attempts": 1, "notes": []},
        {"id": 2, "title": "mul", "description": "d", "acceptance": ["a"], "status": "pending", "attempts": 0, "notes": []}]})
    session.commit(ws, "seed")
    return ws


def test_done_needs_new_tests_and_regressions_are_named():
    ws = seeded_project({"calc.py": "def add(a, b):\n    return a + b\n",
                         "tests/test_add.py": CALC_TEST.format(name="add", expected=7)})
    script = [calls(("read_file", {"path": "calc.py"})),
              calls(("str_replace", {"path": "calc.py", "old_str": "def add(a, b):\n    return a + b\n",
                                     "new_str": "def add(a, b):\n    return 0\n\n\ndef mul(a, b):\n    return a * b\n"})),
              calls(("update_feature", {"id": 2, "status": "done", "note": "mul added"})),
              text_chunks("Added mul (no tests).")]
    post, _ = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        outcome = session.work_session(ws, 1, quiet=True)
    check("breaking an earlier feature is reported as a new failure",
          outcome["status"] == "failing" and outcome["new_failures"] == ["test_add (tests.test_add.Test)"], outcome)
    progress = open(os.path.join(ws, "progress.md")).read()
    check("progress.md names what this session broke", "New failures this session: test_add" in progress)

    ws = seeded_project({"calc.py": "def add(a, b):\n    return a + b\n",
                         "tests/test_add.py": CALC_TEST.format(name="add", expected=7)})
    script = [calls(("read_file", {"path": "calc.py"})),
              calls(("str_replace", {"path": "calc.py", "old_str": "    return a + b\n",
                                     "new_str": "    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"})),
              calls(("update_feature", {"id": 2, "status": "done", "note": "mul added"})),
              text_chunks("Added mul.")]
    post, _ = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        outcome = session.work_session(ws, 1, quiet=True)
    check("'done' without any new test is not accepted", outcome["status"] == "failing"
          and "no tests were added" in features.load(ws)["features"][1]["notes"][-1])


def test_interrupted_session_is_committed_and_stops():
    ws = seeded_project({})
    post, _ = fake_model([calls(("bash", {"command": "sleep 30"}))])
    real = agent.execute_tool

    def ctrl_c_on_bash(name, args):
        if name == "bash" and "sleep" in args.get("command", ""):
            raise KeyboardInterrupt
        return real(name, args)

    with mock.patch.object(requests, "post", side_effect=post), \
            mock.patch.object(agent, "execute_tool", side_effect=ctrl_c_on_bash):
        outcome = session.work_session(ws, 1, quiet=True)
    log = subprocess.run(["git", "log", "--oneline", "-1"], cwd=ws, capture_output=True, text=True).stdout
    check("Ctrl-C ends the session as 'interrupted' and commits the work in progress",
          outcome["status"] == "interrupted" and "interrupted (WIP)" in log, (outcome, log))
    check("the feature is left in progress for next time", features.load(ws)["features"][1]["status"] == "in_progress")


def test_provider_errors_are_not_counted():
    ws = seeded_project({})
    quota = response([], status=401, body={"error": {"message": "Rate limit reached ... tokens per day (TPD)"}})
    post, _ = fake_model([])
    with mock.patch.object(requests, "post", return_value=quota):
        outcome = session.work_session(ws, 1, quiet=True)
    feature = features.load(ws)["features"][1]
    check("a session stopped by the provider (quota, outage) is reported as such",
          outcome["status"] == "api_error", outcome)
    check("...and does not count as an attempt at the feature", feature["attempts"] == 0 and feature["status"] == "pending")
    check("the reason is recorded for the next session", "stopped by the model provider" in feature["notes"][-1]
          and "not counted" in open(os.path.join(ws, "progress.md")).read().lower())


if __name__ == "__main__":
    home = os.getcwd()
    agent.PROVIDER = "openrouter"  # requests are faked; any provider with a key in .env works
    session.USE_EVALUATOR = False  # these scripts have no evaluator turns; Day 11 tests cover the evaluator
    try:
        test_todo()
        test_feature_list_rules()
        test_multi_session_build()
        test_blocked_after_max_attempts()
        test_parse_test_runs()
        test_initializer_repairs_bad_test_command()
        test_done_needs_new_tests_and_regressions_are_named()
        test_interrupted_session_is_committed_and_stops()
        test_provider_errors_are_not_counted()
    finally:
        os.chdir(home)
    print("\nDay 10 tests passed.")
