"""Day 6: eval harness. Run: python -m tests.test_day6

Uses a fake model (no API calls) that "solves" tasks by writing the reference solution through the
real tools, and skips one task on purpose, to exercise grading, metrics and the report end to end.
"""
import json
import os
import tempfile
from unittest import mock

import requests

from days import day_5_agent as agent
from evals import runner
from evals.report import pass_at_k, pass_hat_k, print_report

SKIPPED_TASK = "pager-off-by-one"


def check(label, condition, detail=""):
    assert condition, f"{label}\n{detail}"
    print(f"✅ {label}")


def test_metrics():
    check("pass@1 = c/n", abs(pass_at_k(4, 1, 1) - 0.25) < 1e-9)
    check("pass@k: one success in 3 trials, k=3 -> 100%", pass_at_k(3, 1, 3) == 1.0)
    check("pass^k: one success in 3 trials, k=3 -> 0%", pass_hat_k(3, 1, 3) == 0.0)
    check("pass^k: all succeed -> 100%", pass_hat_k(3, 3, 3) == 1.0)
    check("pass^2 with 2/4 successes = 1/6", abs(pass_hat_k(4, 2, 2) - 1 / 6) < 1e-9)
    check("k larger than n is undefined", pass_at_k(2, 1, 3) is None)


def fake_model_for(task):
    """A scripted 'model': read each solution file, write it, then say done."""
    solution_dir = os.path.join(task["dir"], "solution")
    solution = {}
    for root, _, names in os.walk(solution_dir):
        for name in names:
            full = os.path.join(root, name)
            solution[os.path.relpath(full, solution_dir)] = open(full).read()

    def calls(kind):
        out = []
        for i, (path, content) in enumerate(solution.items()):
            args = {"path": path} if kind == "read_file" else {"path": path, "content": content}
            out.append({"id": f"{kind}{i}", "type": "function",
                        "function": {"name": kind, "arguments": json.dumps(args)}})
        return out

    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.001}
    script = [] if task["name"] == SKIPPED_TASK else [calls("read_file"), calls("write_file")]
    bodies = [{"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": c}}], "usage": usage} for c in script]
    bodies.append({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
                   "usage": usage})
    return [mock.Mock(status_code=200, headers={}, **{"json.return_value": b}) for b in bodies]


def test_runner_end_to_end():
    tasks = runner.load_tasks()
    check("8 tasks load", len(tasks) == 8, [t["name"] for t in tasks])
    results_path = os.path.join(tempfile.mkdtemp(), "fake.jsonl")
    with open(results_path, "w") as out:
        for task in tasks:
            with mock.patch.object(requests, "post", side_effect=fake_model_for(task)):
                r = runner.run_trial(agent, task, sandbox_mode="off")
            expected = task["name"] != SKIPPED_TASK
            check(f"{task['name']}: graded {'pass' if expected else 'fail'}", r["passed"] == expected, r["grade_output"])
            out.write(json.dumps({"agent": "day_5_agent", "model": "fake", "edit_format": "str_replace",
                                  "sandbox": "off", "label": "fake", "task": task["name"], "trial": 1, **r}) + "\n")

    rows = [json.loads(line) for line in open(results_path)]
    solved = next(r for r in rows if r["task"] == "cli-json-flag")
    check("metrics: steps counted", solved["steps"] == 3, solved)
    check("metrics: tokens and cost summed", solved["prompt_tokens"] == 3000 and abs(solved["cost"] - 0.003) < 1e-9)
    check("metrics: tool calls counted", solved["tool_calls"] == 2)
    check("transcript kept outside the workspace",
          not os.path.exists(os.path.join(solved["trial_dir"], "work", "transcript.jsonl"))
          and os.path.exists(os.path.join(solved["trial_dir"], "transcript.jsonl")))
    print()
    print_report([results_path], k=1)


if __name__ == "__main__":
    test_metrics()
    test_runner_end_to_end()
    print("Day 6 harness tests passed.")
