"""Day 9: history validator, compaction, subagents. Run: python -m tests.test_day9"""
import json
import os
import random
import tempfile
from unittest import mock

import requests

from days import day_9_agent as agent
from context.compaction import SUMMARY_PREFIX, compact, split_units, validate_history
from tests.test_day7 import check, response, sse, text_chunks, tool_chunks
from tools import files


def call(i):
    return {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{i}"})}}


def random_history(rng, turns):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "the original task"}]
    n = 0
    for _ in range(turns):
        kind = rng.random()
        if kind < 0.6:
            calls = [call(n + k) for k in range(rng.randint(1, 3))]
            n += len(calls)
            msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
            msgs += [{"role": "tool", "tool_call_id": c["id"], "content": "x" * rng.randint(10, 3000)} for c in calls]
        elif kind < 0.8:
            msgs.append({"role": "assistant", "content": "thinking " * rng.randint(1, 50)})
        else:
            msgs.append({"role": "user", "content": "more please"})
    return msgs


def test_validator():
    good = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [call(1), call(2)]},
            {"role": "tool", "tool_call_id": "c1", "content": "a"}, {"role": "tool", "tool_call_id": "c2", "content": "b"}]
    check("a valid history has no problems", validate_history(good) == [])
    check("a missing tool result is caught", validate_history(good[:3]))
    check("an orphan tool result is caught", validate_history([good[0], good[2]]))
    check("a tool result after a user message is caught",
          validate_history(good[:2] + [{"role": "user", "content": "?"}] + good[2:]))
    units = split_units(good)
    check("a tool call and its results form one unit", len(units) == 2 and len(units[1]) == 3)


def test_compaction_never_splits_pairs():
    rng = random.Random(7)
    for trial in range(300):
        msgs = random_history(rng, rng.randint(3, 40))
        new, info = compact(msgs, summarize=lambda text: "SUMMARY", keep_recent_tokens=rng.randint(50, 3000))
        assert validate_history(new) == [], (trial, validate_history(new))
        assert new[0]["role"] == "system" and new[1]["content"] == "the original task", trial
        if info["compacted"]:
            assert new[2]["content"].startswith(SUMMARY_PREFIX), trial
    check("300 random histories: compaction always keeps tool calls with their results", True)

    msgs = random_history(random.Random(1), 30)
    same, info = compact(msgs, summarize=lambda text: None, keep_recent_tokens=100)
    check("if summarizing fails, nothing is dropped", same is msgs and not info["compacted"])


def fake_model(script):
    """requests.post replacement: summarizer requests (no tools) get a summary; others follow the script."""
    script = list(script)
    seen = []

    def post(*args, **kwargs):
        payload = json.loads(json.dumps(kwargs["json"]))  # snapshot: the agent keeps mutating its history
        seen.append(payload)
        if "tools" not in payload:
            return response(sse(text_chunks("SUMMARY: read f0..f9; next: answer")))
        return response(sse(script.pop(0)))
    return post, seen


def reset(workdir):
    os.chdir(workdir)
    agent.messages.clear()
    agent.turn_count = 0
    files._read_state.clear()
    agent.policy = agent.Policy(workdir, mode="auto", ask=None)
    agent.log_file = os.path.join(tempfile.mkdtemp(), "t.jsonl")


def events():
    return [json.loads(line) for line in open(agent.log_file)]


def test_compaction_in_the_loop():
    workdir = tempfile.mkdtemp()
    for i in range(10):
        with open(os.path.join(workdir, f"f{i}.txt"), "w") as f:
            f.write(f"file {i}\n" + ("z" * 60 + "\n") * 60)  # ~4k characters in normal-length lines
    reset(workdir)
    script = [tool_chunks([(f"r{i}", "read_file", {"path": f"f{i}.txt"})]) for i in range(10)] + [text_chunks("all read")]
    post, seen = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post), \
            mock.patch.object(agent, "context_window", return_value=12_000):
        agent.agentic_loop("read all ten files")
    compactions = [e["data"] for e in events() if e["type"] == "compaction" and e["data"]["compacted"]]
    check("a long task triggers compaction near the limit", len(compactions) >= 1, compactions)
    check("the history stays valid after compaction", validate_history(agent.messages) == [])
    check("the original request is kept verbatim", agent.messages[1]["content"] == "read all ten files")
    check("a summary replaced the old turns", any(str(m.get("content", "")).startswith(SUMMARY_PREFIX)
                                                  for m in agent.messages))
    check("the task still finishes", agent.messages[-1]["content"] == "all read")
    tool_requests = [p for p in seen if "tools" in p]
    check("every request sent to the API was valid", all(validate_history(p["messages"]) == [] for p in tool_requests))


def test_subagent():
    workdir = tempfile.mkdtemp()
    with open(os.path.join(workdir, "config.py"), "w") as f:
        f.write("TIMEOUT = 30\n")
    reset(workdir)
    script = [
        tool_chunks([("d1", "delegate", {"task": "Find where TIMEOUT is set and report its value."})]),  # main
        tool_chunks([("s1", "read_file", {"path": "config.py"})]),                                      # sub
        text_chunks("REPORT: TIMEOUT = 30 in config.py line 1"),                                          # sub
        text_chunks("The timeout is 30 seconds."),                                                        # main
    ]
    post, seen = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        agent.agentic_loop("what is the timeout?")
    sub_request = seen[1]
    check("the subagent starts with a fresh context (system prompt + task only)",
          len(sub_request["messages"]) == 2 and "SUBAGENT" in sub_request["messages"][0]["content"])
    check("the subagent cannot delegate further", "delegate" not in [t["function"]["name"] for t in sub_request["tools"]])
    result = next(m for m in agent.messages if m["role"] == "tool")
    check("the main agent receives only the subagent's final report", result["content"].startswith("REPORT: TIMEOUT = 30"))
    check("the subagent's own tool calls stay out of the main history",
          not any(c["id"] == "s1" for m in agent.messages for c in m.get("tool_calls") or []))
    check("subagent actions are permission-checked and logged as 'sub'",
          any(e["type"] == "permission" and e["data"].get("agent") == "sub" for e in events()))


def test_interrupt_inside_subagent():
    workdir = tempfile.mkdtemp()
    reset(workdir)
    script = [tool_chunks([("d1", "delegate", {"task": "run the slow thing"})]),
              tool_chunks([("s1", "bash", {"command": "sleep 30"})])]
    post, _ = fake_model(script)
    real = agent.execute_tool

    def ctrl_c_in_subagent_bash(name, args):
        if name == "bash":
            raise KeyboardInterrupt
        return real(name, args)

    with mock.patch.object(requests, "post", side_effect=post), \
            mock.patch.object(agent, "execute_tool", side_effect=ctrl_c_in_subagent_bash):
        agent.agentic_loop("go")
    check("the interrupt happened inside the subagent",
          any(e["type"] == "subagent_start" for e in events())
          and not any(e["type"] == "subagent_end" for e in events()))
    check("Ctrl-C inside a subagent leaves the main history valid", validate_history(agent.messages) == [])
    check("the delegate call is marked interrupted",
          next(m for m in agent.messages if m["role"] == "tool")["content"] == agent.base.INTERRUPTED)


def test_runner_with_day9():
    from evals import runner
    task = next(t for t in runner.load_tasks() if t["name"] == "tab-indented-edit")
    fixed = open(os.path.join(task["dir"], "solution", "legacy.py")).read()
    script = [tool_chunks([("r", "read_file", {"path": "legacy.py"})]),
              tool_chunks([("w", "write_file", {"path": "legacy.py", "content": fixed})]),
              text_chunks("done")]
    post, _ = fake_model(script)
    with mock.patch.object(requests, "post", side_effect=post):
        r = runner.run_trial(agent, task, sandbox_mode="off")
    check("day 9 agent runs through the eval runner", r["passed"], r["grade_output"])


if __name__ == "__main__":
    home = os.getcwd()
    try:
        test_validator()
        test_compaction_never_splits_pairs()
        test_compaction_in_the_loop()
        test_subagent()
        test_interrupt_inside_subagent()
        test_runner_with_day9()
    finally:
        os.chdir(home)
    print("\nDay 9 tests passed.")
