"""Day 13: ideas ported from production harnesses. Run: python -m tests.test_day13"""
import json
import os
import tempfile
from unittest import mock

import requests

from days import day_13_agent as agent
from safety.permissions import Policy
from tests.test_day7 import check, response, sse, text_chunks, tool_chunks
from tools import ask, execute_tool, files

OPTIONS = [{"label": "SQLite", "description": "one file, no server"},
           {"label": "PostgreSQL", "description": "needs a running server"}]


def scripted_input(*answers):
    answers = list(answers)
    return lambda prompt="": answers.pop(0)


def test_ask_user_tool():
    check("needs 2 to 4 options", "between 2 and 4 options" in execute_tool("ask_user", {"question": "?", "options": OPTIONS[:1]}))
    check("options need labels", "option 2 needs a label" in
          execute_tool("ask_user", {"question": "Which?", "options": [OPTIONS[0], {"description": "x"}]}))
    check("labels must differ", "must be different" in
          execute_tool("ask_user", {"question": "Which?", "options": [OPTIONS[0], OPTIONS[0]]}))
    ask.HANDLER = None
    check("unattended runs: the model is told to choose and say so",
          execute_tool("ask_user", {"question": "Which database?", "options": OPTIONS}) == ask.HEADLESS_REPLY)

    shown = []
    pick = lambda *answers: (lambda q, o, m: ask.terminal_handler(q, o, m, read=scripted_input(*answers),
                                                                  write=shown.append))
    ask.HANDLER = pick("2")
    check("a numbered choice comes back as the answer",
          execute_tool("ask_user", {"question": "Which database?", "options": OPTIONS}) == 'User answered: "PostgreSQL"')
    check("the menu shows the question, descriptions and an 'Other' choice",
          "❓ Which database?" in shown[0] and "1. SQLite: one file, no server" in shown[1] and "3. Other" in shown[3])
    ask.HANDLER = pick("3", "DuckDB, it is embedded")
    check("'Other' lets the user type an answer",
          execute_tool("ask_user", {"question": "Which?", "options": OPTIONS}) == 'User answered: "DuckDB, it is embedded"')
    ask.HANDLER = pick("x", "9", "1 2", "1")
    shown.clear()
    check("invalid input is asked again (not a number, out of range, two for a single choice)",
          execute_tool("ask_user", {"question": "Which?", "options": OPTIONS}) == 'User answered: "SQLite"'
          and sum("Please" in line for line in shown) == 3, shown)
    ask.HANDLER = pick("1, 2")
    check("multi_select accepts several", execute_tool("ask_user", {"question": "Which?", "options": OPTIONS,
                                                                    "multi_select": True})
          == 'User answered: "SQLite"; "PostgreSQL"')
    ask.HANDLER = pick("")
    check("Enter skips, and the model is told to decide", execute_tool("ask_user", {"question": "Which?", "options": OPTIONS})
          == ask.SKIPPED_REPLY)
    ask.HANDLER = None
    check("asking is auto-approved even in auto-read mode", Policy(tempfile.mkdtemp()).check("ask_user", {})[0])


def test_agent_asks_and_continues():
    ws = tempfile.mkdtemp()
    os.chdir(ws)
    agent.messages.clear()
    agent.turn_count = 0
    files._read_state.clear()
    agent.policy = Policy(ws, mode="auto", ask=None)
    agent.log_file = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    asked = []
    ask.HANDLER = lambda q, o, m: asked.append((q, [x["label"] for x in o])) or ["SQLite"]
    script = [tool_chunks([("a1", "ask_user", {"question": "Which database should the app use?", "options": OPTIONS})]),
              tool_chunks([("w1", "write_file", {"path": "db.py", "content": "import sqlite3\n"})]),
              text_chunks("Set up SQLite as you chose.")]
    seen = []

    def post(*args, **kwargs):
        seen.append(json.loads(json.dumps(kwargs["json"])))
        return response(sse(script.pop(0)))

    try:
        with mock.patch.object(requests, "post", side_effect=post):
            agent.agentic_loop("add a database layer")
    finally:
        ask.HANDLER = None
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check("the user saw the question and the options", asked == [("Which database should the app use?", ["SQLite", "PostgreSQL"])])
    check("the answer came back as the tool result, in the same turn", agent.turn_count == 1 and
          next(m for m in agent.messages if m["role"] == "tool")["content"] == 'User answered: "SQLite"')
    check("the model acted on the answer", open(os.path.join(ws, "db.py")).read() == "import sqlite3\n")
    check("the system prompt says to ask with options instead of ending the turn",
          "call ask_user with 2-4 concrete options" in seen[0]["messages"][0]["content"])
    check("ask_user is offered to the main agent", "ask_user" in [t["function"]["name"] for t in seen[0]["tools"]])


def test_subagents_cannot_ask():
    offered = []
    with mock.patch.object(agent, "run_agent", side_effect=lambda msgs, names, *a, **k: offered.extend(names) or "report"):
        agent.budget = agent.budget or agent.TokenBudget(1000)
        agent.log_file = os.devnull
        agent.run_subagent("look around")
    check("subagents have no user to ask, so they do not get ask_user", offered and "ask_user" not in offered, offered)


TOO_LARGE = {"error": {"message": "Request too large for model `qwen/qwen3.8-27b` on input tokens per minute (ITPM): "
                                  "Limit 3000, Requested 3400, please reduce your message size", "code": "rate_limit_exceeded"}}


def fresh_agent(ws):
    os.chdir(ws)
    agent.messages.clear()
    agent.turn_count = 0
    files._read_state.clear()
    agent.policy = Policy(ws, mode="auto", ask=None)
    agent.log_file = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    agent.LEARNED_WINDOW = None


def events():
    return [json.loads(line) for line in open(agent.log_file)]


def test_overflow_recovery():
    from context.compaction import validate_history
    ws = tempfile.mkdtemp()
    for i in range(3):
        with open(os.path.join(ws, f"part{i}.txt"), "w") as f:
            f.write(("data " * 12 + "\n") * 60)
    fresh_agent(ws)
    script = [response(sse(tool_chunks([(f"r{i}", "read_file", {"path": f"part{i}.txt"})]))) for i in range(3)]
    script += [response([], status=413, body=TOO_LARGE), response(sse(text_chunks("All three parts read.")))]
    seen = []

    def post(*args, **kwargs):
        payload = json.loads(json.dumps(kwargs["json"]))
        seen.append(payload)
        if "tools" not in payload:  # the summarizer
            return response(sse(text_chunks("SUMMARY: read part0..part2; they hold repeated data.")))
        return script.pop(0)

    try:
        with mock.patch.object(requests, "post", side_effect=post):
            agent.agentic_loop("read the three parts")
    finally:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    recovery = [e["data"] for e in events() if e["type"] == "overflow_recovery"]
    check("a 'request too large' error is recovered from instead of ending the task",
          agent.messages[-1]["content"] == "All three parts read." and len(recovery) == 1, recovery)
    check("the provider's real limit is learned from the error message", agent.LEARNED_WINDOW == 3000)
    check("the conversation was made smaller before retrying", recovery[0]["tokens_after"] < recovery[0]["tokens_before"])
    rejected, retried = [p for p in seen if "tools" in p][-2:]
    check("the retried request is smaller than the rejected one",
          len(json.dumps(retried["messages"])) < len(json.dumps(rejected["messages"])))
    check("the history is still valid after recovery", validate_history(agent.messages) == [])
    check("from now on, compaction plans for the learned limit", agent.working_window() == 3000)


def test_shrink_keeps_the_latest_step():
    call = lambda i, content: {"id": f"c{i}", "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": f"f{i}.py", "content": content})}}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [call(1, "x" * 5000)]},
            {"role": "tool", "tool_call_id": "c1", "content": "y" * 5000},
            {"role": "assistant", "content": "", "tool_calls": [call(2, "z" * 5000)]},
            {"role": "tool", "tool_call_id": "c2", "content": "w" * 5000}]
    shrunk = agent.shrink_tool_outputs(msgs, target_tokens=10)
    check("old tool output and old file contents are blanked", msgs[3]["content"] == agent.SHRUNK
          and "omitted to fit" in msgs[2]["tool_calls"][0]["function"]["arguments"] and shrunk == 2)
    check("the latest step is left intact", msgs[5]["content"] == "w" * 5000 and "zzz" in msgs[4]["tool_calls"][0]["function"]["arguments"])


def test_overflow_gives_up_and_other_errors_are_not_retried():
    ws = tempfile.mkdtemp()
    with open(os.path.join(ws, "big.txt"), "w") as f:
        f.write("line\n" * 3000)
    fresh_agent(ws)
    first = [response(sse(tool_chunks([(f"r{i}", "read_file", {"path": "big.txt", "offset": i * 500 + 1, "limit": 500})])))
             for i in range(3)]
    calls = {"too_large": 0}

    def post(*args, **kwargs):
        if "tools" not in kwargs["json"]:
            return response(sse(text_chunks("SUMMARY")))
        if first:
            return first.pop(0)
        calls["too_large"] += 1
        return response([], status=413, body=TOO_LARGE)

    try:
        with mock.patch.object(requests, "post", side_effect=post):
            agent.agentic_loop("go")
    finally:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check(f"a request that never fits stops after {agent.MAX_OVERFLOW_RECOVERIES} recoveries (no endless loop)",
          calls["too_large"] <= agent.MAX_OVERFLOW_RECOVERIES + 1, calls)

    fresh_agent(tempfile.mkdtemp())
    with mock.patch.object(requests, "post", return_value=response([], status=400, body={"error": {"message": "bad model"}})) as post:
        agent.agentic_loop("go")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check("other errors are not treated as overflow", post.call_count == 1 and agent.LEARNED_WINDOW is None
          and not any(e["type"] == "overflow_recovery" for e in events()))


def test_window_aware_tool_output():
    ws = tempfile.mkdtemp()
    with open(os.path.join(ws, "big.log"), "w") as f:
        f.write("".join(f"log line {i}: " + "x" * 60 + "\n" for i in range(2000)))
    fresh_agent(ws)
    agent.MODEL = "qwen/qwen3.8-27b"
    check("large window: tool output capped at Codex's 10k default (was 30k)", agent.tool_output_limit() == 10_000)
    # read_file returns up to 40k characters (bash already caps each stream at 8k on its own)
    call = {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "big.log"})}}
    agent.budget = agent.TokenBudget(agent.working_window())
    big = agent.run_tool_call(call, "main", agent.tool_names())
    check("...and applied to real tool results", len(big) <= 10_500 and "characters omitted" in big, len(big))
    agent.LEARNED_WINDOW = 7000
    small = agent.run_tool_call(call, "main", agent.tool_names())
    check("after learning a 7k limit, tool output shrinks to fit it", len(small) <= 4_000 < len(big), len(small))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    agent.LEARNED_WINDOW = None
    agent.MODEL = agent.base.MODEL


if __name__ == "__main__":
    agent.PROVIDER = "openrouter"  # model calls are faked
    test_ask_user_tool()
    test_agent_asks_and_continues()
    test_subagents_cannot_ask()
    test_overflow_recovery()
    test_shrink_keeps_the_latest_step()
    test_overflow_gives_up_and_other_errors_are_not_retried()
    test_window_aware_tool_output()
    print("\nDay 13 tests passed.")
