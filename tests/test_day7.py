"""Day 7: streaming, safe interrupts, text protocol. Run: python -m tests.test_day7"""
import json
import os
import signal
import tempfile
import threading
import time
from unittest import mock

import requests

from days import day_7_agent as agent
from evals import runner
from models.sse import StreamError, assemble, iter_events
from tools import execute_tool, files

USAGE = {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0001}


def check(label, condition, detail=""):
    assert condition, f"{label}\n{detail}"
    print(f"✅ {label}")


# ---------- fake streaming server ----------

def sse(chunks, done=True):
    lines = [": OPENROUTER PROCESSING", ""]
    for chunk in chunks:
        lines += [f"data: {json.dumps(chunk)}", ""]
    if done:
        lines += ["data: [DONE]", ""]
    return lines


def text_chunks(text, size=4):
    pieces = [text[i:i + size] for i in range(0, len(text), size)]
    return ([{"choices": [{"delta": {"content": p}}]} for p in pieces]
            + [{"choices": [{"delta": {}, "finish_reason": "stop"}]}, {"choices": [], "usage": USAGE}])


def tool_chunks(calls):
    """calls: list of (id, name, args_dict); arguments are split over several chunks like real streams."""
    chunks = []
    for index, (call_id, name, args) in enumerate(calls):
        raw = json.dumps(args)
        chunks.append({"choices": [{"delta": {"tool_calls": [
            {"index": index, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}]}}]})
        for i in range(0, len(raw), 7):
            chunks.append({"choices": [{"delta": {"tool_calls": [
                {"index": index, "function": {"arguments": raw[i:i + 7]}}]}}]})
    return chunks + [{"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}, {"choices": [], "usage": USAGE}]


def response(lines, status=200, body=None):
    r = mock.MagicMock()
    r.status_code = status
    r.__enter__.return_value = r
    r.iter_lines.return_value = lines
    r.headers = {}
    r.json.return_value = body or {}
    return r


def reset(module, workdir):
    module.messages.clear()
    module.turn_count = 0
    files._read_state.clear()
    module.policy = agent.Policy(workdir, mode="auto", ask=None)
    module.log_file = os.path.join(tempfile.mkdtemp(), "transcript.jsonl")


def assert_tool_calls_answered(messages):
    pending = set()
    for m in messages:
        if m["role"] == "tool":
            pending.discard(m["tool_call_id"])
            continue
        assert not pending, f"unanswered tool calls {pending} before a {m['role']} message"
        pending = {c["id"] for c in m.get("tool_calls") or []}
    assert not pending, f"unanswered tool calls {pending} at the end"


# ---------- tests ----------

def test_sse_parsing():
    seen = []
    body = assemble(iter_events(sse(text_chunks("Hello, streaming world!"))), on_text=seen.append)
    check("text arrives in pieces and is reassembled",
          len(seen) > 3 and body["choices"][0]["message"]["content"] == "Hello, streaming world!")
    check("finish_reason and usage are kept", body["choices"][0]["finish_reason"] == "stop" and body["usage"] == USAGE)

    body = assemble(iter_events(sse(tool_chunks([("a", "read_file", {"path": "x.py"}),
                                                   ("b", "bash", {"command": "ls -la && echo done"})]))))
    calls = body["choices"][0]["message"]["tool_calls"]
    check("parallel tool calls rebuilt from fragments", [c["id"] for c in calls] == ["a", "b"]
          and json.loads(calls[1]["function"]["arguments"]) == {"command": "ls -la && echo done"}, calls)

    gemini = [{"choices": [{"delta": {"tool_calls": [
        {"id": "g1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a"}'},
         "extra_content": {"google": {"thought_signature": "SIG1"}}},
        {"id": "g2", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
    calls = assemble(iter_events(sse(gemini)))["choices"][0]["message"]["tool_calls"]
    check("Gemini-style calls (id, no index) stay separate",
          [(c["id"], c["function"]["name"]) for c in calls] == [("g1", "read_file"), ("g2", "bash")], calls)
    check("provider extras like Gemini's thought_signature are kept",
          calls[0]["extra_content"]["google"]["thought_signature"] == "SIG1")

    multi = ['data: {"choices": [{"delta":', 'data: {"content": "hi"}}]}', "", "data: [DONE]", ""]
    check("an event spread over several data: lines is joined",
          assemble(iter_events(multi))["choices"][0]["message"]["content"] == "hi")
    no_done = ['data: {"choices": [{"delta": {"content": "x"}}]}']
    check("a stream cut off without [DONE] still yields its data", len(list(iter_events(no_done))) == 1)
    try:
        assemble(iter_events(sse([{"error": {"code": 502, "message": "provider died"}}])))
        check("mid-stream error raises", False)
    except StreamError as e:
        check("mid-stream error raises StreamError with its code", e.code == 502)


def test_call_model_streaming():
    agent.log_file = os.devnull
    with mock.patch.object(requests, "post", return_value=response(sse(text_chunks("hi there")))) as post:
        seen = []
        body = agent.call_model({"model": "x", "messages": []}, on_text=seen.append)
    sent = post.call_args.kwargs["json"]
    check("request asks for a stream with usage", sent["stream"] is True and sent["stream_options"]["include_usage"])
    check("call_model returns the assembled body", body["choices"][0]["message"]["content"] == "hi there"
          and "".join(seen) == "hi there")

    with mock.patch.object(requests, "post", side_effect=[
            response(sse([{"choices": [{"delta": {"content": "par"}}]}, {"error": {"code": 503, "message": "x"}}])),
            response(sse(text_chunks("full answer")))]), mock.patch.object(agent.time, "sleep"):
        body = agent.call_model({"model": "x", "messages": []})
    check("an error in the middle of a stream is retried", body["choices"][0]["message"]["content"] == "full answer")

    with mock.patch.object(requests, "post", return_value=response([], status=400,
                                                                   body={"error": {"code": 400, "message": "bad"}})):
        check("non-retryable HTTP errors still return None", agent.call_model({"model": "x", "messages": []}) is None)


def test_repair_history():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}, {"id": "2"}, {"id": "3"}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "user", "content": "next"}]
    check("repair fills in missing results", agent.repair_history(msgs) == 2)
    assert_tool_calls_answered(msgs)
    check("filled results sit right after the tool call, before the next user message",
          [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool", "tool", "user"])
    check("repair is a no-op on a valid history", agent.repair_history(msgs) == 0)


def test_interrupt_during_tools():
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    reset(agent, workdir)
    calls = [("c1", "write_file", {"path": "a.txt", "content": "a"}),
             ("c2", "bash", {"command": "sleep 30"}),
             ("c3", "write_file", {"path": "c.txt", "content": "c"})]
    real = execute_tool

    def ctrl_c_on_bash(name, args):
        if name == "bash":
            raise KeyboardInterrupt
        return real(name, args)

    with mock.patch.object(requests, "post", side_effect=[response(sse(tool_chunks(calls)))]), \
            mock.patch.object(agent, "execute_tool", side_effect=ctrl_c_on_bash):
        agent.agentic_loop("do three things")
    check("Ctrl-C during a tool returns to the prompt", True)
    assert_tool_calls_answered(agent.messages)
    results = {m["tool_call_id"]: m["content"] for m in agent.messages if m["role"] == "tool"}
    check("finished tool keeps its real result", results["c1"].startswith("Wrote"))
    check("interrupted and not-yet-run tools get 'interrupted' results",
          results["c2"] == agent.INTERRUPTED and results["c3"] == agent.INTERRUPTED)
    check("tools after the interrupt did not run", not os.path.exists("c.txt"))

    with mock.patch.object(requests, "post", return_value=response(sse(text_chunks("ok, stopped")))) as post:
        agent.agentic_loop("never mind")
    assert_tool_calls_answered(post.call_args.kwargs["json"]["messages"])
    check("the conversation continues with a valid history", agent.messages[-1]["content"] == "ok, stopped")


def test_interrupt_during_stream():
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    reset(agent, workdir)

    def lines():
        yield from sse([{"choices": [{"delta": {"content": "Let me explain the"}}]}], done=False)
        raise KeyboardInterrupt

    r = response(None)
    r.iter_lines.return_value = lines()
    with mock.patch.object(requests, "post", return_value=r):
        agent.agentic_loop("explain")
    last = agent.messages[-1]
    check("Ctrl-C while streaming keeps the partial reply, marked as interrupted",
          last["role"] == "assistant" and last["content"].startswith("Let me explain the")
          and "interrupted" in last["content"], last)


def test_ctrl_c_kills_host_command():
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    try:
        execute_tool("bash", {"command": "sleep 60 & echo $! > child.pid; wait", "timeout": 30})
        check("Ctrl-C reaches the tool", False)
    except KeyboardInterrupt:
        pass
    time.sleep(0.3)
    pid = int(open("child.pid").read())
    try:
        os.kill(pid, 0)
        alive = open(f"/proc/{pid}/stat").read().split()[2] != "Z"
    except ProcessLookupError:
        alive = False
    check("Ctrl-C during bash also kills the processes it started", not alive)


def test_text_protocol():
    check("one bash block is parsed", agent.parse_commands("I'll list.\n```bash\nls -la\n```") == ["ls -la"])
    check("no block means done", agent.parse_commands("All finished.") == [])
    check("two blocks are both found (format error)",
          len(agent.parse_commands("```bash\na\n```\nand\n```sh\nb\n```")) == 2)

    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    reset(agent, workdir)
    agent.PROTOCOL = "text"
    replies = ["Creating the file.\n```bash\necho hello > greeting.txt && cat greeting.txt\n```",
               "Two at once:\n```bash\nls\n```\n```bash\npwd\n```",
               "Checking.\n```bash\nwc -c greeting.txt\n```",
               "Done: greeting.txt contains hello."]
    with mock.patch.object(requests, "post", side_effect=[response(sse(text_chunks(r))) for r in replies]) as post:
        agent.agentic_loop("make a greeting file")
    check("text agent sends no tools to the API", "tools" not in post.call_args.kwargs["json"])
    check("text agent ran the command", open("greeting.txt").read() == "hello\n")
    user_msgs = [m["content"] for m in agent.messages if m["role"] == "user"]
    check("command output goes back as a user message", any(m.startswith("Command output:") and "hello" in m
                                                            for m in user_msgs))
    check("two code blocks get a format error", any(m.startswith("Format error") for m in user_msgs))
    check("a reply without a block ends the turn", agent.messages[-1]["content"].startswith("Done"))
    agent.PROTOCOL = "tools"


def test_text_protocol_in_evals():
    task = next(t for t in runner.load_tasks() if t["name"] == "pager-off-by-one")
    solution = open(os.path.join(task["dir"], "solution", "pager.py")).read()
    replies = [f"Rewriting pager.py.\n```bash\ncat > pager.py <<'EOF'\n{solution}EOF\n```",
               "Fixed both bugs."]
    with mock.patch.object(requests, "post", side_effect=[response(sse(text_chunks(r))) for r in replies]):
        agent.PROTOCOL = "text"
        try:
            r = runner.run_trial(agent, task, sandbox_mode="off")
        finally:
            agent.PROTOCOL = "tools"
    check("text agent runs through the eval runner and passes a task", r["passed"], r["grade_output"])
    check("eval metrics work for the text agent", r["steps"] == 2 and r["tool_calls"] == 1, r)


if __name__ == "__main__":
    home = os.getcwd()
    try:
        test_sse_parsing()
        test_call_model_streaming()
        test_repair_history()
        test_interrupt_during_tools()
        test_interrupt_during_stream()
        test_ctrl_c_kills_host_command()
        test_text_protocol()
        test_text_protocol_in_evals()
    finally:
        os.chdir(home)
    print("\nDay 7 tests passed.")
