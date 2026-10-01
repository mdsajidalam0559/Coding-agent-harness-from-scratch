"""Day 14: model adapters, the agent core, two agents on one core, headless mode. Run: python -m tests.test_day14"""
import contextlib
import io
import json
import os
import re
import tempfile
from unittest import mock

import requests

from agents import make_agent
from context.compaction import validate_history
from core.agent import Agent
from models import anthropic_http
from models.base import ModelAdapter, ModelError, ModelResponse
from models.openai_compat import OpenAICompatAdapter
from models.registry import make_model
from models.text_protocol import TextProtocolAdapter
from tests.test_day7 import check, response, sse, text_chunks, tool_chunks
from tools import files, todo

HERE = os.path.dirname(os.path.abspath(__file__))


class ScriptedModel(ModelAdapter):
    """A model that follows a script: ModelResponse items, ModelError items, or callables(messages, tools)."""
    name = "scripted"

    def __init__(self, script, context_window=128_000):
        super().__init__()
        self.script, self.context_window, self.calls = list(script), context_window, []

    def complete(self, messages, tools, on_text=None, max_tokens=4000):
        if not tools and messages and messages[0]["content"].startswith("You compact the working memory"):
            return answer("SUMMARY: earlier steps")  # summarizer requests are answered automatically
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": [t["function"]["name"] for t in tools or []]})
        item = self.script.pop(0)
        if callable(item):
            item = item(messages, tools)
        if isinstance(item, ModelError):
            self.log("api_error", {"attempt": 0, "status": item.status, "error": (item.body or {}).get("error"),
                                   "retryable": False, "final": True})
            raise item
        if on_text and item.text:
            on_text(item.text)
        return item


def call(call_id, name, **args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def tools_reply(*calls):
    return ModelResponse(tool_calls=list(calls), usage={"prompt_tokens": 500, "completion_tokens": 20}, finish_reason="tool_calls")


def answer(text):
    return ModelResponse(text=text, usage={"prompt_tokens": 600, "completion_tokens": 30}, finish_reason="stop")


def workspace(files_map=None):
    ws = tempfile.mkdtemp()
    for rel, content in (files_map or {}).items():
        os.makedirs(os.path.dirname(os.path.join(ws, rel)) or ws, exist_ok=True)
        with open(os.path.join(ws, rel), "w") as f:
            f.write(content)
    files._read_state.clear()
    todo.reset()
    return ws


@contextlib.contextmanager
def inside(ws):
    home = os.getcwd()
    os.chdir(ws)
    try:
        yield
    finally:
        os.chdir(home)


# ---------------- adapters ----------------

def test_openai_adapter():
    os.environ["TEST_KEY"] = "k"
    adapter = OpenAICompatAdapter("https://example.test/v1", "m", "TEST_KEY", 1000)
    logged = []
    adapter.log = lambda e, d: logged.append((e, d))
    with mock.patch.object(requests, "post", return_value=response(sse(tool_chunks([("a", "bash", {"command": "ls"})])))) as post:
        r = adapter.complete([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "bash"}}])
    check("openai adapter: streamed tool calls become canonical tool_calls",
          r.tool_calls[0]["function"]["name"] == "bash" and r.finish_reason == "tool_calls")
    check("openai adapter: posts to <url>/chat/completions with the key",
          post.call_args.args[0] == "https://example.test/v1/chat/completions"
          and post.call_args.kwargs["headers"]["Authorization"] == "Bearer k")
    failed = response([], status=400, body={"error": {"code": "tool_use_failed", "message": "Failed to parse tool call arguments as JSON",
                                                      "failed_generation": '{"name": "bash", "arguments": {"command": "apply_patch <<EOF'}})
    with mock.patch.object(requests, "post", return_value=failed):
        r = adapter.complete([{"role": "user", "content": "hi"}], [])
    check("openai adapter: a provider-rejected malformed tool call becomes a format error for the model to fix",
          r.format_error and "not valid JSON" in r.format_error and "apply_patch" in r.format_error)
    with mock.patch.object(requests, "post", return_value=response([], status=413, body={"error": {"message": "Request too large, Limit 7000, Requested 7100"}})):
        try:
            adapter.complete([{"role": "user", "content": "hi"}], [])
            check("openai adapter: final errors raise", False)
        except ModelError as e:
            check("openai adapter: a 413 raises ModelError and is logged as a final api_error",
                  e.status == 413 and logged[-1][0] == "api_error" and logged[-1][1]["final"])


ANTHROPIC_EVENTS = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 12, "cache_read_input_tokens": 900, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me look"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"pa'}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'th": "a.py"}'}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 40}},
    {"type": "message_stop"},
]


def test_anthropic_adapter():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": "", "tool_calls": [call("t1", "read_file", path="a.py"), call("t2", "bash", command="ls")]},
                {"role": "tool", "tool_call_id": "t1", "content": "x = 1"},
                {"role": "tool", "tool_call_id": "t2", "content": "a.py"},
                {"role": "user", "content": "go on"}]
    system, converted, tools = anthropic_http.to_anthropic(messages, [{"type": "function", "function": {
        "name": "read_file", "description": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}])
    check("anthropic: system prompt moves to the system parameter, with a cache breakpoint",
          system == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}])
    check("anthropic: tool calls become tool_use blocks", converted[1]["content"][0] ==
          {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}})
    check("anthropic: tool results and the next user message merge into ONE user message (strict alternation)",
          [m["role"] for m in converted] == ["user", "assistant", "user"]
          and [b["type"] for b in converted[2]["content"]] == ["tool_result", "tool_result", "text"])
    check("anthropic: the newest block gets a cache breakpoint", converted[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"})
    check("anthropic: tools use input_schema", tools[0]["input_schema"]["properties"] == {"path": {"type": "string"}})

    seen = []
    r = anthropic_http.assemble_stream(iter(ANTHROPIC_EVENTS), on_text=seen.append)
    check("anthropic: streamed text and tool_use (from input_json_delta pieces) are rebuilt",
          r.text == "Let me look" and seen == ["Let me look"] and r.tool_calls[0]["function"]["arguments"] == '{"path": "a.py"}'
          and r.finish_reason == "tool_calls")
    check("anthropic: usage counts cached prompt tokens", r.usage["prompt_tokens"] == 912
          and r.usage["prompt_tokens_details"]["cached_tokens"] == 900 and r.usage["completion_tokens"] == 40)
    try:
        anthropic_http.assemble_stream(iter([{"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}]))
        check("anthropic: overloaded raises", False)
    except ModelError as e:
        check("anthropic: an 'overloaded' stream error is retryable", e.retryable and e.status == 529)

    os.environ["TEST_ANTHROPIC_KEY"] = "sk-ant-test"
    adapter = anthropic_http.AnthropicAdapter("claude-test", "TEST_ANTHROPIC_KEY")
    lines = [line for e in ANTHROPIC_EVENTS for line in (f"event: {e['type']}", f"data: {json.dumps(e)}", "")]
    with mock.patch.object(requests, "post", return_value=response(lines)) as post:
        r = adapter.complete(messages, [])
    headers, payload = post.call_args.kwargs["headers"], post.call_args.kwargs["json"]
    check("anthropic: raw HTTP with x-api-key and anthropic-version headers",
          headers["x-api-key"] == "sk-ant-test" and headers["anthropic-version"] == "2023-06-01" and payload["stream"] is True)
    check("anthropic: the adapter returns canonical output", r.tool_calls[0]["id"] == "toolu_1")


def test_text_protocol():
    inner = ScriptedModel([answer('I will read it.\n```tool\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```\n```bash\nls -la\n```'),
                           answer("```tool\n{not json}\n```"), answer("All done.")])
    adapter = TextProtocolAdapter(inner)
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                               "parameters": {"properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    r = adapter.complete([{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}], tools)
    check("text protocol: fenced tool and bash blocks become tool_calls",
          [c["function"]["name"] for c in r.tool_calls] == ["read_file", "bash"]
          and json.loads(r.tool_calls[1]["function"]["arguments"]) == {"command": "ls -la"})
    check("text protocol: the inner model gets no tools, and the protocol + tool list in its system prompt",
          inner.calls[0]["tools"] == [] and "read_file(path: string): Read a file" in inner.calls[0]["messages"][0]["content"])
    check("text protocol: bad JSON is a format error for the model to fix",
          adapter.complete([{"role": "user", "content": "x"}], tools).format_error.startswith("Format error"))
    history = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "calling", "tool_calls": [call("t1", "read_file", path="a")]},
               {"role": "tool", "tool_call_id": "t1", "content": "RESULT"}]
    translated = adapter.translate(history, tools)
    check("text protocol: tool results go back as user messages", translated[-1] == {"role": "user", "content": "[result of call t1]\nRESULT"}
          and "tool_calls" not in translated[1])


def test_registry():
    check("spec: provider:model", make_model("groq:qwen/qwen3.8-27b").name == "groq:qwen/qwen3.8-27b")
    check("spec: +text wraps the text protocol", isinstance(make_model("ollama:qwen3:8b+text"), TextProtocolAdapter))
    check("spec: anthropic uses the native adapter", isinstance(make_model("anthropic:claude-haiku-4-5"), anthropic_http.AnthropicAdapter))
    check("context windows come from the known list", make_model("groq:openai/gpt-oss-120b").context_window == 131_072)
    try:
        make_model("nope")
        check("bad specs are explained", False)
    except ValueError as e:
        check("bad specs are explained", "should look like" in str(e))


# ---------------- core ----------------

def test_core_coding_agent():
    ws = workspace({"calc.py": "def add(a, b):\n    return a - b\n"})
    model = ScriptedModel([
        tools_reply(call("c1", "read_file", path="calc.py")),
        tools_reply(call("c2", "str_replace", path="calc.py", old_str="return a - b", new_str="return a + b")),
        tools_reply(call("c3", "bash", command="python3 -c 'from calc import add; print(add(2, 3))'")),
        answer("Fixed add: it subtracted. add(2, 3) now prints 5."),
    ])
    with inside(ws):
        agent = make_agent("coding", model, ws, mode="auto")
        result = agent.run("add() is wrong, fix it")
    check("core: a coding task runs end to end on any adapter", result.status == "done" and "5" in result.text
          and open(os.path.join(ws, "calc.py")).read() == "def add(a, b):\n    return a + b\n")
    check("core: the bash result reached the model", "5" in model.calls[3]["messages"][-1]["content"])
    check("core: the history is valid", validate_history(agent.messages) == [])
    check("core: usage is tracked per turn", result.usage["calls"] == 4 and result.usage["prompt"] == 2100)
    from evals.runner import summarize_transcript
    stats = summarize_transcript(agent.log_file)
    check("core: transcripts keep the eval format", stats["steps"] == 4 and stats["tool_calls"] == 3)


def test_core_features():
    ws = workspace({"big.txt": ("data " * 20 + "\n") * 400})
    too_large = ModelError("HTTP 413", status=413, body={"error": {"message": "Request too large ... Limit 3000, Requested 3500"}})
    model = ScriptedModel([tools_reply(call("r1", "read_file", path="big.txt")),
                           tools_reply(call("r2", "read_file", path="big.txt", offset=100)),
                           too_large,
                           answer("done after recovery")])
    with inside(ws):
        agent = make_agent("coding", model, ws, mode="auto")
        result = agent.run("read big.txt twice")
    check("core: overflow recovery learns the limit and finishes", result.status == "done" and agent.learned_window == 3000)
    check("core: tool output is capped for the learned window afterwards", agent.tool_output_limit() < 2_000)

    ws = workspace()
    model = ScriptedModel([tools_reply(call("d1", "delegate", task="count the files")),
                           tools_reply(call("s1", "list_dir", path=".")),
                           answer("REPORT: 0 files"), answer("There are no files.")])
    with inside(ws):
        agent = make_agent("coding", model, ws, mode="auto")
        agent.run("how many files?")
    check("core: delegate runs a subagent with a fresh context and without delegate",
          len(model.calls[1]["messages"]) == 2 and "delegate" not in model.calls[1]["tools"]
          and next(m for m in agent.messages if m["role"] == "tool")["content"] == "REPORT: 0 files")

    model = ScriptedModel([tools_reply(call("w1", "write_file", path="x.py", content="x = 1\n"))])
    with inside(ws):
        agent = make_agent("coding", model, ws, mode="auto-read")  # edits need approval, nobody to ask
        model.script.append(answer("I could not write the file."))
        agent.run("write x.py")
    check("core: permissions apply (auto-read with no human denies edits)", not os.path.exists(os.path.join(ws, "x.py"))
          and next(m for m in agent.messages if m["role"] == "tool")["content"].startswith("Permission denied"))

    model = ScriptedModel([tools_reply(call(f"l{i}", "list_dir", path=".")) for i in range(5)])
    with inside(ws):
        agent = make_agent("coding", model, ws, mode="auto")
        agent.config.max_steps = 3
        check("core: the step limit stops runaway loops", agent.run("loop").status == "max_steps")

    model = ScriptedModel([tools_reply(call("b1", "bash", command="sleep 30"))])
    with inside(ws), mock.patch("core.agent.execute_tool", side_effect=KeyboardInterrupt):
        agent = make_agent("coding", model, ws, mode="auto")
        result = agent.run("sleep")
    check("core: Ctrl-C returns 'interrupted' with a valid history", result.status == "interrupted"
          and validate_history(agent.messages) == [])

    inner = ScriptedModel([answer("```tool\n{broken\n```"), answer('```tool\n{"name": "list_dir", "arguments": {"path": "."}}\n```'),
                           answer("Listed.")])
    with inside(ws):
        agent = make_agent("coding", TextProtocolAdapter(inner), ws, mode="auto")
        result = agent.run("list")
    check("core + text protocol: a format error is sent back, then the tool call works",
          result.status == "done" and inner.calls[1]["messages"][-1]["content"].startswith("Format error")
          and any(m["role"] == "tool" for m in agent.messages))


def test_two_agents_one_core():
    ws = workspace({"data/sales.csv": "region,units,price\nNorth,10,2\nSouth,3,5\nNorth,1,2\n",
                    "report_tool.py": "print('hi')\n"})
    data_model = ScriptedModel([
        tools_reply(call("q1", "list_tables")),
        tools_reply(call("q2", "query", sql="SELECT region, SUM(units*price) AS revenue FROM sales GROUP BY region ORDER BY revenue DESC")),
        tools_reply(call("q3", "write_report", filename="answer.md", content="North: 22")),
        answer("North has the most revenue: 22."),
    ])
    coding_model = ScriptedModel([tools_reply(call("c1", "read_file", path="report_tool.py")), answer("It prints hi.")])
    with inside(ws):
        data_agent = make_agent("data", data_model, ws, mode="auto")
        coding_agent = make_agent("coding", coding_model, ws, mode="auto")
        data_result = data_agent.run("Which region has the most revenue?")
        coding_result = coding_agent.run("What does report_tool.py do?")
    check("two agents on one core: the data agent answers from SQL", data_result.status == "done"
          and any(m["role"] == "tool" and re.search(r"North\s*\|\s*22", m["content"]) for m in data_agent.messages)
          and open(os.path.join(ws, "reports/answer.md")).read() == "North: 22")
    check("two agents on one core: the coding agent works in the same process", coding_result.text == "It prints hi.")
    check("...each with its own tools", data_model.calls[0]["tools"][:3] == ["list_tables", "describe_table", "query"]
          and "bash" not in data_model.calls[0]["tools"] and "bash" in coding_model.calls[0]["tools"])
    check("...its own system prompt", data_model.calls[0]["messages"][0]["content"].startswith("You are a data analyst")
          and "coding agent" in coding_model.calls[0]["messages"][0]["content"])
    check("...and its own state", len(data_agent.messages) != len(coding_agent.messages)
          and data_agent.log_file != coding_agent.log_file)


def test_agent_scoped_state():
    a_ws = workspace({"shared.py": "x = 1\n"})
    b_ws = workspace({"shared.py": "x = 1\n"})
    reader = ScriptedModel([tools_reply(call("r", "read_file", path="shared.py")), answer("read it")])
    writer = ScriptedModel([tools_reply(call("e", "str_replace", path="shared.py", old_str="x = 1", new_str="x = 2")),
                            answer("tried")])
    elsewhere = tempfile.mkdtemp()
    with inside(elsewhere):  # the process is NOT in either workspace
        agent_a = make_agent("coding", reader, a_ws, mode="auto")
        agent_b = make_agent("coding", writer, b_ws, mode="auto")
        agent_a.run("read shared.py")
        agent_b.run("change x to 2 without reading")
    check("tools work in the agent's own workspace, wherever the process is",
          next(m for m in agent_a.messages if m["role"] == "tool")["content"].strip().startswith("1\tx = 1"))
    check("agents do not share their record of read files (B may not edit what only A read)",
          next(m for m in agent_b.messages if m["role"] == "tool")["content"].startswith("Error: you have not read")
          and open(os.path.join(b_ws, "shared.py")).read() == "x = 1\n")

    parent = ScriptedModel([tools_reply(call("r", "read_file", path="shared.py")),
                            tools_reply(call("d", "delegate", task="set x to 3")),
                            tools_reply(call("s", "str_replace", path="shared.py", old_str="x = 1", new_str="x = 3")),
                            answer("sub tried"), answer("done")])
    with inside(elsewhere):
        agent = make_agent("coding", parent, a_ws, mode="auto")
        agent.run("read, then delegate")
    sub_result = next(m for m in agent.messages if m["role"] == "tool" and m["tool_call_id"] == "d")
    check("a subagent starts with no read files, even ones its parent read",
          open(os.path.join(a_ws, "shared.py")).read() == "x = 1\n" and "sub tried" in sub_result["content"])

    class FakeSandbox:
        def __init__(self, name):
            self.name, self.commands = name, []

        def exec(self, command, timeout, env):
            self.commands.append(command)
            return 0, f"ran in {self.name}", "", False

    sandbox_a, sandbox_b = FakeSandbox("A"), FakeSandbox("B")
    with inside(elsewhere):
        agent_a = make_agent("coding", ScriptedModel([tools_reply(call("x", "bash", command="echo a")), answer("ok")]),
                             a_ws, mode="auto", sandbox=sandbox_a)
        agent_b = make_agent("coding", ScriptedModel([tools_reply(call("y", "bash", command="echo b")), answer("ok")]),
                             b_ws, mode="auto", sandbox=sandbox_b)
        agent_a.run("a")
        agent_b.run("b")
    check("each agent's commands go to its own sandbox", sandbox_a.commands == ["echo a"] and sandbox_b.commands == ["echo b"])


# ---------------- headless + runner ----------------

def test_headless():
    from ui import headless
    ws = workspace({"note.txt": "hello\n"})
    model = ScriptedModel([tools_reply(call("w1", "write_file", path="out.txt", content="done\n")), answer("Wrote out.txt.")])
    out = io.StringIO()
    with mock.patch.object(headless, "make_model", return_value=model), contextlib.redirect_stdout(out), \
            mock.patch("sys.stderr", io.StringIO()):
        code = headless.main(["write out.txt", "--workspace", ws, "--sandbox", "off", "--unsandboxed-ok", "--json"])
    os.chdir(HERE)
    result = json.loads(out.getvalue())
    check("headless: exit code 0 and a JSON result on stdout", code == 0 and result["status"] == "done"
          and result["answer"] == "Wrote out.txt.")
    check("headless: reports the files it changed", result["files_changed"] == ["out.txt"])
    with contextlib.redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
        check("headless: refuses unattended unsandboxed commands without an explicit flag",
              headless.main(["x", "--workspace", ws, "--sandbox", "off"]) == 2)
    model = ScriptedModel([tools_reply(call(f"l{i}", "list_dir", path=".")) for i in range(30)])
    with mock.patch.object(headless, "make_model", return_value=model), contextlib.redirect_stdout(io.StringIO()), \
            mock.patch("sys.stderr", io.StringIO()):
        code = headless.main(["loop", "--workspace", ws, "--sandbox", "off", "--unsandboxed-ok", "--quiet"])
    os.chdir(HERE)
    check("headless: a stopped run exits 1", code == 1)


def test_runner_core_agents():
    from evals import runner
    task = next(t for t in runner.load_tasks() if t["name"] == "pager-off-by-one")
    fixed = open(os.path.join(task["dir"], "solution", "pager.py")).read()
    model = ScriptedModel([tools_reply(call("r", "read_file", path="pager.py")),
                           tools_reply(call("w", "write_file", path="pager.py", content=fixed)), answer("fixed")])
    r = runner.run_core_trial("coding", model, task, "off")
    check("runner: core coding agent solves a coding task", r["passed"] and r["status"] == "done" and r["steps"] == 3)
    task = next(t for t in runner.load_tasks(suite="data") if t["name"] == "revenue-by-region")
    solution = open(os.path.join(task["dir"], "solution", "reports", "answer.md")).read()
    model = ScriptedModel([tools_reply(call("w", "write_report", filename="answer.md", content=solution)), answer("done")])
    r = runner.run_core_trial("data", model, task, "off")
    check("runner: core data agent is graded on the data suite", r["passed"], r["grade_output"])


if __name__ == "__main__":
    test_openai_adapter()
    test_anthropic_adapter()
    test_text_protocol()
    test_registry()
    test_core_coding_agent()
    test_core_features()
    test_two_agents_one_core()
    test_agent_scoped_state()
    test_headless()
    test_runner_core_agents()
    print("\nDay 14 tests passed.")
