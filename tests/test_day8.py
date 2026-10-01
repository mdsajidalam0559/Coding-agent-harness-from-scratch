"""Day 8: system prompt, project memory, token budget, output caps, cache breakpoints. Run: python -m tests.test_day8"""
import json
import os
import tempfile
from unittest import mock

import requests

from days import day_8_agent as agent
from context.budget import TokenBudget, clip_tool_result, context_window
from context.prompt import build_system_prompt
from tests.test_day7 import check, response, sse, text_chunks, tool_chunks
from tools import execute_tool, files


def test_tool_arguments():
    out = execute_tool("read_file", {"path": "x", "line_start": 1, "line_end": 9})
    check("invented parameters get a helpful error", "unknown parameter(s) line_end, line_start" in out
          and "Valid parameters: path (required), offset (optional), limit (optional)" in out, out)
    out = execute_tool("write_file", {"path": "x"})
    check("missing required parameters are named", "missing required parameter(s) content" in out, out)
    check("unknown tools list the real ones", "Available tools:" in execute_tool("read_files", {}))
    offered = agent.tool_names()
    check("namespaced names like gpt-oss's 'repo_browser.list_dir' map to the real tool",
          agent.resolve_tool_name("repo_browser.list_dir", offered) == ("list_dir", None))
    name, error = agent.resolve_tool_name("delete_everything", offered)
    check("a tool that was not offered gets the list of real tools", name is None and "Available tools" in error)
    check("list_dir treats an empty path as the current directory", not execute_tool("list_dir", {"path": ""}).startswith("Error"))


def test_rate_limit_with_text_code():
    """Groq answers 429 with error code "rate_limit_exceeded" (a string): it must still be retried."""
    from days import day_7_agent
    day_7_agent.log_file = os.devnull
    limited = response([], status=429, body={"error": {"code": "rate_limit_exceeded", "message": "try again in 5s"}})
    limited.headers = {"retry-after": "5"}
    with mock.patch.object(requests, "post", side_effect=[limited, response(sse(text_chunks("ok")))]), \
            mock.patch.object(day_7_agent.time, "sleep") as sleep:
        body = day_7_agent.call_model({"model": "x", "messages": []})
    check("HTTP 429 is retried even when the body's error code is text",
          body is not None and body["choices"][0]["message"]["content"] == "ok" and sleep.called)


def test_read_file_caps():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "big.txt")
    with open(path, "w") as f:
        f.write("".join(f"{'x' * 99}\n" for _ in range(1000)))  # 100k characters
    out = execute_tool("read_file", {"path": path})
    check("read_file stops at the character cap and says how to continue",
          len(out) <= files.MAX_READ_CHARS + 200 and "call again with offset=" in out, len(out))
    with open(path, "w") as f:
        f.write("a" * 50_000 + "\nsecond line\n")
    out = execute_tool("read_file", {"path": path})
    check("very long lines are shortened", "line truncated" in out and "second line" in out)


def test_budget():
    check("clip keeps short results", clip_tool_result("abc") == "abc")
    clipped = clip_tool_result("A" * 20_000 + "B" * 20_000, limit=10_000)
    check("clip keeps head and tail and explains", clipped.startswith("A") and clipped.endswith("B")
          and "characters omitted" in clipped and len(clipped) < 10_500)

    b = TokenBudget(window=100_000)
    b.record({"prompt_tokens": 10_000, "completion_tokens": 500, "prompt_tokens_details": {"cached_tokens": 8_000},
              "cost": 0.25})
    b.record({"prompt_tokens": 12_000, "completion_tokens": 300, "cost": 0.5})
    check("budget sums calls, tokens, cached tokens and cost",
          b.turn == {"calls": 2, "prompt": 22_000, "cached": 8_000, "completion": 800, "cost": 0.75}, b.turn)
    check("context size follows the latest call", b.context_tokens == 12_300)
    check("status line reads well", "context 12.3k/100.0k (12%)" in b.status() and "8.0k cached" in b.status(),
          b.status())
    b.start_turn()
    check("a new turn resets turn totals but not the session", b.turn["calls"] == 0 and b.session["calls"] == 2)

    check("known model windows", context_window("openai/gpt-oss-120b") == 131_072)
    check("model families match across providers' naming",
          context_window("claude-haiku-4-5") == 200_000 and context_window("anthropic/claude-haiku-4.5") == 200_000
          and context_window("google/gemini-2.5-flash") == 1_048_576 and context_window("some-unknown-model") == 128_000)
    with mock.patch.dict(os.environ, {"AGENT_CONTEXT_WINDOW": "5000"}):
        check("AGENT_CONTEXT_WINDOW overrides", context_window("openai/gpt-oss-120b") == 5000)


def test_system_prompt_and_memory():
    d = tempfile.mkdtemp()
    plain = build_system_prompt(d)
    check("system prompt mentions the workspace and the edit tool", d in plain and "str_replace" in plain)
    check("system prompt is stable (cacheable prefix)", build_system_prompt(d) == plain)
    with open(os.path.join(d, "AGENTS.md"), "w") as f:
        f.write("Run tests with `make test`. Never edit generated/.")
    with_memory = build_system_prompt(d, "apply_edits")
    check("AGENTS.md is loaded into the system prompt", "Project notes (from AGENTS.md)" in with_memory
          and "make test" in with_memory and "apply_edits" in with_memory)


def test_cache_breakpoints():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"}]
    marked = agent.with_cache_breakpoints(msgs)
    check("system prompt and newest message get cache_control",
          marked[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
          and marked[-1]["content"][0]["cache_control"] == {"type": "ephemeral"})
    check("the stored history is not modified", msgs[0]["content"] == "sys" and msgs[-1]["content"] == "result")
    check("at most two breakpoints per request",
          json.dumps(marked).count("cache_control") == 2)


def test_agent_loop():
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    with open("AGENTS.md", "w") as f:
        f.write("The project uses tabs.")
    with open("huge.txt", "w") as f:
        f.write("".join(f"line {i} " + "y" * 90 + "\n" for i in range(2000)))
    agent.messages.clear()
    agent.turn_count = 0
    files._read_state.clear()
    agent.policy = agent.Policy(workdir, mode="auto", ask=None)
    agent.log_file = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    usage = {"prompt_tokens": 5000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 4096}}
    first = response(sse(tool_chunks([("c1", "read_file", {"path": "huge.txt"})])[:-1] + [{"choices": [], "usage": usage}]))
    with mock.patch.object(requests, "post", side_effect=[first, response(sse(text_chunks("done")))]) as post:
        agent.agentic_loop("look at huge.txt")
    sent = post.call_args_list[0].kwargs["json"]["messages"]
    check("the system prompt comes first, with AGENTS.md", sent[0]["role"] == "system" and "uses tabs" in sent[0]["content"])
    tool_msg = next(m for m in agent.messages if m["role"] == "tool")
    check("an oversized tool result is capped before it reaches the model",
          len(tool_msg["content"]) <= 30_500, len(tool_msg["content"]))
    events = [json.loads(line) for line in open(agent.log_file)]
    check("the clip is recorded in the transcript", any(e["type"] == "tool_result" and e["data"]["clipped"] for e in events))
    turn = next(e["data"] for e in events if e["type"] == "turn_usage")
    check("per-turn usage (with cached tokens) is logged", turn["calls"] == 2 and turn["cached"] == 4096, turn)


if __name__ == "__main__":
    home = os.getcwd()
    try:
        test_tool_arguments()
        test_rate_limit_with_text_code()
        test_read_file_caps()
        test_budget()
        test_system_prompt_and_memory()
        test_cache_breakpoints()
        test_agent_loop()
    finally:
        os.chdir(home)
    print("\nDay 8 tests passed.")
