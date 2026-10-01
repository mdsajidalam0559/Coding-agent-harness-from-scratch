"""Day 12: extensibility. Day 10 agent + MCP tools (client written from the spec), hooks around
every tool call, and skills loaded on demand. See ext/.

From Day 10: planning. Day 9 agent + a todo checklist tool; long-running work lives in longrun/.

From Day 9: context engineering II - compaction and subagents.

On top of Day 8:
- The history is validated before every API call (every tool call must have its result).
- Near the context limit, old turns are summarized; tool calls and their results are kept or
  summarized together, never split.
- A `delegate` tool runs a subtask in a fresh context and returns only the subagent's final report.
- Retrieval stays on demand: nothing is loaded up front; the agent searches and reads what it needs.
"""
import json
import os

from days import day_7_agent as base
from days import day_8_agent as d8
import tools.shell
from context.budget import TokenBudget, clip_tool_result, context_window, estimate_tokens
from context.compaction import SUMMARIZER_PROMPT, compact, validate_history
from context.prompt import build_system_prompt, load_memory
from safety.permissions import Policy
from safety.sandbox import DockerSandbox
from tools import execute_tool, tool, tool_schemas
from tools import todo as todo_tool
from ext import hooks as hooks_mod, mcp_tools, skills
from ext.trust import terminal_ask

PROVIDERS = base.PROVIDERS
PROVIDER = base.PROVIDER
MODEL = base.MODEL
MAX_STEPS = base.MAX_STEPS
SUB_MAX_STEPS = 15
EDIT_FORMAT = base.EDIT_FORMAT
EDIT_TOOLS = base.EDIT_TOOLS
PERMISSION_MODE = base.PERMISSION_MODE
SANDBOX_MODE = base.SANDBOX_MODE
NETWORK = base.NETWORK
COMPACT_AT = float(os.getenv("AGENT_COMPACT_AT", "0.7"))  # compact when the estimate passes this share of the window
KEEP_RECENT = 0.25  # share of the window kept verbatim (most recent units) when compacting
repair_history = base.repair_history

log_file = "agent_transcript.jsonl"
messages = []
turn_count = 0
policy = None
budget = None

SUBAGENT_SUFFIX = """

You are running as a SUBAGENT. Another agent delegated this one focused task to you and will see ONLY your final reply, not your tool calls. When you are done, reply with a complete, self-contained report: the answer or result, the evidence (file paths, line numbers, relevant output), and anything you changed. Do not ask questions; do the best you can with what you have."""


def log_to_jsonl(event_type, data):
    from datetime import datetime
    with open(log_file, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(), "type": event_type, "data": data}) + "\n")


EXTRA_TOOLS = []  # set by longrun/ to offer create_feature_list or update_feature during long-running sessions
MCP_TOOLS = []    # names of tools registered from connected MCP servers
MCP_CLIENTS = []
HOOKS = hooks_mod.Hooks()
HOOKS.after_tool(hooks_mod.python_syntax_check, matcher="write_file|str_replace|apply_edits")


def tool_names():
    names = ["read_file", "list_dir", "search", "write_file", EDIT_FORMAT, "bash", "todo_write", "delegate"]
    if skills.STATE["skills"]:
        names.append("load_skill")
    return names + MCP_TOOLS + EXTRA_TOOLS


def system_prompt():
    """Stable for the whole session (cacheable): base prompt + AGENTS.md + skills list + MCP server notes."""
    return (build_system_prompt(os.getcwd(), EDIT_FORMAT) + skills.prompt_section(skills.STATE["skills"])
            + mcp_tools.server_instructions(MCP_CLIENTS))


def setup_extensions(workspace, trust_project=False, ask=None, mcp_config=None, syntax_check=True, log=print):
    """Connect MCP servers, load hooks and discover skills for this workspace."""
    global HOOKS
    HOOKS = hooks_mod.load_hooks(workspace, trust_project, ask, syntax_check)
    found, problems = skills.discover(workspace)
    for problem in problems:
        log(f"⚠️  skill ignored: {problem}")
    if found:
        log(f"🧩 skills: {', '.join(found)}")
    configs = mcp_tools.load_config(workspace, trust_project, ask, mcp_config)
    clients, names = mcp_tools.connect_all(workspace, configs, log)
    MCP_CLIENTS[:] = clients
    MCP_TOOLS[:] = names
    return {"skills": list(found), "mcp_servers": [c.name for c in clients], "mcp_tools": names}


def close_extensions():
    for client in MCP_CLIENTS:
        client.close()
    mcp_tools.unregister(MCP_TOOLS)
    MCP_CLIENTS.clear()
    MCP_TOOLS.clear()


# Note: tools are registered globally by name, so import only one day's agent per process
# (day_9_agent also registers a `delegate`).
@tool(
    "Hand a focused subtask to a subagent that starts with a fresh, empty context and returns only its final "
    "report. Use it for self-contained work whose details you do not need to keep, e.g. investigating how "
    "something works across many files, or tracking down where a bug comes from. The subagent cannot see this "
    "conversation, so put everything it needs in `task`, including what the report should contain. It has the "
    "same tools as you, except delegate.",
    {"task": {"type": "string", "description": "Complete, self-contained instructions, including what to report back"}},
)
def delegate(task):
    return run_subagent(task)


def run_subagent(task):
    sub = [{"role": "system", "content": system_prompt() + SUBAGENT_SUFFIX},
           {"role": "user", "content": task}]
    print("    ↳ subagent started")
    log_to_jsonl("subagent_start", {"task": task})
    sub_tools = [n for n in tool_names() if n not in ("delegate", "todo_write") and n not in EXTRA_TOOLS]
    answer = run_agent(sub, sub_tools, SUB_MAX_STEPS, label="sub", stream=False)
    log_to_jsonl("subagent_end", {"answered": answer is not None, "messages": len(sub)})
    print("    ↳ subagent finished")
    if answer is not None:
        return answer
    notes = [m["content"] for m in sub if m.get("role") == "assistant" and m.get("content")]
    return ("The subagent stopped before finishing (step limit or API error)."
            + (f" Its last notes:\n{notes[-1]}" if notes else ""))


def summarize(text):
    """One model call that turns the old part of a conversation into a summary."""
    body = base.call_model({"model": MODEL, "max_tokens": 1500, "messages": [
        {"role": "system", "content": SUMMARIZER_PROMPT}, {"role": "user", "content": text}]},
        log=log_to_jsonl, provider=PROVIDER)
    if body is None:
        return None
    budget.record(body.get("usage"))
    log_to_jsonl("response", {"purpose": "compaction", "body": body})
    return body["choices"][0]["message"].get("content")


def maybe_compact(msgs, names, label):
    window = context_window(MODEL)
    size = estimate_tokens(msgs) + estimate_tokens(tool_schemas(names))
    if size < COMPACT_AT * window:
        return
    print(f"  🗜  context ~{size // 1000}k of {window // 1000}k tokens: summarizing older turns...")
    new, info = compact(msgs, summarize, keep_recent_tokens=int(KEEP_RECENT * window))
    log_to_jsonl("compaction", {"agent": label, "estimate_before": size, **info})
    if info["compacted"]:
        msgs[:] = new  # in place: callers hold a reference to this list
        print(f"  🗜  {info['units_summarized']} older units summarized "
              f"(~{info['tokens_before'] // 1000}k → ~{info['tokens_after'] // 1000}k tokens)")
    else:
        print(f"  🗜  could not compact: {info['reason']}")


def check_history(msgs, label):
    """Validator: run before every API call. Repairs what it can and logs what it found."""
    problems = validate_history(msgs)
    if not problems:
        return
    log_to_jsonl("history_invalid", {"agent": label, "problems": problems})
    repair_history(msgs)
    remaining = validate_history(msgs)
    if remaining:  # a harness bug; the API would reject this request
        raise RuntimeError(f"invalid conversation history: {remaining}")


def run_tool_call(tool_call, label, names):
    requested, raw_args = tool_call["function"]["name"], tool_call["function"]["arguments"]
    indent = "    ↳ " if label == "sub" else "  → "
    name, name_error = d8.resolve_tool_name(requested, names)  # only tools this agent was offered
    args = None
    if name_error:
        result = name_error
    else:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as e:
            result = f"Error: arguments were not valid JSON ({e}). Raw arguments: {raw_args}"
    if args is not None:
        allowed, reason = policy.check(name, args)
        log_to_jsonl("permission", {"agent": label, "call_id": tool_call["id"], "tool": name,
                                    "allowed": allowed, "reason": reason})
        if allowed:
            allowed, args, hook_reason = HOOKS.run_before(name, args)
            if not allowed:
                print(f"{indent}🪝 {name} blocked by a hook: {hook_reason}")
                log_to_jsonl("hook_blocked", {"agent": label, "call_id": tool_call["id"], "tool": name,
                                              "reason": hook_reason})
                result = f"Blocked: {hook_reason}. Do not retry the same action; address the reason or find another way."
            else:
                print(f"{indent}{name} {json.dumps(args)[:110]}")
                result = HOOKS.run_after(name, args, execute_tool(name, args))
        else:
            print(f"{indent}⛔ {name} blocked: {reason}")
            result = (f"Permission denied: {reason}. Do not retry the same action; "
                      f"find another way to do the task, or explain to the user why you cannot.")
    full_length = len(result)
    result = clip_tool_result(result)
    log_to_jsonl("tool_result", {"agent": label, "call_id": tool_call["id"], "tool": name or requested,
                                 "args": raw_args, "result": result, "chars": full_length,
                                 "clipped": len(result) < full_length})
    return result


def run_agent(msgs, names, max_steps, label="main", stream=True):
    """Drive the model on `msgs` until it replies without tool calls.

    Returns the final reply text, or None if it stopped early (step limit, API error).
    KeyboardInterrupt propagates to the caller after the history has been made valid again.
    """
    for step in range(1, max_steps + 1):
        check_history(msgs, label)
        maybe_compact(msgs, names, label)
        payload = {"model": MODEL, "tools": tool_schemas(names), "max_tokens": 4000,
                   "messages": d8.with_cache_breakpoints(msgs) if MODEL.startswith("anthropic/") else msgs}
        log_to_jsonl("request", {"agent": label, "turn": turn_count, "step": step, "provider": PROVIDER,
                                 "payload": payload})

        streamed = []

        def show(piece):
            if not streamed:
                print("\nAssistant: ", end="", flush=True)
            streamed.append(piece)
            print(piece, end="", flush=True)

        try:
            body = base.call_model(payload, on_text=show if stream else None, log=log_to_jsonl, provider=PROVIDER)
        except KeyboardInterrupt:
            if streamed:
                msgs.append({"role": "assistant", "content": "".join(streamed) + " [interrupted by the user]"})
            raise
        if streamed:
            print()
        if body is None:
            return None
        usage = budget.record(body.get("usage"))
        log_to_jsonl("response", {"agent": label, "turn": turn_count, "step": step, "body": body,
                                  "cached_tokens": usage["cached"]})
        message = body["choices"][0]["message"]

        if not message.get("tool_calls"):
            text = message.get("content") or ""
            msgs.append({"role": "assistant", "content": text})
            return text

        message["content"] = message.get("content") or ""
        msgs.append(message)
        try:
            for tool_call in message["tool_calls"]:
                msgs.append({"role": "tool", "tool_call_id": tool_call["id"],
                             "content": run_tool_call(tool_call, label, names)})
        except KeyboardInterrupt:
            repair_history(msgs)
            raise

    print(f"Reached max steps ({max_steps}), stopping" + (" the subagent" if label == "sub" else ""))
    log_to_jsonl("max_steps_reached", {"agent": label, "turn": turn_count, "steps": max_steps})
    return None


def agentic_loop(user_message):
    global turn_count, policy, budget
    if not messages:
        todo_tool.reset()  # the checklist belongs to one conversation
        if policy is None:
            policy = Policy(os.getcwd(), PERMISSION_MODE)
        budget = TokenBudget(context_window(MODEL))
        system = system_prompt()
        messages.append({"role": "system", "content": system})
        log_to_jsonl("system_prompt", {"chars": len(system), "memory_file": "# Project notes" in system})

    turn_count += 1
    budget.start_turn()
    messages.append({"role": "user", "content": user_message})
    log_to_jsonl("user_message", {"turn": turn_count, "content": user_message})
    try:
        run_agent(messages, tool_names(), MAX_STEPS)
    except KeyboardInterrupt:
        filled = repair_history(messages)
        log_to_jsonl("interrupted", {"turn": turn_count, "unfinished_tool_calls": filled})
        print("\n⏹  Interrupted.")
    finally:
        log_to_jsonl("turn_usage", {"turn": turn_count, **budget.turn, "context_tokens": budget.context_tokens})


def main():
    global PROVIDER, MODEL
    import argparse
    parser = argparse.ArgumentParser(description="Day 12 coding agent (MCP, hooks, skills)")
    parser.add_argument("--provider", default=PROVIDER, choices=list(PROVIDERS))
    parser.add_argument("--model", help="model id (default: AGENT_MODEL, else the provider's default)")
    parser.add_argument("--mcp-config", help="an extra mcpServers JSON file to load")
    parser.add_argument("--trust-project", action="store_true",
                        help="use this project's .agent/mcp.json and .agent/hooks.json without asking")
    parser.add_argument("--no-syntax-check", action="store_true", help="disable the built-in Python syntax hook")
    args = parser.parse_args()
    PROVIDER = args.provider
    MODEL = args.model or os.getenv("AGENT_MODEL") or PROVIDERS[PROVIDER]["default_model"]
    base.provider_settings(PROVIDER)

    print("Starting Day 12: Extensible Agent (Ctrl-C interrupts; Ctrl-C at the prompt exits)\n")
    workspace = os.getcwd()
    info = setup_extensions(workspace, args.trust_project, terminal_ask, args.mcp_config, not args.no_syntax_check)
    user_skills = os.path.isdir(skills.USER_DIR) and any(s["source"] == "user" for s in skills.STATE["skills"].values())
    global policy
    policy = Policy(workspace, PERMISSION_MODE, readable_roots=[skills.USER_DIR] if user_skills else ())
    sandbox = None
    try:
        if SANDBOX_MODE == "docker":
            mounts = [(skills.USER_DIR, skills.SANDBOX_USER_DIR)] if user_skills else []
            sandbox = DockerSandbox(workspace, network=NETWORK, mounts=mounts).start()
            tools.shell.SANDBOX = sandbox
            skills.STATE["user_dir_in_sandbox"] = skills.SANDBOX_USER_DIR if user_skills else None
            print(f"🔒 bash runs in container {sandbox.name} (network {'on' if NETWORK else 'off'})")
        else:
            print("⚠️  No sandbox: bash runs directly on this machine")
        if MCP_CLIENTS:
            print("   MCP servers run on this machine, outside the sandbox; their tools ask for approval "
                  "unless the server marks them read-only.")
        memory_name, _ = load_memory(workspace)
        print(f"Provider: {PROVIDER}   Model: {MODEL} ({context_window(MODEL) // 1000}k context)   "
              f"Permission mode: {PERMISSION_MODE}   Project notes: {memory_name or 'none'}   "
              f"MCP tools: {len(info['mcp_tools'])}   Skills: {len(info['skills'])}\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input:
                agentic_loop(user_input)
                for warning in HOOKS.warnings:
                    print(f"  ⚠️  {warning}")
                HOOKS.warnings.clear()
                print(f"  [{budget.status()}]")
    finally:
        if sandbox:
            sandbox.stop()
        close_extensions()


if __name__ == "__main__":
    main()
