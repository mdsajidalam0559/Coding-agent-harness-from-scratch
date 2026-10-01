"""Day 8: context engineering I - measure and budget.

On top of Day 7: a deliberate system prompt with project memory (AGENTS.md), per-turn token
accounting, a cap on every tool result, and prompt caching (a stable prefix everywhere, plus explicit
cache breakpoints for Anthropic models).
"""
import copy
import json
import os

from days import day_7_agent as base
import tools.shell
from context.budget import TokenBudget, clip_tool_result, context_window
from context.prompt import build_system_prompt
from safety.permissions import Policy
from safety.sandbox import DockerSandbox
from tools import execute_tool, tool_schemas

# Provider plumbing (URLs, keys, streaming, retries) is unchanged from Day 7
PROVIDERS = base.PROVIDERS
PROVIDER = base.PROVIDER
MODEL = base.MODEL
MAX_STEPS = base.MAX_STEPS
EDIT_FORMAT = base.EDIT_FORMAT
EDIT_TOOLS = base.EDIT_TOOLS
PERMISSION_MODE = base.PERMISSION_MODE
SANDBOX_MODE = base.SANDBOX_MODE
NETWORK = base.NETWORK
INTERRUPTED = base.INTERRUPTED
repair_history = base.repair_history

log_file = "agent_transcript.jsonl"
messages = []
turn_count = 0
policy = None
budget = None  # TokenBudget, created with the first message


def log_to_jsonl(event_type, data):
    from datetime import datetime
    with open(log_file, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(), "type": event_type, "data": data}) + "\n")


def tool_names():
    return ["read_file", "list_dir", "search", "write_file", EDIT_FORMAT, "bash"]


def with_cache_breakpoints(msgs):
    """Anthropic models only cache what you mark. Mark the system prompt and the newest message,
    so each request reuses everything up to the previous one. Returns a marked copy; the stored
    history stays unmarked so markers never pile up (Anthropic allows at most 4)."""
    marked = copy.deepcopy(msgs)
    targets = [m for m in (marked[0], marked[-1]) if m.get("role") in ("system", "user", "tool")]
    for m in {id(m): m for m in targets}.values():
        if isinstance(m.get("content"), str) and m["content"]:
            m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
    return marked


def build_payload():
    msgs = with_cache_breakpoints(messages) if MODEL.startswith("anthropic/") else messages
    return {"model": MODEL, "messages": msgs, "tools": tool_schemas(tool_names()), "max_tokens": 4000}


def resolve_tool_name(name, offered):
    """(real name, None) or (None, error). Accepts namespaced names like gpt-oss's `repo_browser.list_dir`."""
    if name in offered:
        return name, None
    short = name.rsplit(".", 1)[-1]
    if short in offered:
        return short, None
    return None, f"Error: there is no tool named {name!r}. Available tools: {', '.join(offered)}."


def run_tool_call(tool_call):
    """Parse, permission-check, run and size-cap one tool call. Returns the text the model sees."""
    requested, raw_args = tool_call["function"]["name"], tool_call["function"]["arguments"]
    name, name_error = resolve_tool_name(requested, tool_names())
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
        log_to_jsonl("permission", {"call_id": tool_call["id"], "tool": name, "allowed": allowed, "reason": reason})
        if allowed:
            print(f"  → {name} {json.dumps(args)[:120]}")
            result = execute_tool(name, args)
        else:
            print(f"  ⛔ {name} blocked: {reason}")
            result = (f"Permission denied: {reason}. Do not retry the same action; "
                      f"find another way to do the task, or explain to the user why you cannot.")
    full_length = len(result)
    result = clip_tool_result(result)
    print(f"    {result.splitlines()[0][:100] if result else '(empty)'}")
    log_to_jsonl("tool_result", {"call_id": tool_call["id"], "tool": name or requested, "args": raw_args,
                                 "result": result, "chars": full_length, "clipped": len(result) < full_length})
    return result


def start_session():
    """First message of a conversation: fresh budget and the system prompt (with AGENTS.md if present)."""
    global policy, budget
    if policy is None:
        policy = Policy(os.getcwd(), PERMISSION_MODE)
    budget = TokenBudget(context_window(MODEL))
    system = build_system_prompt(os.getcwd(), EDIT_FORMAT)
    messages.append({"role": "system", "content": system})
    log_to_jsonl("system_prompt", {"chars": len(system), "memory_file": "# Project notes" in system})


def agentic_loop(user_message):
    global turn_count
    if not messages:
        start_session()

    turn_count += 1
    budget.start_turn()
    messages.append({"role": "user", "content": user_message})
    log_to_jsonl("user_message", {"turn": turn_count, "content": user_message})
    try:
        _run_turn()
    finally:
        log_to_jsonl("turn_usage", {"turn": turn_count, **budget.turn, "context_tokens": budget.context_tokens})


def _run_turn():
    for step in range(1, MAX_STEPS + 1):
        repaired = repair_history(messages)
        if repaired:
            log_to_jsonl("history_repaired", {"missing_tool_results": repaired})

        payload = build_payload()
        log_to_jsonl("request", {"turn": turn_count, "step": step, "provider": PROVIDER, "payload": payload})

        streamed = []

        def show(piece):
            if not streamed:
                print("\nAssistant: ", end="", flush=True)
            streamed.append(piece)
            print(piece, end="", flush=True)

        try:
            response_data = base.call_model(payload, on_text=show, log=log_to_jsonl, provider=PROVIDER)
        except KeyboardInterrupt:
            partial = "".join(streamed)
            if partial:
                messages.append({"role": "assistant", "content": partial + " [interrupted by the user]"})
            log_to_jsonl("interrupted", {"turn": turn_count, "step": step, "during": "model", "partial": partial})
            print("\n⏹  Interrupted.")
            return
        if streamed:
            print()
        if response_data is None:
            return
        usage = budget.record(response_data.get("usage"))
        log_to_jsonl("response", {"turn": turn_count, "step": step, "body": response_data, "cached_tokens": usage["cached"]})
        message = response_data["choices"][0]["message"]

        if not message.get("tool_calls"):
            messages.append({"role": "assistant", "content": message.get("content") or ""})
            return

        message["content"] = message.get("content") or ""
        messages.append(message)
        try:
            for tool_call in message["tool_calls"]:
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": run_tool_call(tool_call)})
        except KeyboardInterrupt:
            filled = repair_history(messages)
            log_to_jsonl("interrupted", {"turn": turn_count, "step": step, "during": "tools", "unfinished": filled})
            print(f"\n⏹  Interrupted. {filled} unfinished tool call(s) marked as interrupted.")
            return

    print(f"Reached max steps ({MAX_STEPS}) for this turn, stopping")
    log_to_jsonl("max_steps_reached", {"turn": turn_count, "steps": MAX_STEPS})


def main():
    global PROVIDER, MODEL
    PROVIDER, MODEL = base.parse_args("Day 8 coding agent (system prompt, AGENTS.md, token budget)")
    print("Starting Day 8: Context-Budgeted Agent (Ctrl-C interrupts the agent; Ctrl-C at the prompt exits)\n")
    sandbox = None
    if SANDBOX_MODE == "docker":
        sandbox = DockerSandbox(os.getcwd(), network=NETWORK).start()
        tools.shell.SANDBOX = sandbox
        print(f"🔒 bash runs in container {sandbox.name} (network {'on' if NETWORK else 'off'})")
    else:
        print("⚠️  No sandbox: bash runs directly on this machine")
    from context.prompt import load_memory
    memory_name, _ = load_memory(os.getcwd())
    print(f"Provider: {PROVIDER}   Model: {MODEL} ({context_window(MODEL) // 1000}k context)   "
          f"Permission mode: {PERMISSION_MODE}   Project notes: {memory_name or 'none (add AGENTS.md)'}\n")
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input:
                try:
                    agentic_loop(user_input)
                except KeyboardInterrupt:
                    print("\n⏹  Interrupted.")
                print(f"  [{budget.status()}]")
    finally:
        if sandbox:
            sandbox.stop()


if __name__ == "__main__":
    main()
