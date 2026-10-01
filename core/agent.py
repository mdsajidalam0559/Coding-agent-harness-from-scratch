"""The agent core: one loop, any model, any tool set.

An Agent owns its state (conversation, token budget, learned limits, transcript), so several agents can
live in one process. What makes it a *coding* agent or a *data* agent is only its AgentConfig: the tools
it is offered and its system prompt (see agents/). The model is any ModelAdapter (see models/).

Per step: validate history -> compact if near the window -> call the model -> run each requested tool
(permission check, hooks, execute, cap the output) -> repeat until the model answers without tools.
"""
import contextvars
import json
import os
from dataclasses import dataclass, field
from datetime import datetime

from context.budget import CHARS_PER_TOKEN, TokenBudget, clip_tool_result, context_overflow, estimate_tokens
from context.compaction import SUMMARIZER_PROMPT, compact, validate_history
from models.base import ModelError, ModelResponse
from tools import execute_tool, tool, tool_schemas
from tools.registry import REGISTRY, ToolContext, reset_context, set_context

INTERRUPTED = "Interrupted by the user before this finished. Wait for the user's next instruction."
SHRUNK = "[output removed to fit the context window; run the tool again if you still need it]"
CURRENT = contextvars.ContextVar("current_agent")  # the agent running a tool (for delegate)


@dataclass
class AgentConfig:
    name: str
    tools: list                      # tool names from the registry
    system_prompt: object            # str, or callable(agent) -> str (built once per conversation)
    max_steps: int = 25
    sub_max_steps: int = 15
    subagent_suffix: str = ""        # appended to the system prompt of subagents
    no_subagent_tools: tuple = ("delegate", "ask_user", "todo_write")
    compact_at: float = 0.7          # compact when the estimate passes this share of the window
    keep_recent: float = 0.25        # share of the window kept verbatim when compacting
    tool_output_max_chars: int = 10_000
    tool_output_window_share: float = 0.15
    max_overflow_recoveries: int = 3
    max_output_tokens: int = 4000
    log_label: str = "main"          # how this agent's steps are labelled in a shared transcript


class NullUI:
    """No output (headless). ConsoleUI in ui/ prints the same events."""
    def text(self, piece): pass
    def text_end(self): pass
    def tool_start(self, agent, name, args): pass
    def tool_end(self, agent, name, result): pass
    def blocked(self, agent, name, reason): pass
    def status(self, message): pass


@dataclass
class TurnResult:
    status: str                      # done | max_steps | api_error | interrupted
    text: str = ""
    steps: int = 0
    usage: dict = field(default_factory=dict)
    error: str | None = None


class Agent:
    def __init__(self, config, model, workspace, policy, hooks=None, log_file=None, ui=None, sandbox=None):
        self.config, self.model = config, model
        self.workspace = os.path.realpath(workspace)
        self.policy, self.hooks, self.ui = policy, hooks, ui or NullUI()
        self.log_file = log_file or os.path.join(self.workspace, ".agent", "transcript.jsonl")
        self.model.log = self.log
        self.messages = []
        self.turn_count = 0
        self.learned_window = None
        self.last_api_error = {}
        self.budget = TokenBudget(self.window())
        # what tools see while this agent runs: its workspace, its sandbox, the files it has read
        self.tool_context = ToolContext(self.workspace, sandbox)

    @property
    def sandbox(self):
        return self.tool_context.sandbox

    @sandbox.setter
    def sandbox(self, value):
        self.tool_context.sandbox = value

    # ---------- bookkeeping ----------

    def log(self, event_type, data):
        if event_type == "api_error":
            self.last_api_error = dict(data)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps({"timestamp": datetime.now().isoformat(), "type": event_type, "data": data},
                               default=str) + "\n")

    def window(self):
        known = self.model.context_window
        return min(known, self.learned_window) if self.learned_window else known

    def tool_output_limit(self):
        c = self.config
        return max(1_000, min(c.tool_output_max_chars, int(self.window() * CHARS_PER_TOKEN * c.tool_output_window_share)))

    def tool_names(self):
        return [name for name in self.config.tools if name in REGISTRY]

    def system_prompt(self):
        prompt = self.config.system_prompt
        return prompt(self) if callable(prompt) else prompt

    # ---------- public API ----------

    def run(self, user_message):
        """One user turn. Returns a TurnResult; never raises for model or tool failures."""
        if not self.messages:
            system = self.system_prompt()
            self.messages.append({"role": "system", "content": system})
            self.log("system_prompt", {"agent": self.config.name, "model": self.model.name, "chars": len(system)})
        self.turn_count += 1
        self.budget.start_turn()
        self.messages.append({"role": "user", "content": user_message})
        self.log("user_message", {"turn": self.turn_count, "content": user_message})
        token, context_token = CURRENT.set(self), set_context(self.tool_context)
        try:
            result = self.loop(self.messages, self.tool_names(), self.config.max_steps, label=self.config.log_label,
                               stream=True)
        except KeyboardInterrupt:
            filled = repair_history(self.messages)
            self.log("interrupted", {"turn": self.turn_count, "unfinished_tool_calls": filled})
            result = TurnResult("interrupted")
        finally:
            reset_context(context_token)
            CURRENT.reset(token)
            self.log("turn_usage", {"turn": self.turn_count, **self.budget.turn,
                                    "context_tokens": self.budget.context_tokens})
        result.usage = dict(self.budget.turn)
        return result

    def run_subagent(self, task):
        sub = [{"role": "system", "content": self.system_prompt() + self.config.subagent_suffix},
               {"role": "user", "content": task}]
        names = [n for n in self.tool_names() if n not in self.config.no_subagent_tools]
        self.log("subagent_start", {"task": task})
        # same workspace and sandbox, but its own record of read files: it starts knowing nothing
        context_token = set_context(ToolContext(self.workspace, self.sandbox))
        try:
            result = self.loop(sub, names, self.config.sub_max_steps, label="sub", stream=False)
        finally:
            reset_context(context_token)
        self.log("subagent_end", {"status": result.status, "messages": len(sub)})
        if result.status == "done":
            return result.text
        notes = [m["content"] for m in sub if m.get("role") == "assistant" and m.get("content")]
        return (f"The subagent stopped before finishing ({result.status})."
                + (f" Its last notes:\n{notes[-1]}" if notes else ""))

    # ---------- the loop ----------

    def loop(self, msgs, names, max_steps, label="main", stream=True):
        step, recoveries = 0, 0
        while step < max_steps:
            step += 1
            self.check_history(msgs, label)
            self.maybe_compact(msgs, names, label)
            self.log("request", {"agent": label, "turn": self.turn_count, "step": step, "model": self.model.name,
                                 "payload": {"messages": msgs, "tools": tool_schemas(names)}})
            streamed = []

            def on_text(piece):
                streamed.append(piece)
                self.ui.text(piece)

            self.model.log = self.log  # the adapter may be shared (e.g. generator and evaluator): log to this agent
            try:
                response = self.model.complete(msgs, tool_schemas(names), on_text=on_text if stream else None,
                                               max_tokens=self.config.max_output_tokens)
            except KeyboardInterrupt:
                if streamed:
                    msgs.append({"role": "assistant", "content": "".join(streamed) + " [interrupted by the user]"})
                raise
            except ModelError as e:
                if streamed:
                    self.ui.text_end()
                overflow, limit = context_overflow(self.last_api_error)
                if overflow and recoveries < self.config.max_overflow_recoveries and self.recover(msgs, names, label, limit):
                    recoveries += 1
                    step -= 1
                    continue
                return TurnResult("api_error", steps=step - 1, error=str(e))  # steps = completed model calls
            if streamed:
                self.ui.text_end()
            recoveries = 0
            usage = self.budget.record(response.usage)
            message = response.as_message()
            self.log("response", {"agent": label, "turn": self.turn_count, "step": step, "cached_tokens": usage["cached"],
                                  "body": {"choices": [{"finish_reason": response.finish_reason, "message": message}],
                                           "usage": response.usage}})

            if response.format_error:  # text protocol: let the model fix its own formatting
                msgs.append({"role": "assistant", "content": response.text})
                msgs.append({"role": "user", "content": response.format_error})
                self.log("format_error", {"agent": label, "error": response.format_error})
                continue
            if not response.tool_calls:
                msgs.append({"role": "assistant", "content": response.text})
                return TurnResult("done", text=response.text, steps=step)

            msgs.append(message)
            try:
                for call in response.tool_calls:
                    msgs.append({"role": "tool", "tool_call_id": call["id"], "content": self.run_tool(call, label, names)})
            except KeyboardInterrupt:
                repair_history(msgs)
                raise
        self.log("max_steps_reached", {"agent": label, "turn": self.turn_count, "steps": max_steps})
        return TurnResult("max_steps", steps=max_steps)

    def run_tool(self, call, label, names):
        requested, raw_args = call["function"]["name"], call["function"]["arguments"]
        name, error = resolve_tool_name(requested, names)
        args = None
        if error:
            result = error
        else:
            try:
                args = json.loads(raw_args or "{}")
                if not isinstance(args, dict):
                    raise json.JSONDecodeError("arguments must be a JSON object", raw_args, 0)
            except json.JSONDecodeError as e:
                args, result = None, f"Error: arguments were not valid JSON ({e}). Raw arguments: {raw_args}"
        if args is not None:
            allowed, reason = self.policy.check(name, args)
            self.log("permission", {"agent": label, "call_id": call["id"], "tool": name, "allowed": allowed, "reason": reason})
            if not allowed:
                self.ui.blocked(label, name, reason)
                result = (f"Permission denied: {reason}. Do not retry the same action; "
                          f"find another way to do the task, or explain to the user why you cannot.")
            else:
                if self.hooks:
                    allowed, args, reason = self.hooks.run_before(name, args)
                if not allowed:
                    self.ui.blocked(label, name, reason)
                    self.log("hook_blocked", {"agent": label, "call_id": call["id"], "tool": name, "reason": reason})
                    result = f"Blocked: {reason}. Do not retry the same action; address the reason or find another way."
                else:
                    self.ui.tool_start(label, name, args)
                    result = execute_tool(name, args)
                    if self.hooks:
                        result = self.hooks.run_after(name, args, result)
        full_length = len(result)
        result = clip_tool_result(result, limit=self.tool_output_limit())
        self.ui.tool_end(label, name or requested, result)
        self.log("tool_result", {"agent": label, "call_id": call["id"], "tool": name or requested, "args": raw_args,
                                 "result": result, "chars": full_length, "clipped": len(result) < full_length})
        return result

    # ---------- context management ----------

    def check_history(self, msgs, label):
        problems = validate_history(msgs)
        if problems:
            self.log("history_invalid", {"agent": label, "problems": problems})
            repair_history(msgs)
            remaining = validate_history(msgs)
            if remaining:
                raise RuntimeError(f"invalid conversation history: {remaining}")

    def summarize(self, text):
        cap = int(self.window() * CHARS_PER_TOKEN * 0.5)  # the summarizer's own request must fit
        if len(text) > cap:
            text = "[... the oldest part is omitted ...]\n" + text[-cap:]
        self.model.log = self.log
        try:
            response = self.model.complete([{"role": "system", "content": SUMMARIZER_PROMPT},
                                            {"role": "user", "content": text}], tools=None, max_tokens=1500)
        except ModelError:
            return None
        self.budget.record(response.usage)
        self.log("response", {"purpose": "compaction", "body": {"choices": [{"message": response.as_message()}],
                                                                "usage": response.usage}})
        return response.text

    def maybe_compact(self, msgs, names, label):
        window = self.window()
        fixed = estimate_tokens(msgs[:2]) + estimate_tokens(tool_schemas(names))  # system prompt, request, tools
        size = estimate_tokens(msgs) + estimate_tokens(tool_schemas(names))
        if size < self.config.compact_at * window or size - fixed <= self.config.keep_recent * window:
            return  # small enough, or nothing beyond what compaction would keep anyway
        self.ui.status(f"context ~{size // 1000}k of {window // 1000}k tokens: summarizing older turns")
        new, info = compact(msgs, self.summarize, keep_recent_tokens=int(self.config.keep_recent * window))
        self.log("compaction", {"agent": label, "estimate_before": size, **info})
        if info["compacted"]:
            msgs[:] = new

    def recover(self, msgs, names, label, limit):
        """After 'request too large': learn the real limit, compact or shrink, and signal a retry."""
        if limit and (self.learned_window is None or limit < self.learned_window):
            self.learned_window = limit
            self.budget.window = self.window()
        window, before = self.window(), estimate_tokens(msgs)
        self.ui.status(f"request too large for the provider (limit {limit or 'unknown'}); making room")
        new, info = compact(msgs, self.summarize, keep_recent_tokens=int(self.config.keep_recent * window))
        if info["compacted"]:
            msgs[:] = new
        shrunk = shrink_tool_outputs(msgs, int(0.6 * window) - estimate_tokens(tool_schemas(names)))
        after = estimate_tokens(msgs)
        self.log("overflow_recovery", {"agent": label, "limit": limit, "window": window, "compacted": info["compacted"],
                                       "shrunk_items": shrunk, "tokens_before": before, "tokens_after": after})
        return after < before


# ---------- helpers ----------

def repair_history(messages):
    """Give every tool call a result (the API rejects a history with an unanswered call). Returns how many."""
    repaired, i = 0, 0
    while i < len(messages):
        message = messages[i]
        i += 1
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        answered = set()
        while i < len(messages) and messages[i].get("role") == "tool":
            answered.add(messages[i]["tool_call_id"])
            i += 1
        for call in message["tool_calls"]:
            if call["id"] not in answered:
                messages.insert(i, {"role": "tool", "tool_call_id": call["id"], "content": INTERRUPTED})
                i += 1
                repaired += 1
    return repaired


def resolve_tool_name(name, offered):
    """(real name, None) or (None, error). Accepts namespaced names like gpt-oss's `repo_browser.list_dir`."""
    if name in offered:
        return name, None
    short = name.rsplit(".", 1)[-1]
    if short in offered:
        return short, None
    return None, f"Error: there is no tool named {name!r}. Available tools: {', '.join(offered)}."


def shrink_tool_outputs(msgs, target_tokens):
    """Blank the largest old tool results and file contents in old tool arguments, never the latest step."""
    last_assistant = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=len(msgs))
    candidates = []
    for i, m in enumerate(msgs[:last_assistant]):
        if m.get("role") == "tool" and len(m.get("content") or "") > len(SHRUNK):
            candidates.append((i, None))
        for n, call in enumerate(m.get("tool_calls") or []):
            if len(call["function"]["arguments"]) > 400:
                candidates.append((i, n))
    shrunk = 0
    for i, n in candidates:
        if estimate_tokens(msgs) <= target_tokens:
            break
        if n is None:
            msgs[i]["content"] = SHRUNK
        else:
            try:
                args = json.loads(msgs[i]["tool_calls"][n]["function"]["arguments"])
            except json.JSONDecodeError:
                continue
            for key in ("content", "new_str", "old_str", "edits"):
                if isinstance(args.get(key), str) and len(args[key]) > 200:
                    args[key] = "[omitted to fit the context window]"
            msgs[i]["tool_calls"][n]["function"]["arguments"] = json.dumps(args)
        shrunk += 1
    return shrunk


@tool(
    "Hand a focused subtask to a subagent that starts with a fresh, empty context and returns only its final "
    "report. Use it for self-contained work whose details you do not need to keep. The subagent cannot see "
    "this conversation, so put everything it needs in `task`, including what the report should contain.",
    {"task": {"type": "string", "description": "Complete, self-contained instructions, including what to report back"}},
)
def delegate(task):
    agent = CURRENT.get(None)
    if agent is None:
        return "Error: delegate can only be used inside an agent."
    return agent.run_subagent(task)
