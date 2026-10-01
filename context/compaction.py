"""Context engineering II: keep long conversations inside the window.

The history is split into atomic units; an assistant message that calls tools and the results of those
calls form ONE unit. Compaction replaces whole units in the middle of the conversation with a summary,
so a tool call is never separated from its result (the API rejects that).
"""
import json

from context.budget import estimate_tokens

SUMMARY_PREFIX = "[Summary of the earlier part of this conversation, written to save context]\n\n"

SUMMARIZER_PROMPT = """You compact the working memory of an AI coding agent. Below is the older part of its conversation. Write a summary the agent can rely on to continue the task WITHOUT seeing the original. Include:
- The user's goal and any constraints or preferences they stated.
- What has been done: files read, files changed (and how), commands run and their outcomes.
- Key facts discovered: exact file paths, function and variable names, error messages, test results.
- What failed or was ruled out, and why.
- The current state and the next steps.
Keep exact names, paths and commands. Be concise (at most about 400 words). Write the summary only."""


def split_units(messages):
    """Group messages into units that must be kept or dropped together."""
    units, i = [], 0
    while i < len(messages):
        unit = [messages[i]]
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                i += 1
                unit.append(messages[i])
        units.append(unit)
        i += 1
    return units


def validate_history(messages):
    """Problems that would make the API reject the conversation. Empty list = valid."""
    problems, pending = [], {}
    for index, m in enumerate(messages):
        role = m.get("role")
        if role == "tool":
            if m.get("tool_call_id") not in pending:
                problems.append(f"message {index}: tool result {m.get('tool_call_id')!r} has no matching tool call")
            pending.pop(m.get("tool_call_id"), None)
            continue
        if pending:
            problems.append(f"message {index}: tool call(s) {sorted(pending)} have no result before this {role} message")
            pending = {}
        if role == "assistant":
            pending = {call["id"]: index for call in m.get("tool_calls") or []}
    if pending:
        problems.append(f"end of conversation: tool call(s) {sorted(pending)} have no result")
    return problems


def render_for_summary(units, max_chars_per_item=1_500):
    """Plain-text transcript of units for the summarizer (long tool output shortened)."""
    def short(text):
        text = text if isinstance(text, str) else json.dumps(text)
        return text if len(text) <= max_chars_per_item else text[:max_chars_per_item] + " [...]"

    lines = []
    for unit in units:
        for m in unit:
            role = m.get("role")
            if role == "tool":
                lines.append(f"TOOL RESULT:\n{short(m.get('content') or '')}")
                continue
            if m.get("content"):
                lines.append(f"{role.upper()}: {short(m['content'])}")
            for call in m.get("tool_calls") or []:
                lines.append(f"TOOL CALL {call['function']['name']}: {short(call['function']['arguments'])}")
    return "\n\n".join(lines)


def compact(messages, summarize, keep_recent_tokens):
    """Return a shorter history: system prompt + first user request + summary + recent units.

    summarize(text) -> summary string, or None if summarizing failed (then nothing changes).
    Returns (new_messages, info) where info describes what happened.
    """
    units = split_units(messages)
    head = []
    while units and units[0][0].get("role") == "system":
        head += units.pop(0)
    if units and units[0][0].get("role") == "user":  # the original request stays verbatim
        head += units.pop(0)

    tail, tail_tokens = [], 0
    while units:
        size = estimate_tokens(units[-1])
        if tail and tail_tokens + size > keep_recent_tokens:
            break
        tail.insert(0, units.pop())
        tail_tokens += size
    middle = units
    if not middle:
        return messages, {"compacted": False, "reason": "nothing old enough to summarize"}
    if all(len(u) == 1 and str(u[0].get("content", "")).startswith(SUMMARY_PREFIX) for u in middle):
        # re-summarizing a summary costs a model call and saves nothing
        return messages, {"compacted": False, "reason": "only an earlier summary is left to compact"}

    summary = summarize(render_for_summary(middle))
    if not summary:
        return messages, {"compacted": False, "reason": "summarizer failed"}
    new = head + [{"role": "user", "content": SUMMARY_PREFIX + summary.strip()}] + [m for u in tail for m in u]
    return new, {"compacted": True, "units_summarized": len(middle),
                 "messages_before": len(messages), "messages_after": len(new),
                 "tokens_before": estimate_tokens(messages), "tokens_after": estimate_tokens(new)}
