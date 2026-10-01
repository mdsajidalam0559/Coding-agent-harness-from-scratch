"""A checklist the agent keeps for itself during one session (the long-term plan lives in files)."""
from .registry import tool

STATUSES = ("pending", "in_progress", "completed")
MARKS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}

_todos = []


def reset():
    _todos.clear()


def current():
    return list(_todos)


def render(items):
    done = sum(item["status"] == "completed" for item in items)
    lines = [f"{MARKS[item['status']]} {item['content']}" for item in items]
    return f"Todo list ({done}/{len(items)} done):\n" + "\n".join(lines)


@tool(
    "Plan and track the steps of the current task as a short checklist. Use it for tasks with 3 or more "
    "steps: write the plan first, then update it as you go. Always send the COMPLETE list (it replaces the "
    "previous one). Exactly one item should be in_progress while you work; mark items completed as soon as "
    "they are done, not in batches.",
    {"items": {"type": "array", "description": "The full checklist, in order", "items": {
        "type": "object",
        "properties": {"content": {"type": "string", "description": "What to do, as a short imperative"},
                       "status": {"type": "string", "enum": list(STATUSES)}},
        "required": ["content", "status"]}}},
)
def todo_write(items):
    if not isinstance(items, list) or not items:
        return "Error: items must be a non-empty list of {content, status}."
    cleaned = []
    for n, item in enumerate(items, 1):
        if not isinstance(item, dict) or not str(item.get("content", "")).strip():
            return f"Error: item {n} needs a non-empty content."
        if item.get("status") not in STATUSES:
            return f"Error: item {n} has status {item.get('status')!r}; use one of {', '.join(STATUSES)}."
        cleaned.append({"content": str(item["content"]).strip(), "status": item["status"]})
    in_progress = sum(item["status"] == "in_progress" for item in cleaned)
    if in_progress > 1:
        return f"Error: {in_progress} items are in_progress; keep exactly one in progress at a time."
    _todos[:] = cleaned
    return render(cleaned)
