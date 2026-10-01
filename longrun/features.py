"""feature_list.json: the long-term plan that survives between sessions.

The harness owns this file. The model proposes the list once (create_feature_list) and afterwards can
only *claim* a feature is done or blocked (update_feature); the harness decides the real status after
running the tests itself.
"""
import json
import os

from tools import tool

FEATURE_FILE = "feature_list.json"
STATUSES = ("pending", "in_progress", "failing", "done", "blocked")
MAX_FEATURES = 12

# Harness state for the running session (set by longrun.session)
STATE = {"workspace": None, "plan": None, "claims": {}, "allow_create": False}


def path(workspace):
    return os.path.join(workspace, FEATURE_FILE)


def load(workspace):
    with open(path(workspace)) as f:
        return json.load(f)


def save(workspace, plan):
    with open(path(workspace), "w") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")


def pick_next(plan):
    """Unfinished work first (in_progress, failing), then the next pending feature, in list order."""
    for wanted in (("in_progress", "failing"), ("pending",)):
        for feature in plan["features"]:
            if feature["status"] in wanted:
                return feature
    return None


def render(plan):
    marks = {"pending": "[ ]", "in_progress": "[>]", "failing": "[!]", "done": "[x]", "blocked": "[-]"}
    return "\n".join(f"{marks[f['status']]} {f['id']}. {f['title']} ({f['status']})" for f in plan["features"])


@tool(
    "Record the project plan as an ordered list of features. Each feature must be small enough to build and "
    "test in one session, and testable on its own. Order them so each builds on the previous ones.",
    {
        "test_command": {"type": "string", "description": "Shell command that runs the whole test suite, e.g. 'python -m unittest'"},
        "features": {"type": "array", "description": f"3 to {MAX_FEATURES} features in build order", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short name"},
                "description": {"type": "string", "description": "What to build, specific enough to implement"},
                "acceptance": {"type": "array", "items": {"type": "string"},
                               "description": "Checks that prove it works (each should be a test)"},
            },
            "required": ["title", "description", "acceptance"]}},
    },
)
def create_feature_list(test_command, features):
    if not STATE["allow_create"]:
        return "Error: the feature list already exists; it can only be created by the initializer session."
    if not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES:
        return f"Error: provide between 1 and {MAX_FEATURES} features."
    cleaned = []
    for n, item in enumerate(features, 1):
        if not isinstance(item, dict) or not str(item.get("title", "")).strip() or not str(item.get("description", "")).strip():
            return f"Error: feature {n} needs a title and a description."
        acceptance = item.get("acceptance") or []
        if not isinstance(acceptance, list) or not acceptance:
            return f"Error: feature {n} needs at least one acceptance check."
        cleaned.append({"id": n, "title": item["title"].strip(), "description": item["description"].strip(),
                        "acceptance": [str(a) for a in acceptance], "status": "pending", "attempts": 0, "notes": []})
    STATE["plan"]["test_command"] = str(test_command).strip() or "python -m unittest"
    STATE["plan"]["features"] = cleaned
    save(STATE["workspace"], STATE["plan"])
    return f"Saved {len(cleaned)} features to {FEATURE_FILE}:\n{render(STATE['plan'])}"


@tool(
    "Report the outcome of the feature you worked on this session. status 'done' means it is implemented, "
    "tested, and the whole test suite passes; the harness will re-run the tests to confirm. status 'blocked' "
    "means you cannot finish it; explain why in note.",
    {
        "id": {"type": "integer", "description": "Feature id"},
        "status": {"type": "string", "enum": ["done", "blocked"]},
        "note": {"type": "string", "description": "What you did, or why it is blocked"},
    },
)
def update_feature(id, status, note):
    plan = STATE["plan"]
    if not plan or not plan.get("features"):
        return "Error: there is no feature list yet."
    if status not in ("done", "blocked"):
        return "Error: status must be 'done' or 'blocked'."
    ids = [f["id"] for f in plan["features"]]
    if id not in ids:
        return f"Error: no feature with id {id}. Ids: {ids}."
    STATE["claims"][id] = {"status": status, "note": str(note)}
    return (f"Recorded: feature {id} claimed {status}. The harness will run `{plan['test_command']}` "
            "after this session to confirm. Now reply with your summary for the progress log.")
