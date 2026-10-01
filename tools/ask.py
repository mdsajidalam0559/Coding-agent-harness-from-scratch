"""ask_user: the model asks the user a multiple-choice question in the middle of a task.

The answer comes back as the tool result, so the model carries on in the same turn instead of ending
it with a question. Unattended runs (evals, long-running sessions) have no one to ask: the tool then
tells the model to choose itself and say what it chose, so the run never stalls.
"""
from .registry import tool

MIN_OPTIONS, MAX_OPTIONS = 2, 4
HEADLESS_REPLY = ("No user is available to answer (this is an unattended run). Choose the most reasonable option "
                  "yourself, continue, and say in your final answer which option you chose and why.")
SKIPPED_REPLY = ("The user skipped the question. Proceed with the most reasonable option and say in your final "
                 "answer which one you chose.")

# Set by an interactive agent: handler(question, options, multi_select) -> list of answers, or None if skipped.
HANDLER = None


@tool(
    "Ask the user a multiple-choice question and wait for the answer. Use it only when a decision changes the "
    "outcome and you cannot settle it from the code or the request: choosing between designs or libraries, "
    "an ambiguous requirement, or before deleting or overwriting something the user may want to keep. Do not "
    "ask to confirm routine steps. Offer 2 to 4 distinct options, each with a short description of its "
    "consequences; the user can always type a different answer instead.",
    {
        "question": {"type": "string", "description": "A clear, specific question ending with '?'"},
        "options": {"type": "array", "description": f"{MIN_OPTIONS} to {MAX_OPTIONS} choices", "items": {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "Short choice (1-5 words)"},
                           "description": {"type": "string", "description": "What choosing it means"}},
            "required": ["label"]}},
        "multi_select": {"type": "boolean", "description": "Allow choosing several options (default false)"},
    },
    required=["question", "options"],
)
def ask_user(question, options, multi_select=False):
    question = str(question).strip()
    if not question:
        return "Error: the question is empty."
    if not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        return f"Error: give between {MIN_OPTIONS} and {MAX_OPTIONS} options."
    cleaned = []
    for n, option in enumerate(options, 1):
        label = str(option.get("label", "")).strip() if isinstance(option, dict) else ""
        if not label:
            return f"Error: option {n} needs a label."
        cleaned.append({"label": label, "description": str(option.get("description", "")).strip()})
    if len({o["label"].lower() for o in cleaned}) != len(cleaned):
        return "Error: option labels must be different from each other."

    if HANDLER is None:
        return HEADLESS_REPLY
    answers = HANDLER(question, cleaned, bool(multi_select))
    if not answers:
        return SKIPPED_REPLY
    return "User answered: " + "; ".join(f'"{answer}"' for answer in answers)


def terminal_handler(question, options, multi_select, read=input, write=print):
    """Numbered menu in the terminal, plus 'Other' for a typed answer. Enter alone skips."""
    write(f"\n  ❓ {question}")
    for n, option in enumerate(options, 1):
        write(f"     {n}. {option['label']}" + (f": {option['description']}" if option["description"] else ""))
    other = len(options) + 1
    write(f"     {other}. Other (type your own answer)")
    prompt = "  Choose numbers, e.g. 1,3" if multi_select else "  Choose a number"
    while True:
        raw = read(f"{prompt} (Enter to skip): ").strip()
        if not raw:
            return None
        try:
            picks = sorted({int(part) for part in raw.replace(",", " ").split()})
        except ValueError:
            write("  Please enter numbers from the list.")
            continue
        if not picks or any(p < 1 or p > other for p in picks):
            write(f"  Please choose between 1 and {other}.")
            continue
        if len(picks) > 1 and not multi_select:
            write("  Please choose just one.")
            continue
        answers = [options[p - 1]["label"] for p in picks if p != other]
        if other in picks:
            typed = read("  Your answer: ").strip()
            if typed:
                answers.append(typed)
        if answers:
            return answers
