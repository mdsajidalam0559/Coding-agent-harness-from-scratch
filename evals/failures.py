"""Why did trials fail? Reads a results file and each trial's transcript.

    python -m evals.failures evals/results/day12-baseline.jsonl [--all]

For every failed trial (or every trial with --all): steps, tool errors by tool, whether the agent ran
anything after its last edit (did it verify?), how the turn ended, and the tail of the grader's output.
"""
import collections
import json
import os
import sys

EDIT_TOOLS = {"write_file", "str_replace", "apply_edits"}
ERROR_PREFIXES = ("Error", "Tool error", "No files were changed", "Permission denied", "Blocked")


def analyze(trial):
    path = os.path.join(trial["trial_dir"], "transcript.jsonl")
    events = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    tool_results = [e["data"] for e in events if e["type"] == "tool_result" and e["data"].get("agent", "main") == "main"]
    errors = collections.Counter(r["tool"] for r in tool_results if r["result"].startswith(ERROR_PREFIXES))
    last_edit = max((i for i, r in enumerate(tool_results) if r["tool"] in EDIT_TOOLS), default=None)
    ran_after_edit = last_edit is not None and any(r["tool"] == "bash" for r in tool_results[last_edit + 1:])
    edit_errors = [r["result"].splitlines()[0][:140] for r in tool_results
                   if r["tool"] in EDIT_TOOLS and r["result"].startswith(ERROR_PREFIXES)]
    repeated = collections.Counter((r["tool"], r["args"]) for r in tool_results)
    finals = [e["data"]["body"]["choices"][0]["message"].get("content") or "" for e in events
              if e["type"] == "response" and not e["data"]["body"]["choices"][0]["message"].get("tool_calls")]
    return {
        "edits": sum(r["tool"] in EDIT_TOOLS for r in tool_results),
        "tool_errors": dict(errors),
        "edit_errors": edit_errors,
        "verified_after_last_edit": ran_after_edit if last_edit is not None else None,
        "identical_calls_repeated": sum(n - 1 for n in repeated.values() if n > 1),
        "final_answer": finals[-1][:200].replace("\n", " ") if finals else None,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    rows = [json.loads(line) for line in open(args[0])]
    show_all = "--all" in sys.argv
    summary = collections.Counter()
    for trial in rows:
        info = analyze(trial)
        summary["trials"] += 1
        summary["passed"] += trial["passed"]
        summary["no_verification"] += info["verified_after_last_edit"] is False
        summary["with_edit_errors"] += bool(info["edit_errors"])
        summary["hit_max_steps"] += trial["hit_max_steps"]
        summary["no_edits"] += info["edits"] == 0
        if trial["passed"] and not show_all:
            continue
        mark = "PASS" if trial["passed"] else "FAIL"
        print(f"== {mark} {trial['task']} (trial {trial['trial']}): {trial['steps']} steps, "
              f"{info['edits']} edits, verified after last edit: {info['verified_after_last_edit']}, "
              f"repeated identical calls: {info['identical_calls_repeated']}, hit max steps: {trial['hit_max_steps']}")
        if info["tool_errors"]:
            print(f"   tool errors: {info['tool_errors']}")
        for err in info["edit_errors"][:4]:
            print(f"   edit error: {err}")
        print(f"   final answer: {info['final_answer']}")
        if not trial["passed"]:
            tail = [l for l in trial["grade_output"].splitlines() if l.strip()][-4:]
            print("   grader: " + " | ".join(tail)[:400])
        print(f"   transcript: {trial['trial_dir']}/transcript.jsonl")
    print(f"\n{summary['passed']}/{summary['trials']} passed. Of all trials: {summary['no_verification']} never ran "
          f"anything after their last edit, {summary['with_edit_errors']} had failed edits, "
          f"{summary['no_edits']} made no edits, {summary['hit_max_steps']} hit the step limit.")


if __name__ == "__main__":
    main()
