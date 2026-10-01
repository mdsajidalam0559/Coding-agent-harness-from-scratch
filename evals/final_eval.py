"""Final eval: the Day 6 baseline vs the final core agent vs mini-SWE-agent, same model, same tasks.

    python -m evals.final_eval --provider groq --model openai/gpt-oss-20b [--trials 1] [--tasks a,b]

Task-major order (all three agents on task 1, then task 2, ...), so if a quota or outage stops the run
early, every finished task still has a fair three-way comparison. Rerunning the same command resumes:
runs already recorded under the label are skipped. Unsandboxed (mini-SWE-agent runs on
the host anyway), so run it only on these eval tasks.
"""
import argparse
import importlib
import json
import os

from evals import runner
from evals.report import print_report

AGENTS = ["baseline", "final", "mini-swe"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tasks")
    parser.add_argument("--label", default="final")
    args = parser.parse_args()

    baseline = importlib.import_module("days.day_5_agent")  # the agent as it was when the evals were built (Day 6)
    runner.legacy_transport(baseline, args.provider)
    baseline.MODEL = args.model
    run = {
        "baseline": lambda task: runner.run_trial(baseline, task, "off"),
        "final": lambda task: runner.run_core_trial("coding", f"{args.provider}:{args.model}", task, "off"),
        "mini-swe": lambda task: runner.run_mini_swe_trial(f"{args.provider}/{args.model}", task),
    }
    names = {"baseline": "day_5_agent (Day 6 baseline)", "final": "core:coding (Day 14)", "mini-swe": "mini-SWE-agent"}
    tasks = runner.load_tasks(args.tasks.split(",") if args.tasks else None)
    paths = {a: os.path.join(runner.RESULTS_DIR, f"{args.label}-{a}.jsonl") for a in AGENTS}
    os.makedirs(runner.RESULTS_DIR, exist_ok=True)
    done = set()  # (agent, task, trial) already recorded: a rerun continues where the last one stopped
    for agent, path in paths.items():
        if os.path.exists(path):
            done |= {(agent, r["task"], r["trial"]) for r in map(json.loads, open(path))}
    print(f"{len(tasks)} tasks x {args.trials} trials x {len(AGENTS)} agents on {args.provider}:{args.model}"
          + (f" ({len(done)} runs already recorded, skipped)" if done else ""))

    try:
        for trial in range(1, args.trials + 1):
            for task in tasks:
                for agent in AGENTS:
                    if (agent, task["name"], trial) in done:
                        continue
                    r = run[agent](task)
                    if r.get("api_error") and not r["passed"]:
                        print(f"🛑 {agent} on {task['name']}: provider error ({str(r['api_error'])[:160]}). "
                              f"Stopping; results so far are kept (task {task['name']} is incomplete for comparison).")
                        return
                    meta = {"label": f"{args.label}-{agent}", "agent": names[agent], "provider": args.provider,
                            "model": args.model, "sandbox": "off", "suite": "coding"}
                    with open(paths[agent], "a") as f:
                        f.write(json.dumps({**meta, "task": task["name"], "trial": trial, **r}) + "\n")
                    mark = "✅" if r["passed"] else ("💥" if r.get("crash") else "❌")
                    print(f"{mark} {agent:9} {task['name']:24} {r['steps']:>2} steps {r['seconds']:>6.1f}s "
                          f"{r['prompt_tokens'] + r['completion_tokens']:>7} tokens")
    finally:
        existing = [paths[a] for a in AGENTS if os.path.exists(paths[a])]
        if existing:
            print()
            print_report(existing, k=1)


if __name__ == "__main__":
    main()
