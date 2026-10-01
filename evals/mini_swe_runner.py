"""Run one task with mini-SWE-agent, unmodified (its own mini.yaml), for the comparison in the final eval.

Runs inside its own environment (it depends on litellm), launched by evals/runner.py:
    uv run --with mini-swe-agent python evals/mini_swe_runner.py --workdir W --task-file T --model groq/qwen/qwen3.8-27b --out R
Writes a JSON summary (exit status, steps, tokens) to --out; the trajectory goes next to it.
"""
import argparse
import json
import os

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--model", required=True, help="litellm model name, e.g. groq/qwen/qwen3.8-27b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--step-limit", type=int, default=25)
    args = parser.parse_args()

    config = yaml.safe_load(open(builtin_config_dir / "mini.yaml"))
    trajectory = os.path.splitext(args.out)[0] + ".traj.json"
    agent_config = {**config["agent"], "step_limit": args.step_limit, "cost_limit": 0, "output_path": trajectory}
    agent = DefaultAgent(get_model(args.model, config.get("model", {})),
                         LocalEnvironment(**{**config.get("environment", {}), "cwd": args.workdir}), **agent_config)
    try:
        result = agent.run(open(args.task_file).read())
        status = result.get("exit_status")
    except Exception as e:  # report, don't crash the eval
        status = f"crash: {e!r}"[:300]

    prompt_tokens = completion_tokens = calls = 0
    for message in agent.messages:
        usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
        if usage:
            calls += 1
            prompt_tokens += usage.get("prompt_tokens") or 0
            completion_tokens += usage.get("completion_tokens") or 0
    with open(args.out, "w") as f:
        json.dump({"exit_status": status, "steps": calls or getattr(agent, "n_calls", 0),
                   "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                   "cost": getattr(agent, "cost", 0.0), "trajectory": trajectory}, f, indent=2)


if __name__ == "__main__":
    main()
