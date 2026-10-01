"""Run every eval task several times against the agent and record the results.

    python -m evals.runner --model anthropic/claude-haiku-4.5 --trials 3 --cap 2.00 --label baseline

Each trial gets a fresh copy of the task's repo and a fresh sandbox container. After the agent
finishes, the hidden grading test is copied in and run in another fresh container, so the agent
can neither see nor tamper with it. Results are appended to evals/results/<label>.jsonl.
"""
import argparse
import contextlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import tools.shell
from context.budget import cached_tokens, context_overflow
from safety.permissions import Policy
from tools import files

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(EVALS_DIR, "tasks")
SUITES = {"coding": TASKS_DIR, "data": os.path.join(EVALS_DIR, "tasks_data")}
RESULTS_DIR = os.path.join(EVALS_DIR, "results")
GRADE_CMD = "python -m unittest -q grade_test"
GRADE_TIMEOUT = 120


def load_tasks(names=None, suite="coding"):
    tasks, folder = [], SUITES[suite]
    for name in sorted(os.listdir(folder)):
        if names and name not in names:
            continue
        with open(os.path.join(folder, name, "task.json")) as f:
            tasks.append({"name": name, "dir": os.path.join(folder, name), **json.load(f)})
    if names and len(tasks) != len(names):
        raise SystemExit(f"unknown task(s): {set(names) - {t['name'] for t in tasks}}")
    return tasks


def prepare(task, workdir, with_solution=False):
    shutil.copytree(os.path.join(task["dir"], "repo"), workdir)
    if with_solution:
        shutil.copytree(os.path.join(task["dir"], "solution"), workdir, dirs_exist_ok=True)


def grade(task, workdir, sandbox_mode):
    """Copy in the hidden grading test and run it. Returns (passed, output)."""
    shutil.copytree(os.path.join(task["dir"], "grade"), workdir, dirs_exist_ok=True)
    if sandbox_mode == "docker":
        from safety.sandbox import DockerSandbox
        with DockerSandbox(workdir) as sb:
            code, out, err, timed_out = sb.exec(GRADE_CMD, GRADE_TIMEOUT, {})
        return code == 0 and not timed_out, (out + err)[-3000:]
    try:
        result = subprocess.run(GRADE_CMD.replace("python", sys.executable, 1).split(), cwd=workdir,
                                capture_output=True, text=True, timeout=GRADE_TIMEOUT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False, "grading timed out"
    return result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def is_context_overflow(error_event):
    return context_overflow(error_event)[0]


def summarize_transcript(path):
    stats = {"steps": 0, "tool_calls": 0, "denied": 0, "prompt_tokens": 0, "cached_tokens": 0,
             "completion_tokens": 0, "cost": 0.0, "api_error": None, "hit_max_steps": False,
             "clipped_results": 0, "compactions": 0, "subagents": 0, "context_overflow": False}
    if not os.path.exists(path):
        return stats
    for line in open(path):
        e = json.loads(line)
        kind, data = e["type"], e["data"]
        if kind == "response":
            usage = data["body"].get("usage") or {}
            stats["steps"] += 1
            stats["prompt_tokens"] += usage.get("prompt_tokens") or 0
            stats["cached_tokens"] += cached_tokens(usage)
            stats["completion_tokens"] += usage.get("completion_tokens") or 0
            stats["cost"] += usage.get("cost") or 0
        elif kind == "tool_result":
            stats["tool_calls"] += 1
            stats["clipped_results"] += bool(data.get("clipped"))
        elif kind == "compaction" and data.get("compacted"):
            stats["compactions"] += 1
        elif kind == "subagent_start":
            stats["subagents"] += 1
        elif kind == "permission" and not data["allowed"]:
            stats["denied"] += 1
        elif kind == "api_error" and (data.get("final") or not data["retryable"]):
            if is_context_overflow(data):  # the agent let its context outgrow the model: its failure, not infra
                stats["context_overflow"] = True
            else:
                stats["api_error"] = data["status"] or "network"
        elif kind == "max_steps_reached":
            stats["hit_max_steps"] = True
    return stats


REVIEW_ROUNDS = 2


def review_loop(model, task, workdir, last_reply, send_feedback, log_file, sandbox=None):
    """Day 11: an evaluator agent checks the work; its findings go back to the agent (up to REVIEW_ROUNDS).

    last_reply() returns the agent's latest answer; send_feedback(text) gives the agent another turn.
    """
    from longrun import evaluator
    verdicts = []
    for _ in range(REVIEW_ROUNDS):
        review = evaluator.run_evaluator(model, workdir, task["prompt"], last_reply(),
                                         evaluator.dir_diff(os.path.join(task["dir"], "repo"), workdir),
                                         log_file=log_file, sandbox=sandbox)
        verdicts.append(review["verdict"])
        if review["verdict"] != "fail":
            break
        send_feedback(evaluator.FEEDBACK_PROMPT.format(issues=evaluator.format_issues(review), claim_again=""))
    return verdicts


def last_assistant_reply(messages):
    replies = [m.get("content") or "" for m in messages if m.get("role") == "assistant" and not m.get("tool_calls")]
    return replies[-1] if replies else ""


def run_trial(agent, task, sandbox_mode, use_evaluator=False):
    trial_dir = tempfile.mkdtemp(prefix=f"eval-{task['name']}-")
    workdir = os.path.join(trial_dir, "work")
    prepare(task, workdir)

    agent.messages.clear()
    agent.turn_count = 0
    files._read_state.clear()
    agent.policy = Policy(workdir, mode="auto", ask=None)  # unattended: deny rules + sandbox are the guard
    agent.log_file = os.path.join(trial_dir, "transcript.jsonl")  # outside the workspace, invisible to the agent

    sandbox = None
    if sandbox_mode == "docker":
        from safety.sandbox import DockerSandbox
        sandbox = DockerSandbox(workdir).start()
        tools.shell.SANDBOX = sandbox

    home, start = os.getcwd(), time.monotonic()
    os.chdir(workdir)
    crash = None
    verdicts = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent.agentic_loop(task["prompt"])
            if use_evaluator:
                from models.registry import make_model
                verdicts = review_loop(make_model(f"{getattr(agent, 'PROVIDER', 'openrouter')}:{agent.MODEL}"), task,
                                       workdir, lambda: last_assistant_reply(agent.messages), agent.agentic_loop,
                                       agent.log_file, tools.shell.SANDBOX)
    except Exception as e:  # a harness bug must not kill the whole run
        crash = repr(e)
    finally:
        os.chdir(home)
        tools.shell.SANDBOX = None
        if sandbox:
            sandbox.stop()
    seconds = time.monotonic() - start

    passed, grade_output = grade(task, workdir, sandbox_mode)
    return {"passed": passed, "seconds": round(seconds, 1), "crash": crash, "grade_output": grade_output,
            "evaluator_verdicts": verdicts,
            "trial_dir": trial_dir, **summarize_transcript(agent.log_file)}


def run_core_trial(kind, model_spec, task, sandbox_mode, use_evaluator=False):
    """A trial for an agent built on core/ (Day 14): a fresh Agent, so nothing carries over between trials."""
    from agents import make_agent
    from models.registry import make_model
    from tools import todo
    trial_dir = tempfile.mkdtemp(prefix=f"eval-{task['name']}-")
    workdir = os.path.join(trial_dir, "work")
    prepare(task, workdir)
    files._read_state.clear()
    todo.reset()
    log_file = os.path.join(trial_dir, "transcript.jsonl")
    agent = make_agent(kind, make_model(model_spec) if isinstance(model_spec, str) else model_spec, workdir,
                       mode="auto", ask=None, log_file=log_file)
    sandbox = None
    if sandbox_mode == "docker":
        from safety.sandbox import DockerSandbox
        sandbox = agent.sandbox = DockerSandbox(workdir).start()
    start, crash, status, verdicts = time.monotonic(), None, None, []
    try:  # no os.chdir: the agent's tools work in its workspace wherever the process is
        with contextlib.redirect_stdout(io.StringIO()):
            status = agent.run(task["prompt"]).status
            if use_evaluator:
                verdicts = review_loop(agent.model, task, workdir, lambda: last_assistant_reply(agent.messages),
                                       lambda text: agent.run(text), log_file, sandbox)
    except Exception as e:
        crash = repr(e)
    finally:
        if sandbox:
            sandbox.stop()
    passed, grade_output = grade(task, workdir, sandbox_mode)
    return {"passed": passed, "seconds": round(time.monotonic() - start, 1), "crash": crash, "status": status,
            "grade_output": grade_output, "evaluator_verdicts": verdicts, "trial_dir": trial_dir,
            **summarize_transcript(log_file)}


def run_mini_swe_trial(litellm_model, task, step_limit=25):
    """A trial with mini-SWE-agent (unmodified, in its own uv environment). Runs on this machine."""
    trial_dir = tempfile.mkdtemp(prefix=f"eval-{task['name']}-")
    workdir = os.path.join(trial_dir, "work")
    prepare(task, workdir)
    task_file, out = os.path.join(trial_dir, "task.md"), os.path.join(trial_dir, "result.json")
    with open(task_file, "w") as f:
        f.write(task["prompt"])
    from dotenv import dotenv_values
    keys = dotenv_values(os.path.join(os.path.dirname(EVALS_DIR), ".env"))
    env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors", "MSWEA_CONFIGURED": "true"}
    # litellm's names for our keys (set explicitly, so they win over mini-swe-agent's own global config)
    for ours, litellm_name in (("GROQAPI_KEY", "GROQ_API_KEY"), ("GEMINIAPI_KEY", "GEMINI_API_KEY"),
                               ("OPENROUTER_KEY", "OPENROUTER_API_KEY")):
        if keys.get(ours):
            env[litellm_name] = keys[ours].strip()
    start, crash = time.monotonic(), None
    try:
        proc = subprocess.run(["uv", "run", "--quiet", "--with", "mini-swe-agent", "python",
                               os.path.join(EVALS_DIR, "mini_swe_runner.py"), "--workdir", workdir, "--task-file",
                               task_file, "--model", litellm_model, "--out", out, "--step-limit", str(step_limit)],
                              capture_output=True, text=True, env=env, timeout=3600, stdin=subprocess.DEVNULL)
        if proc.returncode != 0 or not os.path.exists(out):
            crash = f"mini-swe exited {proc.returncode}: {proc.stderr.strip()[-400:]}"
    except subprocess.TimeoutExpired:
        crash = "mini-swe timed out"
    summary = json.load(open(out)) if os.path.exists(out) else {}
    passed, grade_output = grade(task, workdir, "off")
    exit_status = str(summary.get("exit_status"))
    infra = ("ratelimit", "rate_limit", "apierror", "authentication", "api key", "api_key", "apiconnection",
             "serviceunavailable", "timeout")
    api_error = exit_status if any(s in exit_status.lower() for s in infra) else None
    return {"passed": passed, "seconds": round(time.monotonic() - start, 1), "crash": crash, "status": exit_status,
            "grade_output": grade_output, "evaluator_verdicts": [], "trial_dir": trial_dir,
            "steps": summary.get("steps", 0), "tool_calls": summary.get("steps", 0), "denied": 0,
            "prompt_tokens": summary.get("prompt_tokens", 0), "cached_tokens": 0,
            "completion_tokens": summary.get("completion_tokens", 0), "cost": summary.get("cost") or 0.0,
            "api_error": api_error, "hit_max_steps": "limit" in exit_status.lower(),
            "clipped_results": 0, "compactions": 0, "subagents": 0,
            "context_overflow": "contextwindow" in exit_status.lower().replace("_", "")}


def legacy_transport(agent, provider):
    """Day 2-6 agents only know OpenRouter. Swap just their HTTP call for the Day 7 provider-aware one, so the
    old loop (prompt, tools, logic) runs unchanged on another provider."""
    from days import day_7_agent
    agent.call_model = lambda payload, max_retries=4: day_7_agent.call_model(
        payload, log=agent.log_to_jsonl, provider=provider, max_retries=max_retries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="day_5_agent",
                        help="a day snapshot (day_5_agent ... day_13_agent, from days/), core:coding / core:data, or mini-swe")
    parser.add_argument("--suite", choices=list(SUITES), default="coding", help="which task set")
    parser.add_argument("--provider", help="provider name, for agents that support it (day_7+)")
    parser.add_argument("--model", help="model id (default: the agent's default for the provider)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cap", type=float, default=2.00, help="stop once total spend (USD) exceeds this")
    parser.add_argument("--tasks", help="comma-separated task names (default: all)")
    parser.add_argument("--sandbox", choices=["docker", "off"], default="docker")
    parser.add_argument("--label", help="name of the results file (default: timestamp)")
    parser.add_argument("--evaluator", action="store_true", help="Day 11: review each attempt with an evaluator agent")
    args = parser.parse_args()

    if args.sandbox == "off":
        print("⚠️  --sandbox off: the agent's commands run on this machine, unattended.")
    else:
        from safety.sandbox import require_docker
        require_docker()  # fail before any trial runs or a results file is created
    tasks = load_tasks(args.tasks.split(",") if args.tasks else None, args.suite)

    protocol = None
    if args.agent.startswith("core:"):
        kind = args.agent.split(":", 1)[1]
        spec = args.model if args.model and ":" in args.model and not args.provider else \
            f"{args.provider or 'groq'}:{args.model or 'openai/gpt-oss-120b'}"
        run = lambda task: run_core_trial(kind, spec, task, args.sandbox, use_evaluator=args.evaluator)
        model_name, provider = spec, spec.split(":", 1)[0]
    elif args.agent == "mini-swe":
        if args.sandbox != "off":
            raise SystemExit("mini-swe runs its commands on this machine: use --sandbox off")
        model_name = f"{args.provider or 'groq'}/{args.model or 'openai/gpt-oss-120b'}"
        run = lambda task: run_mini_swe_trial(model_name, task)
        provider = args.provider or "groq"
    else:
        module = f"days.{args.agent}" if args.agent.startswith("day_") else args.agent  # short names still work
        agent = importlib.import_module(module)
        if args.provider:
            if hasattr(agent, "PROVIDER"):
                agent.PROVIDER = args.provider
            else:
                legacy_transport(agent, args.provider)
            if not args.model:
                agent.MODEL = importlib.import_module("days.day_7_agent").PROVIDERS[args.provider]["default_model"]
        if args.model:
            agent.MODEL = args.model
        run = lambda task: run_trial(agent, task, args.sandbox, use_evaluator=args.evaluator)
        model_name, provider = agent.MODEL, getattr(agent, "PROVIDER", args.provider or "openrouter")
        protocol = getattr(agent, "PROTOCOL", None)  # day 7: tools or text (AGENT_PROTOCOL=text)

    label = args.label or datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{label}.jsonl")
    meta = {"evaluator": args.evaluator, "agent": args.agent, "provider": provider, "model": model_name,
            "suite": args.suite, "sandbox": args.sandbox, "label": label,
            **({"protocol": protocol} if protocol else {})}
    print(f"Running {len(tasks)} tasks x {args.trials} trials -> {out_path}")

    spent = 0.0
    with open(out_path, "a") as out:
        for trial in range(1, args.trials + 1):  # trial-major order: an early stop still covers every task
            for task in tasks:
                if spent > args.cap:
                    print(f"💸 Spending cap ${args.cap:.2f} reached (${spent:.2f}); stopping early.")
                    return report(out_path)
                r = run(task)
                spent += r["cost"]
                if r["api_error"] and not r["passed"]:
                    print(f"🛑 API error {r['api_error']} on {task['name']} ({r['trial_dir']}/transcript.jsonl). "
                          "Infrastructure failure, not a model failure: stopping without recording it.")
                    return report(out_path)
                out.write(json.dumps({**meta, "task": task["name"], "trial": trial, **r}) + "\n")
                out.flush()
                mark = "✅" if r["passed"] else ("💥" if r["crash"] else "❌")
                print(f"{mark} trial {trial} {task['name']:24} {r['steps']:>2} steps {r['seconds']:>6.1f}s "
                      f"${r['cost']:.4f}" + (f"  crash: {r['crash']}" if r["crash"] else ""))
    report(out_path)


def report(path):
    from evals.report import print_report
    print()
    print_report([path])


if __name__ == "__main__":
    main()
