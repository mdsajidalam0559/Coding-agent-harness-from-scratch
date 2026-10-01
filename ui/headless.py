"""Headless mode: run one task from a script, CI job or cron, with no human in the loop.

    python -m ui.headless "Fix the failing tests" --workspace ./repo --model groq:qwen/qwen3.8-27b
    echo "How many orders per region?" | python -m ui.headless --agent data --json
    python -m ui.headless --task-file task.md --sandbox off --unsandboxed-ok --json > result.json

stdout carries only the result (the final answer, or JSON with --json); progress goes to stderr.
Nobody can answer questions or approve actions: ask_user tells the model to decide, and anything the
permission mode would ask about is denied. Exit codes: 0 done, 1 stopped (step limit / API error),
2 usage error, 130 interrupted.
"""
import argparse
import json
import os
import sys
import time

import tools.shell
from agents import make_agent, runs_commands
from agents.coding import Extensions
from longrun.evaluator import snapshot
from models.registry import DEFAULT_MODEL, make_model
from safety.permissions import MODES
from tools import ask as ask_tool
from ui.console import StderrUI

EXIT = {"done": 0, "max_steps": 1, "api_error": 1, "interrupted": 130}


def changed_files(before, after):
    return sorted(rel for rel in before.keys() | after.keys() if before.get(rel) != after.get(rel))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one task with no UI")
    parser.add_argument("task", nargs="?", help="the task (or use --task-file, or stdin)")
    parser.add_argument("--task-file")
    parser.add_argument("--agent", choices=["coding", "data"], default="coding")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL_SPEC", DEFAULT_MODEL))
    parser.add_argument("--permission-mode", choices=list(MODES), default="auto",
                        help="default auto: nobody is there to approve, so the sandbox is the protection")
    parser.add_argument("--sandbox", choices=["docker", "off"], default="docker")
    parser.add_argument("--unsandboxed-ok", action="store_true",
                        help="required with --sandbox off and a permission mode that runs commands unattended")
    parser.add_argument("--trust-project", action="store_true", help="use the project's .agent/ MCP and hook config")
    parser.add_argument("--json", action="store_true", help="print a JSON result instead of the answer")
    parser.add_argument("--quiet", action="store_true", help="no progress on stderr")
    args = parser.parse_args(argv)

    task = open(args.task_file).read() if args.task_file else args.task or (sys.stdin.read() if not sys.stdin.isatty() else "")
    if not task.strip():
        parser.print_usage(sys.stderr)
        print("error: give the task as an argument, with --task-file, or on stdin", file=sys.stderr)
        return 2
    workspace = os.path.realpath(args.workspace)
    os.chdir(workspace)
    ask_tool.HANDLER = None  # no human: the model decides and says so
    extensions = Extensions(workspace, args.trust_project, ask=None,
                            log=lambda m: print(m, file=sys.stderr)) if args.agent == "coding" else None
    ui = None if args.quiet else StderrUI()
    agent = make_agent(args.agent, make_model(args.model), workspace, args.permission_mode, ask=None, ui=ui,
                       extensions=extensions)
    needs_sandbox = runs_commands(agent)  # e.g. the data agent has no command tools: nothing to sandbox
    if needs_sandbox and args.sandbox == "off" and args.permission_mode == "auto" and not args.unsandboxed_ok:
        if extensions:
            extensions.close()
        print("error: --sandbox off with --permission-mode auto runs commands unattended on this machine; "
              "add --unsandboxed-ok if you really mean it", file=sys.stderr)
        return 2
    sandbox = None
    before = snapshot(workspace)
    start = time.monotonic()
    try:
        if args.sandbox == "docker" and needs_sandbox:
            from safety.sandbox import DockerSandbox
            sandbox = DockerSandbox(workspace).start()
            tools.shell.SANDBOX = sandbox
        result = agent.run(task.strip())
    finally:
        if sandbox:
            sandbox.stop()
            tools.shell.SANDBOX = None
        if extensions:
            extensions.close()

    files = changed_files(before, snapshot(workspace))
    if args.json:
        print(json.dumps({"status": result.status, "answer": result.text, "error": result.error,
                          "steps": result.steps, "seconds": round(time.monotonic() - start, 1),
                          "usage": result.usage, "files_changed": files, "model": agent.model.name,
                          "agent": args.agent, "transcript": agent.log_file}, indent=2))
    else:
        print(result.text if result.status == "done" else f"[{result.status}] {result.error or ''}".strip())
    return EXIT.get(result.status, 1)


if __name__ == "__main__":
    sys.exit(main())
