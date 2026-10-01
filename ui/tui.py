"""Interactive terminal agent.

    python -m ui.tui                                   # coding agent, default model
    python -m ui.tui --agent data --model groq:qwen/qwen3.8-27b
    python -m ui.tui --model anthropic:claude-haiku-4-5 --mcp-config servers.json
"""
import argparse
import os

import tools.shell
from agents import make_agent, runs_commands
from agents.coding import Extensions
from ext import skills
from ext.trust import terminal_ask as trust_ask
from models.registry import DEFAULT_MODEL, make_model
from safety.permissions import MODES, terminal_ask
from tools import ask as ask_tool
from ui.console import ConsoleUI


def main():
    parser = argparse.ArgumentParser(description="Interactive agent (coding or data)")
    parser.add_argument("--agent", choices=["coding", "data"], default="coding")
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL_SPEC", DEFAULT_MODEL),
                        help="provider:model (e.g. groq:qwen/qwen3.8-27b), add +text for the text protocol")
    parser.add_argument("--permission-mode", choices=list(MODES), default="auto-read")
    parser.add_argument("--sandbox", choices=["docker", "off"], default="docker")
    parser.add_argument("--network", action="store_true", help="allow network access inside the sandbox")
    parser.add_argument("--mcp-config", help="extra mcpServers JSON file (coding agent)")
    parser.add_argument("--trust-project", action="store_true", help="use the project's .agent/ config without asking")
    args = parser.parse_args()

    workspace = os.getcwd()
    model = make_model(args.model)
    extensions = Extensions(workspace, args.trust_project, trust_ask, args.mcp_config) if args.agent == "coding" else None
    ask_tool.HANDLER = ask_tool.terminal_handler
    ui = ConsoleUI()
    agent = make_agent(args.agent, model, workspace, args.permission_mode, ask=terminal_ask, ui=ui,
                       extensions=extensions)
    sandbox = None
    try:
        if args.sandbox == "docker" and runs_commands(agent):
            from safety.sandbox import DockerSandbox
            user_skills = any(s["source"] == "user" for s in skills.STATE["skills"].values())
            mounts = [(skills.USER_DIR, skills.SANDBOX_USER_DIR)] if user_skills else []
            sandbox = DockerSandbox(workspace, network=args.network, mounts=mounts).start()
            tools.shell.SANDBOX = sandbox
            skills.STATE["user_dir_in_sandbox"] = skills.SANDBOX_USER_DIR if user_skills else None
        print(f"{args.agent} agent · {model.name} ({model.context_window // 1000}k context) · permissions "
              f"{args.permission_mode} · sandbox {'docker' if sandbox else ('not needed (no command tools)' if not runs_commands(agent) else 'OFF (commands run on this machine)')}"
              + (f" · {len(extensions.mcp_tools)} MCP tools, {len(extensions.skills)} skills" if extensions else ""))
        print("Ctrl-C interrupts the agent; Ctrl-C at the prompt exits.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            result = agent.run(user_input)
            if result.status == "interrupted":
                print("\n⏹  Interrupted.")
            elif result.status != "done":
                print(f"\n⚠️  Stopped: {result.status}" + (f" ({result.error})" if result.error else ""))
            if extensions:
                for warning in extensions.hooks.warnings:
                    print(f"  ⚠️  {warning}")
                extensions.hooks.warnings.clear()
            print(f"  [{agent.budget.status()}]")
    finally:
        if sandbox:
            sandbox.stop()
            tools.shell.SANDBOX = None
        if extensions:
            extensions.close()


if __name__ == "__main__":
    main()
