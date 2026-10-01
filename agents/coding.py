"""The coding agent: the core + file/shell/search tools + a coding system prompt + extensions."""
from agents import build_policy
from context.prompt import build_system_prompt
from core.agent import AgentConfig
from ext import hooks as hooks_mod, mcp_tools, skills

ASKING = """

Asking the user:
- When you need a decision from the user, call ask_user with 2-4 concrete options instead of ending your turn with a question. Their answer comes back and you continue.
- Ask only when the answer changes what you build and the code cannot tell you. Never ask for permission to do what the user already requested."""

SUBAGENT_SUFFIX = """

You are running as a SUBAGENT. Another agent delegated this one focused task to you and will see ONLY your final reply, not your tool calls. When you are done, reply with a complete, self-contained report: the answer or result, the evidence (file paths, line numbers, relevant output), and anything you changed. Do not ask questions; do the best you can with what you have."""


class Extensions:
    """MCP servers, hooks and skills for one workspace (connected once per process)."""

    def __init__(self, workspace, trust_project=False, ask=None, mcp_config=None, syntax_check=True, log=print):
        self.hooks = hooks_mod.load_hooks(workspace, trust_project, ask, syntax_check)
        found, problems = skills.discover(workspace)
        for problem in problems:
            log(f"⚠️  skill ignored: {problem}")
        self.skills = list(found)
        self.clients, self.mcp_tools = mcp_tools.connect_all(workspace, mcp_tools.load_config(
            workspace, trust_project, ask, mcp_config), log)

    def close(self):
        for client in self.clients:
            client.close()
        mcp_tools.unregister(self.mcp_tools)


def coding_config(edit_format="str_replace", extensions=None, extra_tools=()):
    tools = ["read_file", "list_dir", "search", "write_file", edit_format, "bash", "todo_write", "ask_user", "delegate"]
    if extensions and extensions.skills:
        tools.append("load_skill")
    if extensions:
        tools += extensions.mcp_tools
    tools += list(extra_tools)

    def prompt(agent):
        return (build_system_prompt(agent.workspace, edit_format) + ASKING + skills.prompt_section(skills.STATE["skills"])
                + (mcp_tools.server_instructions(extensions.clients) if extensions else ""))

    return AgentConfig(name="coding", tools=tools, system_prompt=prompt, subagent_suffix=SUBAGENT_SUFFIX)


__all__ = ["coding_config", "Extensions", "build_policy"]
