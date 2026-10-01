"""Agents built on the core. An agent = a tool set + a system prompt (+ optional extensions)."""
import os

from safety.permissions import Policy


def runs_commands(agent):
    """Does this agent have any tool that executes commands? Only then does a sandbox matter."""
    from safety.permissions import _category
    return any(_category(name) == "exec" for name in agent.tool_names())


def build_policy(workspace, mode, ask=None, readable_roots=()):
    """ask=None means no human is present: anything that needs approval is denied."""
    return Policy(workspace, mode, ask=ask, readable_roots=readable_roots)


def make_agent(kind, model, workspace, mode="auto-read", ask=None, ui=None, log_file=None,
               edit_format="str_replace", extensions=None):
    """An Agent of the given kind ('coding' or 'data') on any model adapter."""
    from core.agent import Agent
    if kind == "coding":
        from agents.coding import coding_config
        config = coding_config(edit_format, extensions)
    elif kind == "data":
        from agents.data import data_config
        config = data_config()
    else:
        raise ValueError(f"unknown agent {kind!r} (choose coding or data)")
    policy = build_policy(workspace, mode, ask)
    return Agent(config, model, workspace, policy, hooks=extensions.hooks if extensions else None,
                 log_file=log_file or os.path.join(workspace, ".agent", f"{kind}-transcript.jsonl"), ui=ui)
