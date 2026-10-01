"""Connect configured MCP servers and expose their tools to the agent as normal tools.

Config (the same shape Claude Desktop and others use), in ~/.agent/mcp.json (yours, trusted) and/or
<project>/.agent/mcp.json (comes with the repo, so it needs your approval before it can start anything):

    {"mcpServers": {
        "git":    {"command": "uvx", "args": ["mcp-server-git", "--repository", "${workspace}"]},
        "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${env:MY_TOKEN}"}}
    }}
"""
import json
import os
import re

from ext.mcp_client import MCPError, format_result, make_client
from ext.trust import approve_project_file
from safety.permissions import TOOL_CATEGORIES
from tools.registry import REGISTRY

USER_CONFIG = os.path.expanduser("~/.agent/mcp.json")
PROJECT_CONFIG = os.path.join(".agent", "mcp.json")


def tool_name(server, tool):
    """mcp__<server>__<tool>, limited to what model APIs accept: [a-zA-Z0-9_-], at most 64 characters."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp__{server}__{tool}")[:64]


def clean_schema(schema):
    """inputSchema as a function-parameters schema that every provider accepts."""
    schema = dict(schema or {})
    schema.pop("$schema", None)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


def register(client):
    """Add the server's tools to the registry; returns their names."""
    names = []
    for tool in client.tools:
        name = tool_name(client.name, tool["name"])
        while name in REGISTRY and REGISTRY[name].get("mcp", {}).get("tool") != tool["name"]:
            name = name[:60] + f"_{len(names)}"  # rare: two tools that sanitize to the same name
        description = (tool.get("description") or tool.get("title") or tool["name"]).strip()
        schema = clean_schema(tool.get("inputSchema"))

        def call(_client=client, _tool=tool["name"], **arguments):
            try:
                return format_result(_client.call_tool(_tool, arguments))
            except MCPError as e:
                return f"Error from MCP server {_client.name!r}: {e.message}"

        REGISTRY[name] = {
            "fn": call,
            "schema": {"type": "function", "function": {
                "name": name, "description": f"{description[:1000]} (tool '{tool['name']}' from MCP server '{client.name}')",
                "parameters": schema}},
            "strict_args": schema.get("additionalProperties") is False,
            "mcp": {"server": client.name, "tool": tool["name"]},
        }
        # third-party tools can do anything, unless the server marks them read-only
        annotations = tool.get("annotations") or {}
        TOOL_CATEGORIES[name] = "read" if annotations.get("readOnlyHint") else "exec"
        names.append(name)
    return names


def unregister(names):
    for name in names:
        REGISTRY.pop(name, None)
        TOOL_CATEGORIES.pop(name, None)


def load_config(workspace, trust_project=False, ask=None, extra_file=None):
    """Server configs by name. Project config needs approval (trust_project=True, or ask() says yes)."""
    servers = {}
    sources = [USER_CONFIG] + ([extra_file] if extra_file else [])
    project_file = os.path.join(workspace, PROJECT_CONFIG)
    if os.path.exists(project_file) and approve_project_file(project_file, trust_project, ask,
                                                             "start these MCP servers on your machine"):
        sources.append(project_file)
    for path in sources:
        if path and os.path.exists(path):
            with open(path) as f:
                servers.update(json.load(f).get("mcpServers") or {})
    return servers


def connect_all(workspace, configs, log=print):
    """Start every configured server. A server that fails is reported and skipped, never fatal."""
    clients, names = [], []
    for server, config in configs.items():
        try:
            client = make_client(server, config, workspace).connect()
        except (MCPError, ValueError, OSError) as e:
            log(f"⚠️  MCP server {server!r} not available: {e}")
            continue
        clients.append(client)
        names += register(client)
        log(f"🔌 MCP server {server!r} ({client.server_info.get('name', '?')} {client.server_info.get('version', '')}): "
            f"{len(client.tools)} tools")
    return clients, names


def server_instructions(clients):
    """The optional usage notes servers send at initialization, for the system prompt."""
    notes = [f"## {c.name}\n{c.instructions.strip()[:2000]}" for c in clients if c.instructions]
    return ("\n\n# Notes from connected MCP servers\n" + "\n\n".join(notes)) if notes else ""
