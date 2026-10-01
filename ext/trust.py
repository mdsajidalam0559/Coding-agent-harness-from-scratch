"""Approval for project-level config that can run commands (MCP servers, hooks).

A cloned repo could ship .agent/mcp.json or .agent/hooks.json that runs anything on your machine the
moment the agent starts. So project files are only used after you approve them; the approval is
remembered per file and per content hash, and asked again if the file changes.
"""
import hashlib
import json
import os

TRUST_FILE = os.path.expanduser("~/.agent/trusted.json")


def _load():
    try:
        with open(TRUST_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def approve_project_file(path, trust=False, ask=None, purpose="run commands"):
    """True if the file may be used: trusted by flag, approved before (same content), or approved now."""
    path = os.path.realpath(path)
    digest = _digest(path)
    if trust or _load().get(path) == digest:
        return True
    if ask is None:  # headless: never run project-supplied commands without explicit trust
        return False
    with open(path) as f:
        content = f.read()
    if not ask(f"This project's {os.path.basename(path)} wants to {purpose}:\n\n{content[:3000]}\n"):
        return False
    trusted = _load()
    trusted[path] = digest
    os.makedirs(os.path.dirname(TRUST_FILE), exist_ok=True)
    with open(TRUST_FILE, "w") as f:
        json.dump(trusted, f, indent=2)
    return True


def terminal_ask(message):
    print(message)
    return input("Allow? [y/N]: ").strip().lower() in ("y", "yes")
