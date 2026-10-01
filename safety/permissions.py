"""Decide whether a tool call may run: deny rules first, then the permission mode."""
import fnmatch
import os
import re

from tools.files import apply_edits_paths

READ_TOOLS = {"read_file", "list_dir", "search"}
WRITE_TOOLS = {"write_file", "str_replace", "apply_edits"}
EXEC_TOOLS = {"bash"}
# Starting a subagent is harmless by itself: each tool call the subagent makes is checked on its own.
AGENT_TOOLS = {"delegate"}
# Planning tools only change the harness's own records (checklist, feature claims).
# Tools registered at runtime (MCP servers, data tools) declare their category in the registry
from tools.registry import TOOL_CATEGORIES  # noqa: E402
PLANNING_TOOLS = {"todo_write", "create_feature_list", "update_feature", "submit_verdict", "ask_user"}

# mode -> tool categories that run without asking
MODES = {
    "ask": set(),                              # ask before every tool call
    "auto-read": {"read"},                     # default: reads are free, edits and commands need approval
    "auto-edit": {"read", "write"},            # only commands need approval
    "auto": {"read", "write", "exec"},         # never ask; only sensible inside the sandbox
}

SECRET_FILES = [".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*", ".netrc", "credentials*"]

# Best-effort filter for obviously dangerous commands. It is easy to bypass
# (e.g. `python -c "import shutil; shutil.rmtree('/')"`), which is why commands also run in the sandbox.
DENIED_COMMANDS = [
    (r"\bsudo\b|\bsu\s", "running as root"),
    (r"\brm\s+(-\w*\s+)*-\w*[rf]\w*\s+(-\w+\s+)*(/|~|\$HOME|\*)(\s|$)", "recursive delete of /, ~ or *"),
    (r"\bmkfs|\bdd\s+.*\bof=/dev/", "writing to a disk device"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b", "piping a download into a shell"),
    (r"\bgit\s+push\b", "pushing to a remote"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "shutting down the machine"),
    (r"\.env\b|id_rsa|\.ssh/|\.aws/", "accessing secret files"),
]


def _category(tool):
    if tool in TOOL_CATEGORIES:
        return TOOL_CATEGORIES[tool]
    if tool in READ_TOOLS or tool in AGENT_TOOLS or tool in PLANNING_TOOLS:
        return "read"
    if tool in WRITE_TOOLS:
        return "write"
    if tool in EXEC_TOOLS:
        return "exec"
    return "unknown"


def _paths(tool, args):
    if tool == "apply_edits":
        return apply_edits_paths(args.get("edits", ""))
    if tool in READ_TOOLS | WRITE_TOOLS:
        return [args.get("path", ".")]
    return []


def _summary(tool, args):
    if tool == "bash":
        return args.get("command", "")
    if tool == "apply_edits":
        return ", ".join(apply_edits_paths(args.get("edits", ""))) or "(no blocks)"
    if tool == "delegate":
        return args.get("task", "")[:200]
    return args.get("path", ".")


def terminal_ask(tool, summary):
    """Ask the human at the terminal. Returns 'yes', 'no' or 'always'."""
    answer = input(f"\n  ⚠️  Allow {tool}: {summary}\n     [y]es / [n]o / [a]lways for {tool}: ").strip().lower()
    return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(answer, "no")


class Policy:
    def __init__(self, workspace, mode="auto-read", ask=terminal_ask, protected=(), readable_roots=()):
        if mode not in MODES:
            raise ValueError(f"unknown permission mode {mode!r}; choose from {list(MODES)}")
        self.workspace = os.path.realpath(workspace)
        # files the agent may read but not edit with file tools (e.g. harness-owned records)
        self.protected = {os.path.realpath(os.path.join(self.workspace, p)) for p in protected}
        # folders outside the workspace the agent may read (not write), e.g. user-level skills
        self.readable_roots = [os.path.realpath(r) for r in readable_roots]
        self.mode = mode
        self.ask = ask  # None = headless: anything that needs approval is denied
        self.always_allowed = set()

    def _outside_workspace(self, path):
        full = os.path.realpath(os.path.join(self.workspace, path))  # resolves symlinks and ..
        return os.path.commonpath([full, self.workspace]) != self.workspace

    def _in_readable_root(self, path):
        full = os.path.realpath(os.path.join(self.workspace, path))
        return any(os.path.commonpath([full, root]) == root for root in self.readable_roots)

    def deny_reason(self, tool, args):
        category = _category(tool)
        if category == "unknown":
            return f"unknown tool {tool}"
        for path in _paths(tool, args):
            if self._outside_workspace(path) and not (category == "read" and self._in_readable_root(path)):
                return f"{path} is outside the workspace ({self.workspace})"
            name = os.path.basename(os.path.normpath(path))
            if any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_FILES):
                return f"{path} may contain secrets"
            if category == "write" and ".git" in os.path.normpath(path).split(os.sep):
                return "editing files inside .git is not allowed"
            if category == "write" and os.path.realpath(os.path.join(self.workspace, path)) in self.protected:
                return f"{path} is maintained by the harness; do not edit it yourself"
        if tool == "bash":
            command = args.get("command", "")
            for pattern, why in DENIED_COMMANDS:
                if re.search(pattern, command):
                    return f"command matches deny rule: {why}"
        return None

    def check(self, tool, args):
        """Return (allowed, reason)."""
        reason = self.deny_reason(tool, args)
        if reason:
            return False, f"denied by policy: {reason}"
        if _category(tool) in MODES[self.mode] or tool in self.always_allowed:
            return True, f"auto-approved ({self.mode})"
        if self.ask is None:
            return False, f"needs approval in mode {self.mode!r}, but no human is available (headless run)"
        answer = self.ask(tool, _summary(tool, args))
        if answer == "always":
            self.always_allowed.add(tool)
        if answer in ("yes", "always"):
            return True, "approved by user"
        return False, "denied by user"
