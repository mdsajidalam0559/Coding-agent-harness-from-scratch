"""Hooks: callbacks that run before and after every tool call.

Before a tool runs, a hook can block it (with a reason the model sees) or change its arguments.
After it runs, a hook can add feedback to the result (e.g. a linter's complaints) or replace it.

Two kinds:
- Python callbacks: fn(tool, args) / fn(tool, args, result), returning None or a dict (see Hooks).
- Shell commands from config (~/.agent/hooks.json, or the project's .agent/hooks.json once approved):

    {"PreToolUse":  [{"matcher": "bash", "command": "python3 scripts/check_command.py"}],
     "PostToolUse": [{"matcher": "write_file|str_replace|apply_edits", "command": "ruff check --quiet ."}]}

  The command gets {"event", "tool", "args", "result" (after only), "workspace"} as JSON on stdin.
  Exit 0: fine (after a tool, anything printed is added to the result). Exit 2: before a tool, block
  it with stderr as the reason; after, add stderr to the result as a problem report. Other exit codes
  are reported to the user but do not affect the agent.
"""
import ast
import json
import os
import re
import subprocess

from ext.trust import approve_project_file
from tools.files import apply_edits_paths

USER_CONFIG = os.path.expanduser("~/.agent/hooks.json")
PROJECT_CONFIG = os.path.join(".agent", "hooks.json")
HOOK_TIMEOUT = 60
EDIT_TOOLS = {"write_file", "str_replace", "apply_edits"}


class Hooks:
    def __init__(self):
        self.pre, self.post = [], []
        self.warnings = []  # hook failures for the user (not the model)

    def before_tool(self, fn, matcher=None):
        """fn(tool, args) -> None | {"deny": reason} | {"args": new_args}"""
        self.pre.append((re.compile(matcher) if matcher else None, fn))
        return fn

    def after_tool(self, fn, matcher=None):
        """fn(tool, args, result) -> None | {"append": text} | {"result": new_result}"""
        self.post.append((re.compile(matcher) if matcher else None, fn))
        return fn

    @staticmethod
    def _matches(matcher, tool):
        return matcher is None or matcher.fullmatch(tool) is not None

    def run_before(self, tool, args):
        """Returns (allowed, args, reason)."""
        for matcher, fn in self.pre:
            if not self._matches(matcher, tool):
                continue
            outcome = fn(tool, args) or {}
            if "deny" in outcome:
                return False, args, outcome["deny"]
            if isinstance(outcome.get("args"), dict):
                args = outcome["args"]
        return True, args, None

    def run_after(self, tool, args, result):
        for matcher, fn in self.post:
            if not self._matches(matcher, tool):
                continue
            outcome = fn(tool, args, result) or {}
            if "result" in outcome:
                result = str(outcome["result"])
            if outcome.get("append"):
                result = f"{result}\n\n{outcome['append']}"
        return result


def edited_paths(tool, args):
    if tool == "apply_edits":
        return apply_edits_paths(args.get("edits", ""))
    return [args["path"]] if tool in EDIT_TOOLS and "path" in args else []


def python_syntax_check(tool, args, result):
    """Built-in after-hook: tell the model at once when an edit leaves a .py file unparseable."""
    if result.startswith(("Error", "No files were changed", "Permission denied")):
        return None
    problems = []
    for path in edited_paths(tool, args):
        if not path.endswith(".py") or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                ast.parse(f.read(), filename=path)
        except SyntaxError as e:
            problems.append(f"{path} line {e.lineno}: {e.msg}" + (f"\n    {e.text.rstrip()}" if e.text else ""))
    if problems:
        return {"append": "Warning from the syntax check hook: the file no longer parses:\n" + "\n".join(problems)}
    return None


def shell_hook(event, command, workspace, warnings):
    """A config hook as a callback. event: 'PreToolUse' or 'PostToolUse'."""
    def run(tool, args, result=None):
        payload = {"event": event, "tool": tool, "args": args, "workspace": workspace}
        if result is not None:
            payload["result"] = result
        try:
            proc = subprocess.run(command, shell=True, input=json.dumps(payload), capture_output=True,
                                  text=True, cwd=workspace, timeout=HOOK_TIMEOUT)
        except subprocess.TimeoutExpired:
            warnings.append(f"{event} hook timed out after {HOOK_TIMEOUT}s: {command}")
            return None
        if proc.returncode == 2:
            message = proc.stderr.strip() or proc.stdout.strip() or "(no details)"
            if event == "PreToolUse":
                return {"deny": f"blocked by hook `{command}`: {message}"}
            return {"append": f"Problem reported by hook `{command}`:\n{message[:4000]}"}
        if proc.returncode != 0:
            warnings.append(f"{event} hook `{command}` failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}")
            return None
        if event == "PostToolUse" and proc.stdout.strip():
            return {"append": f"Output of hook `{command}`:\n{proc.stdout.strip()[:4000]}"}
        return None
    return run


def load_hooks(workspace, trust_project=False, ask=None, syntax_check=True):
    hooks = Hooks()
    if syntax_check:
        hooks.after_tool(python_syntax_check, matcher="|".join(sorted(EDIT_TOOLS)))
    sources = [USER_CONFIG]
    project_file = os.path.join(workspace, PROJECT_CONFIG)
    if os.path.exists(project_file) and approve_project_file(project_file, trust_project, ask,
                                                             "run these hook commands on your machine"):
        sources.append(project_file)
    for path in sources:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            config = json.load(f)
        for event, register in (("PreToolUse", hooks.before_tool), ("PostToolUse", hooks.after_tool)):
            for entry in config.get(event) or []:
                register(shell_hook(event, entry["command"], workspace, hooks.warnings), entry.get("matcher"))
    return hooks
