import os
import subprocess

from .registry import tool
from .shell import strip_ansi

MAX_MATCHES = 100
MAX_LINE_CHARS = 300


@tool(
    "Search file contents with ripgrep. Returns matching lines as `path:line:text`. "
    "The pattern is a regular expression (escape special characters like `(` or `.` for a literal match). "
    "Skips files ignored by .gitignore, hidden files and binary files. "
    f"Returns at most {MAX_MATCHES} matches; narrow the search with `path` or `glob` if there are more. "
    "Use this instead of reading files one by one to find where something is defined or used.",
    {
        "pattern": {"type": "string", "description": "Regular expression to search for"},
        "path": {"type": "string", "description": "File or directory to search (default '.')"},
        "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py' or 'src/**/*.ts'"},
        "ignore_case": {"type": "boolean", "description": "Case-insensitive search (default false)"},
    },
    required=["pattern"],
)
def search(pattern, path=".", glob=None, ignore_case=False):
    path = path or "."
    if not os.path.exists(path):
        return f"Error: {path} does not exist."
    cmd = ["rg", "--line-number", "--no-heading", "--color=never", "--sort=path",
           f"--max-columns={MAX_LINE_CHARS}", "--max-columns-preview"]
    if ignore_case:
        cmd.append("--ignore-case")
    if glob:
        cmd += ["--glob", glob]
    cmd += ["-e", pattern, "--", path]

    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                            stdin=subprocess.DEVNULL, timeout=30)
    if result.returncode == 1:
        return f"No matches for {pattern!r} in {path}" + (f" (glob {glob})" if glob else "") + "."
    if result.returncode != 0:
        return f"Error: {strip_ansi(result.stderr).strip()}"

    lines = result.stdout.splitlines()
    out = "\n".join(lines[:MAX_MATCHES])
    if len(lines) > MAX_MATCHES:
        out += f"\n... ({len(lines) - MAX_MATCHES} more matches; narrow the search with path or glob)"
    return out
