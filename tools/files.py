import hashlib
import re
import os

from .registry import tool

MAX_READ_LINES = 2000
MAX_READ_CHARS = 40_000  # ~11k tokens per read, whatever the line count
MAX_LINE_CHARS = 2_000
MAX_LIST_ENTRIES = 200
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".agent"}
CONTEXT_LINES = 3

# path -> sha256 of the file content when the agent last read or wrote it
_read_state = {}


def _key(path):
    return os.path.realpath(path)


def _load(path):
    # newline="" keeps \r\n intact so we match exactly what is on disk
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def _save(path, content):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def _digest(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _remember(path, content):
    _read_state[_key(path)] = _digest(content)


def _check_fresh(path, content):
    """Return an error string if the agent may not modify this file, else None."""
    recorded = _read_state.get(_key(path))
    if recorded is None:
        return f"Error: you have not read {path} yet. Call read_file on it before modifying it."
    if recorded != _digest(content):
        return (f"Error: {path} changed on disk since you last read it (for example by a bash command). "
                f"Call read_file again and base your edit on the current content.")
    return None


def _numbered(lines, start):
    """Format lines like `cat -n`; start is the 1-based number of the first line."""
    return "\n".join(f"{n:>6}\t{line.rstrip(chr(13))}" for n, line in enumerate(lines, start))


def _line_of(content, index):
    return content.count("\n", 0, index) + 1


def _snippet(content, first_line, last_line):
    lines = content.split("\n")
    lo = max(1, first_line - CONTEXT_LINES)
    hi = min(len(lines), last_line + CONTEXT_LINES)
    return _numbered(lines[lo - 1:hi], lo)


@tool(
    "Read a text file. Output is prefixed with line numbers and a tab (like `cat -n`); "
    "those prefixes are NOT part of the file, never include them in old_str. "
    f"Reads at most {MAX_READ_LINES} lines; use offset/limit for larger files. "
    "You must read a file before editing or overwriting it.",
    {
        "path": {"type": "string", "description": "Path to the file"},
        "offset": {"type": "integer", "description": "1-based line to start from (default 1)"},
        "limit": {"type": "integer", "description": f"Number of lines to read (default {MAX_READ_LINES})"},
    },
    required=["path"],
)
def read_file(path, offset=1, limit=MAX_READ_LINES):
    if not os.path.isfile(path):
        return f"Error: {path} does not exist or is not a file."
    try:
        content = _load(path)
    except UnicodeDecodeError:
        return f"Error: {path} is not a UTF-8 text file."
    _remember(path, content)

    if content == "":
        return f"({path} is empty)"
    lines = content.split("\n")
    if lines[-1] == "":
        lines.pop()
    offset = max(1, offset)
    out, size, shown_to = [], 0, offset - 1
    for n, line in enumerate(lines[offset - 1:offset - 1 + limit], offset):
        line = line.rstrip("\r")
        if len(line) > MAX_LINE_CHARS:  # minified code, data blobs
            line = line[:MAX_LINE_CHARS] + f" [... line truncated, {len(line) - MAX_LINE_CHARS} more characters]"
        entry = f"{n:>6}\t{line}"
        if out and size + len(entry) > MAX_READ_CHARS:
            break
        out.append(entry)
        size += len(entry) + 1
        shown_to = n
    text = "\n".join(out)
    if shown_to < len(lines):
        text += f"\n... ({len(lines) - shown_to} more lines; call again with offset={shown_to + 1})"
    return text


@tool(
    "List the entries of a directory. Directories end with '/'. "
    "Skips .git, virtualenvs, __pycache__ and node_modules.",
    {"path": {"type": "string", "description": "Directory to list (default '.')"}},
    required=[],
)
def list_dir(path="."):
    path = path or "."
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory."
    entries = []
    for name in sorted(os.listdir(path)):
        if name in SKIP_DIRS:
            continue
        full = os.path.join(path, name)
        entries.append(name + "/" if os.path.isdir(full) else name)
    if not entries:
        return f"({path} is empty)"
    extra = len(entries) - MAX_LIST_ENTRIES
    out = "\n".join(entries[:MAX_LIST_ENTRIES])
    if extra > 0:
        out += f"\n... ({extra} more entries)"
    return out


@tool(
    "Create a new file, or completely overwrite an existing one. Prefer str_replace for changing "
    "part of an existing file. Overwriting an existing file requires reading it first.",
    {
        "path": {"type": "string", "description": "Path to the file"},
        "content": {"type": "string", "description": "Full content of the file"},
    },
)
def write_file(path, content):
    if os.path.exists(path):
        if not os.path.isfile(path):
            return f"Error: {path} exists and is not a file."
        error = _check_fresh(path, _load(path))
        if error:
            return error
    _save(path, content)
    _remember(path, content)
    return f"Wrote {len(content.splitlines())} lines to {path}."


def _describe_difference(expected, actual):
    if actual.endswith("\r") and not expected.endswith("\r"):
        return "the file uses Windows line endings (\\r\\n)"
    exp_indent = expected[:len(expected) - len(expected.lstrip())]
    act_indent = actual[:len(actual) - len(actual.lstrip())]
    if exp_indent != act_indent:
        if "\t" in exp_indent + act_indent:
            return f"indentation differs: tabs vs spaces (you sent {exp_indent!r}, file has {act_indent!r})"
        return f"indentation differs (you sent {len(exp_indent)} spaces, file has {len(act_indent)})"
    if expected.rstrip() == actual.rstrip():
        return "trailing whitespace differs"
    return "whitespace inside the line differs"


def _not_found_help(path, content, old_str):
    norm = lambda s: " ".join(s.split())
    file_lines = content.split("\n")
    old_lines = old_str.split("\n")
    if old_lines and old_lines[-1] == "":
        old_lines.pop()
    target = [norm(line) for line in old_lines]

    candidates = [
        i for i in range(len(file_lines) - len(old_lines) + 1)
        if all(norm(file_lines[i + j]) == target[j] for j in range(len(old_lines)))
    ]
    if candidates:
        parts = [f"Error: old_str not found exactly in {path}, but it matches when whitespace is ignored:"]
        for i in candidates[:3]:
            j = next((j for j in range(len(old_lines)) if old_lines[j] != file_lines[i + j]), 0)
            reason = _describe_difference(old_lines[j], file_lines[i + j])
            parts.append(f"- lines {i + 1}-{i + len(old_lines)}: {reason}. "
                         f"First differing line in the file is {file_lines[i + j]!r}")
        parts.append("Resend old_str with the whitespace exactly as it is in the file.")
        return "\n".join(parts)

    first = next((line.strip() for line in old_lines if line.strip()), "")
    hits = [n for n, line in enumerate(file_lines, 1) if first and first in line]
    if hits:
        return (f"Error: old_str not found in {path}. Its first line appears at line(s) "
                f"{', '.join(map(str, hits[:5]))}, but the lines after it differ. "
                f"Call read_file to see the current content.")
    return f"Error: old_str not found in {path}. Call read_file to see the current content."


@tool(
    "Replace one exact occurrence of old_str with new_str in a file. old_str must match the file "
    "exactly (including indentation and whitespace) and must occur exactly once; include enough "
    "surrounding lines to make it unique. Do not include line-number prefixes from read_file. "
    "The file must have been read with read_file first. Returns the edited region with line numbers.",
    {
        "path": {"type": "string", "description": "Path to the file"},
        "old_str": {"type": "string", "description": "Exact text to replace"},
        "new_str": {"type": "string", "description": "Replacement text (may be empty to delete)"},
    },
)
def str_replace(path, old_str, new_str):
    if not os.path.isfile(path):
        return f"Error: {path} does not exist. Use write_file to create a new file."
    content = _load(path)
    error = _check_fresh(path, content)
    if error:
        return error
    if old_str == "":
        return "Error: old_str is empty. To create a file use write_file."
    if old_str == new_str:
        return "Error: old_str and new_str are identical; nothing to change."

    new_content, result = _replace_once(path, content, old_str, new_str)
    if new_content is None:
        return result
    _save(path, new_content)
    _remember(path, new_content)
    return f"Edited {path}. Updated region:\n{result}"


def _replace_once(path, content, old_str, new_str):
    """Return (new_content, snippet) on success, or (None, error message)."""
    count = content.count(old_str)
    if count == 0:
        return None, _not_found_help(path, content, old_str)
    if count > 1:
        parts = [f"Error: the text to replace occurs {count} times in {path}. Include more surrounding "
                 f"lines so it matches exactly one place. Matches:"]
        start = 0
        for _ in range(min(count, 5)):
            idx = content.index(old_str, start)
            line = _line_of(content, idx)
            parts.append(f"--- match at line {line}:\n{_snippet(content, line, line)}")
            start = idx + len(old_str)
        return None, "\n".join(parts)

    idx = content.index(old_str)
    new_content = content[:idx] + new_str + content[idx + len(old_str):]
    first = _line_of(new_content, idx)
    last = _line_of(new_content, idx + len(new_str))
    return new_content, _snippet(new_content, first, last)


_BLOCK_RE = re.compile(
    r"^(?P<path>[^\n`][^\n]*)\n(?:```[^\n]*\n)?<<<<<<< SEARCH\n(?P<search>.*?)^=======\n(?P<replace>.*?)^>>>>>>> REPLACE$",
    re.S | re.M,
)


@tool(
    "Edit one or more files using SEARCH/REPLACE blocks. Each block is:\n"
    "path/to/file.py\n<<<<<<< SEARCH\nexact existing lines\n=======\nnew lines\n>>>>>>> REPLACE\n\n"
    "The SEARCH part must match the file exactly (including indentation) and occur exactly once; "
    "include enough surrounding lines to make it unique. Use whole lines. An empty SEARCH part "
    "creates a new file. Existing files must have been read with read_file first. Blocks are applied "
    "in order; if any block fails, no files are changed.",
    {"edits": {"type": "string", "description": "One or more SEARCH/REPLACE blocks"}},
)
def apply_edits(edits):
    blocks = list(_BLOCK_RE.finditer(edits))
    if not blocks:
        return ("Error: no SEARCH/REPLACE blocks found. Each block must be: a line with the file path, "
                "then '<<<<<<< SEARCH', the old lines, '=======', the new lines, '>>>>>>> REPLACE'.")

    pending = {}  # path -> new content, written only if every block succeeds
    errors, applied = [], []
    for n, block in enumerate(blocks, 1):
        path = block["path"].strip()
        search, replace = block["search"], block["replace"]
        label = f"Block {n} ({path})"

        if path in pending:
            content = pending[path]
        elif os.path.isfile(path):
            content = _load(path)
            error = _check_fresh(path, content)
            if error:
                errors.append(f"{label}: {error}")
                continue
        elif search == "":
            pending[path] = replace
            applied.append(f"{label}: created new file")
            continue
        else:
            errors.append(f"{label}: Error: {path} does not exist. Use an empty SEARCH part to create it.")
            continue

        if search == "":
            errors.append(f"{label}: Error: empty SEARCH on an existing file. Put the lines to change in SEARCH.")
            continue
        new_content, result = _replace_once(path, content, search, replace)
        if new_content is None:
            errors.append(f"{label}: {result}")
            continue
        pending[path] = new_content
        applied.append(f"{label}:\n{result}")

    if errors:
        return "No files were changed.\n\n" + "\n\n".join(errors)
    for path, content in pending.items():
        _save(path, content)
        _remember(path, content)
    return f"Applied {len(blocks)} block(s).\n\n" + "\n\n".join(applied)


def apply_edits_paths(edits):
    """File paths an apply_edits call would touch (used by the permission check)."""
    return [block["path"].strip() for block in _BLOCK_RE.finditer(edits)]
