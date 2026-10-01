import os
import re
import signal
import subprocess

from .registry import tool

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
MAX_STREAM_CHARS = 8000  # per stream; the tail is kept longer than the head (errors and summaries are at the end)

# Anything that could otherwise stop and wait for a human
NON_INTERACTIVE_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_EDITOR": "true",
    "GIT_PAGER": "cat",
    "EDITOR": "true",
    "VISUAL": "true",
    "PAGER": "cat",
    "DEBIAN_FRONTEND": "noninteractive",
    "PYTHONUNBUFFERED": "1",
    # .pyc staleness is checked by mtime (whole seconds) and size: an agent that edits `a + b` -> `a * b`
    # within a second of the last run gets the OLD compiled code. Never write bytecode caches.
    "PYTHONDONTWRITEBYTECODE": "1",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(\x07|\x1b\\)|\x1b[@-Z\\-_]")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def truncate(text, limit=MAX_STREAM_CHARS):
    if len(text) <= limit:
        return text
    head, tail = limit * 3 // 10, limit * 7 // 10
    return f"{text[:head]}\n\n... [{len(text) - head - tail} characters truncated] ...\n\n{text[-tail:]}"


# Set by the agent to run commands inside a container (safety.sandbox.DockerSandbox); None = run on the host
SANDBOX = None


def _run_on_host(command, timeout):
    """Returns (returncode, stdout, stderr, timed_out)."""
    proc = subprocess.Popen(
        ["bash", "-c", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env={**os.environ, **NON_INTERACTIVE_ENV},
        start_new_session=True,  # own process group, and no controlling terminal (/dev/tty) to prompt on
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        stdout, stderr = proc.communicate()
        return None, stdout, stderr, True
    except KeyboardInterrupt:
        # the command runs in its own session, so Ctrl-C never reached it: stop it ourselves
        _kill_group(proc)
        raise


def _kill_group(proc):
    """Kill the command and every process it started (they share one process group)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            continue


@tool(
    "Run a bash command and return its exit code, stdout and stderr. "
    "Each call starts a fresh shell in the project directory: `cd` does not carry over between calls, "
    "so use `cd dir && command`. Commands cannot read input (stdin is closed) and there is no terminal, "
    "so interactive programs fail instead of waiting; use non-interactive flags (e.g. `-y`, `--no-pager`). "
    f"Commands are killed after `timeout` seconds (default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT}). "
    "Do not start servers or watchers in the foreground; to run something in the background, redirect its "
    "output, e.g. `nohup cmd > server.log 2>&1 &`. Long output is truncated in the middle.",
    {
        "command": {"type": "string", "description": "The bash command to run"},
        "timeout": {"type": "integer", "description": f"Seconds before the command is killed (default {DEFAULT_TIMEOUT})"},
    },
    required=["command"],
)
def bash(command, timeout=DEFAULT_TIMEOUT):
    timeout = max(1, min(int(timeout), MAX_TIMEOUT))
    if SANDBOX is not None:
        returncode, stdout, stderr, timed_out = SANDBOX.exec(command, timeout, NON_INTERACTIVE_ENV)
    else:
        returncode, stdout, stderr, timed_out = _run_on_host(command, timeout)

    parts = []
    if timed_out:
        parts.append(f"Command timed out after {timeout}s and was killed (with any processes it started). "
                     "If it was waiting for input, use a non-interactive flag; if it is a server, run it in the "
                     "background with output redirected. Partial output below.")
    else:
        parts.append(f"exit code: {returncode}")
        if returncode == 137:
            parts.append("(killed with SIGKILL, most likely for using too much memory)")
    stdout, stderr = strip_ansi(stdout).strip("\n"), strip_ansi(stderr).strip("\n")
    if stdout:
        parts.append(f"--- stdout ---\n{truncate(stdout)}")
    if stderr:
        parts.append(f"--- stderr ---\n{truncate(stderr)}")
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts)
