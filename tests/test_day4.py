"""Day 4: bash and search tools. Run: python -m tests.test_day4"""
import os
import tempfile
import time

from tools import execute_tool


def call(name, **args):
    return execute_tool(name, args)


def check(label, result, *expected, absent=()):
    for text in expected:
        assert text in result, f"{label}: expected {text!r} in:\n{result}"
    for text in absent:
        assert text not in result, f"{label}: did not expect {text!r} in:\n{result}"
    print(f"✅ {label}")


def timed(fn):
    start = time.monotonic()
    result = fn()
    return result, time.monotonic() - start


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a killed child may linger as a zombie until reaped; that counts as dead
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split()[2] != "Z"


def main():
    os.chdir(tempfile.mkdtemp())

    # --- bash: results ---
    check("exit code and stdout", call("bash", command="echo hello"), "exit code: 0", "--- stdout ---\nhello")
    check("stderr and non-zero exit", call("bash", command="echo oops >&2; exit 3"),
          "exit code: 3", "--- stderr ---\noops", absent=("--- stdout ---",))
    check("empty output is explicit", call("bash", command="true"), "(no output)")
    check("ANSI colour codes stripped", call("bash", command=r"printf '\033[31mred\033[0m \033[1;32mgreen\033[0m'"),
          "red green", absent=("\x1b",))

    big = call("bash", command="python3 -c \"print('A' * 20000); print('THE END')\"")
    check("long output truncated, tail kept", big, "characters truncated", "THE END")
    assert len(big) < 9000, len(big)

    # --- bash: never hang ---
    result, secs = timed(lambda: call("bash", command="read line; echo got:$line"))
    check("stdin is closed (read gets EOF)", result, "got:")
    assert secs < 5, secs

    result, secs = timed(lambda: call("bash", command="python3 -c \"input('Name? ')\""))
    check("input() fails fast instead of waiting", result, "EOFError")
    assert secs < 5, secs

    result, secs = timed(lambda: call("bash", command="python3 -c \"open('/dev/tty').read()\""))
    check("no controlling terminal to prompt on", result, "exit code: 1")
    assert secs < 5, secs

    result, secs = timed(lambda: call("bash", command="git clone https://github.com/this-org-does-not-exist-xyz/private-repo", timeout=20))
    check("git does not prompt for credentials", result, "exit code:", absent=("timed out",))
    assert secs < 20, secs

    check("pager disabled", call("bash", command="echo $PAGER $GIT_PAGER $GIT_TERMINAL_PROMPT"), "cat cat 0")

    # --- bash: timeouts kill the whole process group ---
    result, secs = timed(lambda: call("bash", command="echo started; sleep 30", timeout=1))
    check("timeout kills and keeps partial output", result, "timed out after 1s", "started")
    assert secs < 5, secs

    result, _ = timed(lambda: call("bash", command="sleep 60 & echo $! > child.pid; wait", timeout=1))
    child = int(open("child.pid").read())
    time.sleep(0.2)
    check("timeout also kills child processes", result, "timed out")
    assert not alive(child), f"child {child} survived the timeout"

    result, secs = timed(lambda: call("bash", command="nohup sleep 60 > bg.log 2>&1 & echo $! > bg.pid; echo launched"))
    check("backgrounded command with redirected output returns immediately", result, "launched")
    assert secs < 5, secs
    os.kill(int(open("bg.pid").read()), 9)

    check("timeout is capped", call("bash", command="echo ok", timeout=99999), "exit code: 0")

    # --- search ---
    os.makedirs("src")
    with open("src/app.py", "w") as f:
        f.write("def load_config(path):\n    return open(path).read()\n\nconfig = load_config('x')\n")
    with open("src/util.py", "w") as f:
        f.write("# TODO: remove\nimport os\n")
    with open("notes.md", "w") as f:
        f.write("load_config is documented here\n")
    with open("min.js", "w") as f:
        f.write("var a=1;" * 2000 + "needle")
    os.makedirs(".git")
    with open(".gitignore", "w") as f:
        f.write("ignored/\n")
    os.makedirs("ignored")
    with open("ignored/x.py", "w") as f:
        f.write("load_config\n")

    check("search finds matches with path and line", call("search", pattern="load_config"),
          "src/app.py:1:def load_config(path):", "src/app.py:4:config", "notes.md:1:", absent=("ignored/",))
    check("search glob filter", call("search", pattern="load_config", glob="*.py"), "src/app.py", absent=("notes.md",))
    check("search path filter", call("search", pattern="import", path="src"), "src/util.py:2:import os")
    check("search case-insensitive", call("search", pattern="todo", ignore_case=True), "src/util.py:1:# TODO")
    check("search no matches", call("search", pattern="nonexistent_symbol"), "No matches")
    check("search invalid regex reports error", call("search", pattern="load_config("), "Error:")
    check("search shortens very long lines", call("search", pattern="needle"), "min.js:1:", absent=("var a=1;" * 100,))
    with open("many.txt", "w") as f:
        f.write("hit\n" * 500)
    check("search caps number of matches", call("search", pattern="hit", path="many.txt"), "more matches")

    print("\nDay 4 tool tests passed.")


if __name__ == "__main__":
    main()
