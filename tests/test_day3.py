"""Day 3: file tools and str_replace preconditions. Run: python -m tests.test_day3"""
import os
import tempfile

from tools import REGISTRY, execute_tool
from tools import files


def call(name, **args):
    return execute_tool(name, args)


def check(label, result, *expected):
    for text in expected:
        assert text in result, f"{label}: expected {text!r} in:\n{result}"
    print(f"✅ {label}")


def main():
    os.chdir(tempfile.mkdtemp())
    files._read_state.clear()

    names = list(REGISTRY)
    assert {"read_file", "list_dir", "write_file", "str_replace", "bash"} <= set(names), names

    call("write_file", path="pkg/app.py", content="def greet(name):\n    return 'hi ' + name\n\nprint(greet('a'))\n")
    check("list_dir marks directories", call("list_dir"), "pkg/")
    check("write_file creates new files and parent dirs", open("pkg/app.py").read(), "def greet")

    # a file written by someone else must be read before editing
    with open("other.py", "w") as f:
        f.write("x = 1\ny = 2\nx = 1\n")
    check("refuses edit of unread file", call("str_replace", path="other.py", old_str="y = 2", new_str="y = 3"),
          "have not read")
    check("refuses overwrite of unread file", call("write_file", path="other.py", content="z"), "have not read")

    check("read_file numbers lines", call("read_file", path="other.py"), "     1\tx = 1", "     2\ty = 2")
    check("rejects ambiguous match with line numbers", call("str_replace", path="other.py", old_str="x = 1", new_str="x = 9"),
          "occurs 2 times", "match at line 1", "match at line 3")
    check("rejects missing text", call("str_replace", path="other.py", old_str="nope", new_str="x"), "not found")
    check("unique match succeeds with snippet", call("str_replace", path="other.py", old_str="y = 2", new_str="y = 3"),
          "Edited other.py", "     2\ty = 3")
    assert open("other.py").read() == "x = 1\ny = 3\nx = 1\n"

    # the agent's own edit keeps the file fresh, but outside changes make it stale
    check("second edit after own edit works",
          call("str_replace", path="other.py", old_str="x = 1\ny = 3", new_str="x = 0\ny = 3"), "Edited")
    with open("other.py", "a") as f:
        f.write("# changed by bash\n")
    check("refuses edit after file changed on disk", call("str_replace", path="other.py", old_str="y = 3", new_str="y = 4"),
          "changed on disk")

    # near-match diagnostics
    with open("tabs.py", "w") as f:
        f.write("def f():\n\treturn 1\n")
    call("read_file", path="tabs.py")
    check("reports tabs vs spaces", call("str_replace", path="tabs.py", old_str="def f():\n    return 1", new_str="x"),
          "tabs vs spaces", "lines 1-2")

    with open("crlf.py", "w", newline="") as f:
        f.write("a = 1\r\nb = 2\r\n")
    call("read_file", path="crlf.py")
    check("reports CRLF line endings", call("str_replace", path="crlf.py", old_str="a = 1\nb = 2", new_str="x"),
          "\\r\\n")
    check("exact CRLF match works and keeps endings",
          call("str_replace", path="crlf.py", old_str="a = 1\r\nb = 2", new_str="a = 1\r\nb = 5"), "Edited")
    assert open("crlf.py", newline="").read() == "a = 1\r\nb = 5\r\n"

    with open("ws.py", "w") as f:
        f.write("x = 1   \ny = 2\n")
    call("read_file", path="ws.py")
    check("reports trailing whitespace", call("str_replace", path="ws.py", old_str="x = 1\ny = 2", new_str="z"),
          "trailing whitespace")

    call("read_file", path="pkg/app.py")
    check("reports partial match location",
          call("str_replace", path="pkg/app.py", old_str="def greet(name):\n    return 'bye'", new_str="x"),
          "first line appears at line(s) 1")

    lines = "".join(f"line {i}\n" for i in range(1, 51))
    with open("long.txt", "w") as f:
        f.write(lines)
    check("read_file offset/limit", call("read_file", path="long.txt", offset=10, limit=3),
          "    10\tline 10", "    12\tline 12", "offset=13")

    # SEARCH/REPLACE blocks
    def block(path, search, replace):
        return f"{path}\n<<<<<<< SEARCH\n{search}=======\n{replace}>>>>>>> REPLACE\n"

    with open("m1.py", "w") as f:
        f.write("a = 1\nb = 2\n")
    with open("m2.py", "w") as f:
        f.write("c = 3\n")
    check("apply_edits refuses unread files", call("apply_edits", edits=block("m1.py", "a = 1\n", "a = 10\n")),
          "No files were changed", "have not read")
    call("read_file", path="m1.py")
    call("read_file", path="m2.py")
    check("apply_edits edits several files at once", call("apply_edits", edits=(
        block("m1.py", "a = 1\n", "a = 10\n") + block("m1.py", "b = 2\n", "b = 20\n")
        + block("m2.py", "c = 3\n", "c = 30\n") + block("new/m3.py", "", "d = 4\n"))),
        "Applied 4 block(s)", "created new file")
    assert open("m1.py").read() == "a = 10\nb = 20\n" and open("m2.py").read() == "c = 30\n"
    assert open("new/m3.py").read() == "d = 4\n"

    check("apply_edits is all-or-nothing", call("apply_edits", edits=(
        block("m1.py", "a = 10\n", "a = 99\n") + block("m2.py", "missing\n", "x\n"))),
        "No files were changed", "Block 2 (m2.py)", "not found")
    assert open("m1.py").read() == "a = 10\nb = 20\n"

    check("apply_edits tolerates code fences", call("apply_edits",
          edits="m2.py\n```python\n<<<<<<< SEARCH\nc = 30\n=======\nc = 31\n>>>>>>> REPLACE\n```"), "Applied 1")
    check("apply_edits explains bad format", call("apply_edits", edits="just some text"), "no SEARCH/REPLACE blocks")

    print("\nDay 3 tool tests passed.")


if __name__ == "__main__":
    main()
