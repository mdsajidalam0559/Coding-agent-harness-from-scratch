"""Check every task is well-formed: the untouched repo must FAIL grading and the reference solution must PASS.

    python -m evals.validate [--sandbox docker]

A task that passes without any work, or that even the reference solution cannot pass, would make
the eval score meaningless. Run this after adding or changing a task.
"""
import argparse
import os
import shutil
import tempfile

from evals.runner import grade, load_tasks, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", choices=["docker", "off"], default="off")
    parser.add_argument("--suite", choices=["coding", "data"], default="coding")
    args = parser.parse_args()

    ok = True
    for task in load_tasks(suite=args.suite):
        results = {}
        for with_solution in (False, True):
            workdir = os.path.join(tempfile.mkdtemp(), "work")
            prepare(task, workdir, with_solution)
            results[with_solution], output = grade(task, workdir, args.sandbox)
            shutil.rmtree(os.path.dirname(workdir))
        good = results[False] is False and results[True] is True
        ok &= good
        print(f"{'✅' if good else '❌'} {task['name']:24} unsolved repo: {'PASS' if results[False] else 'fail'}   "
              f"reference solution: {'PASS' if results[True] else 'fail'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
