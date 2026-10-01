"""Run every day's test suite and summarize: python -m tests.run_all [day numbers...]"""
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    wanted = set(sys.argv[1:])
    suites = sorted(glob.glob(os.path.join(HERE, "test_day*.py")), key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    failed = []
    for path in suites:
        day = re.search(r"test_day(\d+)", path).group(1)
        if wanted and day not in wanted:
            continue
        proc = subprocess.run([sys.executable, "-m", f"tests.test_day{day}"], cwd=REPO, capture_output=True, text=True, timeout=1800)
        passed = (proc.stdout + proc.stderr).count("✅")
        last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
        ok = proc.returncode == 0
        print(f"{'✅' if ok else '❌'} day {day:>2}: {passed:>3} checks   {last[:90]}")
        if not ok:
            failed.append(day)
            print("   " + "\n   ".join((proc.stdout + proc.stderr).strip().splitlines()[-6:]))
    print(f"\n{'all suites passed' if not failed else 'FAILED: days ' + ', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
