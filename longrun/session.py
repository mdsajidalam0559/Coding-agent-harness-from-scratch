"""Build a project across many sessions, resuming each time from files, not chat memory.

    python -m longrun.session --dir ~/projects/notes-cli --goal "A command-line notes app ..." --provider groq
    python -m longrun.session --dir ~/projects/notes-cli --sessions 3        # resume later

Session 0 (the initializer) turns the goal into a project skeleton and feature_list.json.
Every later session starts with an EMPTY context. It is given one feature, the plan, the latest
progress notes and the git log; it implements and tests the feature and claims it done. The harness
then runs the tests itself, records the real status, appends to progress.md and commits.
"""
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
from datetime import datetime

from days import day_10_agent as agent
import tools.shell
from longrun import evaluator, features
from safety.permissions import Policy
from tools import execute_tool, files, todo

PROGRESS_FILE = "progress.md"
LOG_DIR = ".agent"  # session transcripts, git-ignored
MAX_ATTEMPTS = 3  # sessions per feature before it is marked blocked
MAX_REVIEW_ROUNDS = 2  # evaluator feedback rounds inside one session
USE_EVALUATOR = True  # Day 11: an independent agent reviews every 'done' claim

INITIALIZER_PROMPT = """You are setting up a project that will be built over several separate sessions. Each later session starts with NO memory of this conversation: only the files in this directory carry over.

Project goal:
{goal}

Do this now:
1. Look at what is already in the directory.
2. Create the project skeleton: the package/module layout and a tests/ directory using Python's built-in unittest, so the test command works without installing anything. Add a short README.md with the goal and how to run the tests.
3. Split the goal into 3 to 8 features in build order. Each must be small enough for one session and testable on its own. Record them with create_feature_list, together with the command that runs all tests.

Do NOT implement the features. Finish with a short summary of the setup."""

SESSION_PROMPT = """You are continuing a multi-session project. You have no memory of earlier sessions; everything you need is in the files and below.

Project goal:
{goal}

Plan (feature_list.json):
{plan}

Latest progress notes (progress.md):
{progress}

Recent commits:
{commits}

Test status at the start of this session: {test_status}

YOUR TASK THIS SESSION: feature {id}: {title}
{description}
Acceptance checks:
{acceptance}{history}

How to work:
1. Get oriented: read the relevant files and run the tests (`{test_command}`) to see the current state.
2. Implement ONLY this feature, with unit tests for it in tests/.
3. Run the whole test suite. It must pass completely, including any tests that were already failing when you started.
4. When the feature works and every test passes, call update_feature(id={id}, status="done", note=...). If you cannot finish, call it with status="blocked" and explain why.
5. Then reply with a short summary for the progress log: what you did, which files changed, and anything the next session should know.

Do not edit feature_list.json or progress.md; the harness maintains them. Do not start other features."""

REPAIR_PROMPT = """The project's test command `{command}` does not work yet. Its output:

{output}

Fix the project skeleton so that the command runs successfully (running zero tests is fine at this point). If the command itself is wrong for this project, call create_feature_list again with the same features and a working test_command. Do NOT implement any features."""


def git(workspace, *args):
    # identity passed per command, so the user's git config is never touched
    return subprocess.run(["git", "-c", "user.name=coding-agent", "-c", "user.email=coding-agent@localhost", *args],
                          cwd=workspace, capture_output=True, text=True)


def ensure_repo(workspace):
    if not os.path.isdir(os.path.join(workspace, ".git")):
        git(workspace, "init", "-q")
    ignore = os.path.join(workspace, ".gitignore")
    lines = open(ignore).read().splitlines() if os.path.exists(ignore) else []
    missing = [entry for entry in (f"{LOG_DIR}/", "__pycache__/", ".env") if entry not in lines]
    if missing:
        with open(ignore, "a") as f:
            f.write("".join(f"{entry}\n" for entry in missing))


def commit(workspace, message):
    git(workspace, "add", "-A")
    git(workspace, "commit", "-q", "--allow-empty", "-m", message)
    return git(workspace, "rev-parse", "--short", "HEAD").stdout.strip()


def recent_commits(workspace, n=10):
    return git(workspace, "log", "--oneline", f"-{n}").stdout.strip() or "(no commits yet)"


def parse_test_run(output):
    """What a test run says: exit code, how many tests ran, which failed (unittest or pytest output)."""
    match = re.match(r"exit code: (-?\d+)", output)
    code = int(match.group(1)) if match else None  # no exit code: the run timed out
    ran = re.search(r"^Ran (\d+) tests?", output, re.M)
    if ran:
        count = int(ran.group(1))
    else:
        counts = re.findall(r"(\d+) (?:passed|failed|errors?)\b", output)
        count = sum(int(n) for n in counts) if counts else None
    failing = re.findall(r"^(?:FAIL|ERROR): (\S+ \([^)]*\))", output, re.M) + \
        re.findall(r"^(?:FAILED|ERROR) (\S+::\S+)", output, re.M)
    no_tests = code == 5 or count == 0  # exit 5 = "no tests ran" (unittest on Python 3.12+, pytest)
    return {"code": code, "passed": code == 0 and not no_tests, "no_tests": no_tests,
            "count": count or 0, "failing": sorted(set(failing)), "output": output}


def describe(result):
    if result["code"] is None:
        return "the test run TIMED OUT"
    if result["passed"]:
        return f"all {result['count']} tests pass"
    if result["no_tests"] and result["code"] in (0, 5):
        return "no tests yet"
    failing = result["failing"]
    names = f": {', '.join(failing[:5])}" + (" ..." if len(failing) > 5 else "") if failing else ""
    return f"FAILING ({len(failing) or 'some'} failing{names}; exit code {result['code']})"


def run_tests(workspace, command):
    """The harness's own check, run in the project (and in the sandbox when one is active)."""
    home = os.getcwd()
    os.chdir(workspace)
    try:
        output = execute_tool("bash", {"command": command, "timeout": 300})
    finally:
        os.chdir(home)
    return parse_test_run(output)


def read_progress(workspace, chars=3000):
    p = os.path.join(workspace, PROGRESS_FILE)
    if not os.path.exists(p):
        return "(none yet)"
    text = open(p).read()
    return text if len(text) <= chars else "[... earlier notes omitted ...]\n" + text[-chars:]


def append_progress(workspace, text):
    with open(os.path.join(workspace, PROGRESS_FILE), "a") as f:
        f.write(text.rstrip() + "\n\n")


def api_failure(events):
    """The provider error that ended a session (quota, outage), if any. Not the agent's fault, so it must
    not count as an attempt. A context overflow IS the agent's problem and is not reported here."""
    from context.budget import context_overflow
    for event in reversed(events):
        if event["type"] == "api_error" and (event["data"].get("final") or not event["data"]["retryable"]):
            if context_overflow(event["data"])[0]:
                return None
            error = event["data"].get("error")
            message = error.get("message", "") if isinstance(error, dict) else str(error)
            return f"API error {event['data'].get('status')}: {message[:300]}"
        if event["type"] == "response":
            return None
    return None


def run_session(workspace, log_name, prompt, extra_tools, allow_create, quiet):
    """One agent session with a completely fresh context. Returns (final reply, steps, interrupted, api_error)."""
    agent.messages.clear()
    agent.turn_count = 0
    files._read_state.clear()
    todo.reset()
    agent.policy = Policy(workspace, mode="auto", ask=None, protected=(features.FEATURE_FILE, PROGRESS_FILE))
    agent.log_file = os.path.join(workspace, LOG_DIR, f"{log_name}.jsonl")
    agent.EXTRA_TOOLS[:] = extra_tools
    features.STATE.update(claims={}, allow_create=allow_create)

    home = os.getcwd()
    os.chdir(workspace)
    try:
        with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
            agent.agentic_loop(prompt)
    finally:
        os.chdir(home)
        agent.EXTRA_TOOLS[:] = []
    replies = [m.get("content") or "" for m in agent.messages if m.get("role") == "assistant" and not m.get("tool_calls")]
    events = [json.loads(line) for line in open(agent.log_file)] if os.path.exists(agent.log_file) else []
    types = [e["type"] for e in events]
    return (replies[-1] if replies else ""), types.count("response"), "interrupted" in types, api_failure(events)


def judge(claim, after, feature):
    """The harness's own verdict on a claim: (status, reason)."""
    if claim.get("status") == "done":
        if not after["passed"]:
            return "failing", "claimed done, but the harness's test run fails"
        if after["count"] <= feature["tests_before"]:
            return "failing", "claimed done, but no tests were added for this feature"
        return "done", ""
    if claim.get("status") == "blocked":
        return "blocked", ""
    return "in_progress", "the session ended without claiming the feature done"


def continue_session(workspace, message, quiet):
    """Send one more message to the generator of the current session (it keeps its context)."""
    agent.EXTRA_TOOLS[:] = ["update_feature"]
    logged_before = sum(1 for _ in open(agent.log_file))
    home = os.getcwd()
    os.chdir(workspace)
    try:
        with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
            agent.agentic_loop(message)
    finally:
        os.chdir(home)
        agent.EXTRA_TOOLS[:] = []
    replies = [m.get("content") or "" for m in agent.messages if m.get("role") == "assistant" and not m.get("tool_calls")]
    new_events = [json.loads(line) for line in open(agent.log_file)][logged_before:]
    types = [e["type"] for e in new_events]
    return (replies[-1] if replies else ""), types.count("response"), "interrupted" in types, api_failure(new_events)


def initialize(workspace, goal, quiet):
    features.STATE.update(workspace=workspace, plan={"goal": goal, "test_command": None, "features": []})
    summary, steps, _, api_error = run_session(workspace, "session-00", INITIALIZER_PROMPT.format(goal=goal),
                                               ["create_feature_list"], allow_create=True, quiet=quiet)
    if api_error:
        raise SystemExit(f"Setup stopped by the model provider ({api_error}). Run the same command again later.")
    plan = features.STATE["plan"]
    if not plan["features"]:
        raise SystemExit("The initializer did not create a feature list; see "
                         f"{os.path.join(workspace, LOG_DIR, 'session-00.jsonl')}")

    # A broken test command would make every later session "fail": check it now, allow one repair.
    check = run_tests(workspace, plan["test_command"])
    if not (check["passed"] or check["no_tests"]):
        print(f"Session 0 · test command `{plan['test_command']}` does not work; one repair attempt...")
        run_session(workspace, "init-repair", REPAIR_PROMPT.format(command=plan["test_command"],
                                                                  output=check["output"][-3000:]),
                    ["create_feature_list"], allow_create=True, quiet=quiet)
        plan = features.STATE["plan"]
        check = run_tests(workspace, plan["test_command"])
        if not (check["passed"] or check["no_tests"]):
            raise SystemExit(f"The test command `{plan['test_command']}` still does not work:\n"
                             f"{check['output'][-1500:]}\nFix the project setup by hand, then run again.")

    features.save(workspace, plan)
    append_progress(workspace, f"# Progress log\n\nGoal: {goal}\n\n"
                               f"## Session 0: initialized ({datetime.now():%Y-%m-%d %H:%M})\n"
                               f"- {len(plan['features'])} features planned; test command: `{plan['test_command']}` "
                               f"({describe(check)})\n"
                               f"- Initializer's summary: {summary.strip()[:1500]}")
    sha = commit(workspace, f"session 0: initialize project ({len(plan['features'])} features)")
    print(f"Session 0 · initialized {len(plan['features'])} features · {steps} steps · commit {sha}")
    print(features.render(plan))


def provider_stopped(workspace, plan, feature, number, steps, api_error):
    """The provider failed (quota, outage): commit what exists, leave the feature's attempts untouched."""
    if feature["status"] == "pending" and steps:
        feature["status"] = "in_progress"
    feature["notes"].append(f"session {number}: stopped by the model provider ({api_error[:150]}); not counted")
    features.save(workspace, plan)
    append_progress(workspace, f"## Session {number}: feature {feature['id']} ({feature['title']}) -> stopped by the "
                               f"provider ({datetime.now():%Y-%m-%d %H:%M})\n- {api_error}\n- Not counted as an attempt.")
    sha = commit(workspace, f"session {number}: feature {feature['id']} {feature['title']} -> provider error (not counted)")
    print(f"Session {number} · stopped by the model provider, not counted: {api_error[:200]}\n"
          f"   Run the same command again later to continue.")
    return {"feature": feature["id"], "status": "api_error", "tests_pass": None, "steps": steps, "commit": sha}


def work_session(workspace, number, quiet):
    """Returns the outcome dict, or None when nothing is left to do."""
    plan = features.load(workspace)
    features.STATE.update(workspace=workspace, plan=plan)
    feature = features.pick_next(plan)
    if feature is None:
        return None

    before = run_tests(workspace, plan["test_command"])
    feature.setdefault("tests_before", before["count"])  # test count when work on this feature began
    history = ""
    if feature["notes"]:
        history = "\n\nThis feature was attempted before. Notes from those sessions:\n" + \
                  "\n".join(f"- {note}" for note in feature["notes"][-3:])
    prompt = SESSION_PROMPT.format(
        goal=plan["goal"], plan=features.render(plan), progress=read_progress(workspace),
        commits=recent_commits(workspace), test_status=describe(before), id=feature["id"], title=feature["title"],
        description=feature["description"], acceptance="\n".join(f"- {a}" for a in feature["acceptance"]),
        history=history, test_command=plan["test_command"])
    summary, steps, interrupted, api_error = run_session(workspace, f"session-{number:02d}", prompt, ["update_feature"],
                                                         allow_create=False, quiet=quiet)
    if api_error:  # quota or outage: record it, but do not count it against the feature
        return provider_stopped(workspace, plan, feature, number, steps, api_error)

    if interrupted:  # stop the run, but keep the partial work in its own commit
        feature["status"] = "in_progress"
        feature["notes"].append(f"session {number}: interrupted by the user; work in progress was committed")
        features.save(workspace, plan)
        append_progress(workspace, f"## Session {number}: feature {feature['id']} ({feature['title']}) -> "
                                   f"interrupted ({datetime.now():%Y-%m-%d %H:%M})\n"
                                   f"- Stopped by the user; unfinished work committed as-is.")
        sha = commit(workspace, f"session {number}: feature {feature['id']} {feature['title']} -> interrupted (WIP)")
        print(f"Session {number} · interrupted · work in progress committed as {sha}")
        return {"feature": feature["id"], "status": "interrupted", "tests_pass": None, "steps": steps, "commit": sha}

    spec = (f"Feature {feature['id']}: {feature['title']}\n{feature['description']}\nAcceptance checks:\n"
            + "\n".join(f"- {a}" for a in feature["acceptance"]))
    reviews = []
    while True:
        after = run_tests(workspace, plan["test_command"])
        claim = features.STATE["claims"].get(feature["id"], {})
        status, verdict = judge(claim, after, feature)
        if status != "done" or not USE_EVALUATOR:
            break
        review = evaluator.run_evaluator(agent, workspace, spec, f"{claim.get('note', '')}\n{summary}",
                                         evaluator.git_diff(workspace), plan["test_command"], quiet)
        reviews.append(review)
        if review["verdict"] != "fail":  # pass, or no verdict (an evaluator malfunction must not block progress)
            if review["verdict"] == "none":
                verdict = "tests pass, but the evaluator gave no verdict"
            break
        if len(reviews) > MAX_REVIEW_ROUNDS:
            status, verdict = "failing", "the evaluator still found problems: " + \
                "; ".join(i["problem"] for i in review["issues"])[:600]
            break
        # send the findings back to the generator, which still has its context from this session
        features.STATE["claims"].pop(feature["id"], None)
        summary, more_steps, interrupted, api_error = continue_session(workspace, evaluator.FEEDBACK_PROMPT.format(
            issues=evaluator.format_issues(review),
            claim_again=f'When everything works, call update_feature(id={feature["id"]}, status="done", note=...) again. '),
            quiet)
        steps += more_steps
        if api_error:
            return provider_stopped(workspace, plan, feature, number, steps, api_error)
        if interrupted:
            status, verdict = "in_progress", "interrupted during evaluator feedback"
            break
    new_failures = [t for t in after["failing"] if t not in before["failing"]]
    feature["attempts"] += 1
    if status in ("failing", "in_progress") and feature["attempts"] >= MAX_ATTEMPTS:
        status, verdict = "blocked", f"{verdict}; gave up after {MAX_ATTEMPTS} sessions"
    feature["status"] = status
    feature["notes"].append(f"session {number}: {status}; tests: {describe(after)}"
                            + (f"; broke: {', '.join(new_failures)}" if new_failures else "")
                            + (f"; {verdict}" if verdict else "")
                            + f"; {claim.get('note') or summary.strip()[:300]}")
    features.save(workspace, plan)  # also undoes any direct edits the agent made to the file

    review_lines = "".join(f"- Evaluator round {n}: {r['verdict']}. {r['summary'][:300]}\n"
                           + "".join(f"  - {i['problem'][:300]}\n" for i in r["issues"])
                           for n, r in enumerate(reviews, 1))
    append_progress(workspace, f"## Session {number}: feature {feature['id']} ({feature['title']}) -> {status} "
                               f"({datetime.now():%Y-%m-%d %H:%M})\n"
                               f"- Agent's claim: {claim.get('status', 'none')}; harness: {verdict or 'confirmed'}\n"
                               f"- Tests before: {describe(before)}; after: {describe(after)}\n"
                               + (f"- New failures this session: {', '.join(new_failures)}\n" if new_failures else "")
                               + review_lines
                               + f"- Summary: {summary.strip()[:1500] or '(no summary)'}")
    sha = commit(workspace, f"session {number}: feature {feature['id']} {feature['title']} -> {status}")
    print(f"Session {number} · feature {feature['id']} '{feature['title']}' -> {status} "
          f"(claim: {claim.get('status', 'none')}; tests: {describe(after)}) · {steps} steps · commit {sha}")
    if verdict and not quiet:
        print(f"   {verdict}")
    return {"feature": feature["id"], "status": status, "tests_pass": after["passed"], "steps": steps, "commit": sha,
            "new_failures": new_failures, "reviews": [r["verdict"] for r in reviews]}


def main():
    parser = argparse.ArgumentParser(description="Build a project across many fresh-context sessions")
    parser.add_argument("--dir", required=True, help="project directory (created if missing)")
    parser.add_argument("--goal", help="what to build (required for a new project)")
    parser.add_argument("--sessions", type=int, default=10, help="max work sessions this run")
    parser.add_argument("--provider", choices=list(agent.PROVIDERS))
    parser.add_argument("--model")
    parser.add_argument("--sandbox", choices=["docker", "off"], default="docker")
    parser.add_argument("--quiet", action="store_true", help="only print one line per session")
    parser.add_argument("--no-evaluator", action="store_true", help="skip the Day 11 evaluator review")
    args = parser.parse_args()

    global USE_EVALUATOR
    USE_EVALUATOR = not args.no_evaluator
    agent.PROVIDER = args.provider or agent.PROVIDER
    agent.MODEL = args.model or os.getenv("AGENT_MODEL") or agent.PROVIDERS[agent.PROVIDER]["default_model"]
    agent.base.provider_settings(agent.PROVIDER)
    workspace = os.path.realpath(os.path.expanduser(args.dir))
    os.makedirs(os.path.join(workspace, LOG_DIR), exist_ok=True)
    ensure_repo(workspace)

    sandbox = None
    if args.sandbox == "docker":
        from safety.sandbox import DockerSandbox
        sandbox = DockerSandbox(workspace).start()
        tools.shell.SANDBOX = sandbox
    else:
        print("⚠️  --sandbox off: the agent's commands run unattended on this machine.")
    print(f"Project: {workspace}   Provider: {agent.PROVIDER}   Model: {agent.MODEL}\n")

    try:
        if not os.path.exists(features.path(workspace)):
            if not args.goal:
                raise SystemExit("New project: pass --goal to describe what to build.")
            initialize(workspace, args.goal, args.quiet)
        start = sum(1 for name in os.listdir(os.path.join(workspace, LOG_DIR))
                    if re.fullmatch(r"session-\d+\.jsonl", name))
        for number in range(start, start + args.sessions):
            outcome = work_session(workspace, number, args.quiet)
            if outcome is None or outcome["status"] in ("interrupted", "api_error"):
                break
        plan = features.load(workspace)
        print(f"\n{features.render(plan)}")
        done = sum(f["status"] == "done" for f in plan["features"])
        print(f"\n{done}/{len(plan['features'])} features done. Progress log: {os.path.join(workspace, PROGRESS_FILE)}")
    finally:
        if sandbox:
            sandbox.stop()
            tools.shell.SANDBOX = None


if __name__ == "__main__":
    main()
