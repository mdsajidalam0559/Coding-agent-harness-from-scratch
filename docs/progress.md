# Two weeks of progress

What was built each day, how it was checked, and every live measurement so far. Model runs used free
tiers (Groq, Gemini) after the OpenRouter account ran out of credit, so quotas shaped what could be
measured: Groq allows 7k input tokens per request and 200k tokens per model per day.

## Checkpoints

| Day | Checkpoint | Status | Evidence |
|---|---|---|---|
| 1 | Chat CLI with no SDK, every request/response field understood | ✅ | 6 checks; `days/day_1_agent.py`, JSONL token logging |
| 2 | Multi-step task with tools; survives API errors | ✅ | 7 checks; live on OpenRouter |
| 3 | Reliable targeted edits across files | ✅ built, ⏳ comparison | 21 checks; live: Haiku 3/3 on rename+constant before credit ran out |
| 4 | Runs tests, fixes code; interactive commands time out cleanly | ✅ | 22 checks; live: `fix-failing-tests` passed on Groq (gpt-oss-120b, qwen3.8-27b) |
| 5 | A planted malicious instruction causes no damage | ⏳ | 24 permission checks pass; the Docker sandbox and injection checkpoint are written but need the `docker` group |
| 6 | A baseline score table | ✅ partial | runner, report, 8 validated tasks; baseline below |
| 7 | Streaming terminal agent + text-protocol loop | ✅ built, ⏳ comparison | 32 checks; streaming verified live on Groq and Gemini |
| 8 | Cost per task drops | ⏳ | 28 checks; 14% cached prompt tokens observed live on Groq |
| 9 | Long tasks survive compaction | ✅ offline, ⏳ live | 22 checks incl. 300 random histories; live run did not reach the window |
| 10 | 5-feature project across sessions | ⏳ 1/5 | 43 checks; `~/Work/agent-projects/notes-cli`: feature 1 done, resumed across sessions from files; paused by the daily quota (resume: `python -m longrun.session --dir ~/Work/agent-projects/notes-cli --model groq:openai/gpt-oss-120b --sandbox off`) |
| 11 | Evaluator catches bugs the generator claimed fixed | ✅ | 16 checks; live: caught 5 real bugs behind a false "all tests pass" claim (2 not planted) |
| 12 | Agent uses a third-party MCP server's tools | ✅ | 66 checks; live: gpt-oss-20b used `mcp-server-git` correctly; client tested against `server-everything` (stdio + HTTP) |
| 13 | Explainable eval improvement | ✅ built, ⏳ measurement | 30 checks; ports chosen from the baseline's failures (docs/harness_comparison.md) |
| 14 | Two agents on one core + score table | ✅ agents, ⏳ table | 49 checks; both agents solved a task live in headless mode |

392 automated checks pass across the 14 suites (`python -m tests.run_all`).

## Live measurements

| When | Agent | Model | Task | Result |
|---|---|---|---|---|
| Day 3 | day_3 (str_replace) | Claude Haiku 4.5 | rename + constant | 3/3 passed, 7, 7 and 10 steps, ~$0.02-0.03 each |
| Day 7 | day_7 | gpt-oss-120b | fix-failing-tests | passed, 5 steps, 4.3 s |
| Day 9 | day_9 (8k window forced) | gpt-oss-120b | find-bug-in-package | passed, 13 steps; 229 s mostly waiting on rate limits |
| Day 13 | day_12 (baseline) | qwen3.8-27b | 5 coding tasks | 6/7 trials passed; 1 context overflow (7,111 > 7,000 tokens) |
| Day 14 | core:coding (headless) | gpt-oss-20b | pager-off-by-one | passed, 12 steps, 108 s |
| Day 14 | core:data (headless) | gpt-oss-20b | customer-data-quality | passed, 6 steps (handled duplicate rows with `COUNT(DISTINCT id)`) |
| Day 14 | day_5 (Day 6 baseline) | gpt-oss-20b | add-retry-backoff | passed, 9 steps, 14k tokens |

## The final score table

`python -m evals.final_eval` runs the Day 6 baseline (`day_5_agent`), the final core agent and
mini-SWE-agent on the same model and tasks, task by task. The first run completed **1 of 24 runs**
before gpt-oss-20b's daily quota ran out (the baseline passed `add-retry-backoff`). Results are kept
under `evals/results/final-*.jsonl`, and rerunning the command resumes where it stopped.

To finish it: about 15k tokens per agent per task, so roughly 360k tokens for one trial of the 8
tasks with 3 agents. On the free tier that is two days of one model's quota:

    python -m evals.final_eval --provider groq --model openai/gpt-oss-20b     # today, tomorrow: it resumes
    python -m evals.report evals/results/final-baseline.jsonl evals/results/final-final.jsonl evals/results/final-mini-swe.jsonl

## What the numbers do and do not show

- **They show that the harness works end to end on real models and providers**: every checkpoint
  capability except the Docker sandbox has run live at least once.
- **They do not yet show that the final agent scores higher than the baseline.** The completed
  trials are too few (one task can flip a mean by 12 points with 8 tasks), and every agent version
  passed the tasks it completed. On these tasks, with capable models, the differences are more likely
  to show in tokens per task, steps and robustness (overflows, provider quirks) than in pass rate.
- **The main measured failure was context, not reasoning**: a provider limit the agent did not know
  about (7k tokens), and daily token quotas. That finding drove the Day 13 ports (overflow recovery,
  window-aware tool output) and is the clearest two-week lesson: tokens per task is a first-class metric.
- **Robustness fixes found only by running live**: Gemini's parallel tool calls and thought signatures,
  Groq's text error codes and `tool_use_failed`, stale Python bytecode, a harness test run in the wrong
  directory, quota failures counted as attempts. None of these showed up in offline tests.
