# Day by day: where each day's code lives

Each `day_N_agent.py` here is a snapshot of the agent at the end of that day, so you can run and
diff how the harness evolved. Days that built something *around* the agent (evals, the evaluator,
the final core) have no snapshot of their own: their code lives in the packages listed below.

| Day | Topic | Code | Run it | Tests |
|---|---|---|---|---|
| 1 | What an LLM API is: raw HTTP, multi-turn by resending history, JSONL token logging | `days/day_1_agent.py` | `python -m days.day_1_agent` | `tests/test_day1.py` |
| 2 | Tool calling: the agent loop, retries with backoff, step limit, parallel tool calls | `days/day_2_agent.py` | `python -m days.day_2_agent` | `tests/test_day2.py` |
| 3 | File tools and edit formats: read tracking, strict `str_replace`, SEARCH/REPLACE | `days/day_3_agent.py`, `tools/files.py`, `tools/registry.py` | `python -m days.day_3_agent` | `tests/test_day3.py` |
| 4 | Shell and search: bash that never hangs, ripgrep search | `days/day_4_agent.py`, `tools/shell.py`, `tools/search.py` | `python -m days.day_4_agent` | `tests/test_day4.py` |
| 5 | Security: permission modes, deny rules, Docker sandbox | `days/day_5_agent.py`, `safety/`, `sandbox/Dockerfile` | `python -m days.day_5_agent` | `tests/test_day5.py` |
| 6 | Evals: tasks with hidden graders, runner, pass@k / pass^k (agent: Day 5) | `evals/` (`runner.py`, `report.py`, `validate.py`, `tasks/`) | `python -m evals.runner --agent day_5_agent ...` | `tests/test_day6.py` |
| 7 | Streaming (hand-parsed SSE), safe Ctrl-C, providers, and a second loop: the text protocol (bash blocks, no tool API) | `days/day_7_agent.py`, `models/sse.py` | `python -m days.day_7_agent --provider groq` (add `--text-protocol` for the text loop) | `tests/test_day7.py` |
| 8 | Context I: system prompt, AGENTS.md, token budget, output caps, caching | `days/day_8_agent.py`, `context/budget.py`, `context/prompt.py` | `python -m days.day_8_agent` | `tests/test_day8.py` |
| 9 | Context II: compaction, history validator, subagents | `days/day_9_agent.py`, `context/compaction.py` | `python -m days.day_9_agent` | `tests/test_day9.py` |
| 10 | Planning and long-running work: todo tool, feature list, multi-session harness | `days/day_10_agent.py`, `tools/todo.py`, `longrun/` | `python -m longrun.session --dir ... --goal ... --model groq:openai/gpt-oss-120b` | `tests/test_day10.py` |
| 11 | Verification: an independent evaluator agent (now a core `Agent`; reviews any agent) | `longrun/evaluator.py` | `python -m evals.runner --agent core:coding --evaluator ...` | `tests/test_day11.py` |
| 12 | Extensibility: MCP client from the spec, hooks, skills | `days/day_12_agent.py`, `ext/` | `python -m days.day_12_agent --mcp-config servers.json` | `tests/test_day12.py` |
| 13 | Ideas from production harnesses: overflow recovery, window-aware output, ask_user | `days/day_13_agent.py`, `tools/ask.py`, `docs/harness_comparison.md` | `python -m days.day_13_agent` | `tests/test_day13.py` |
| 14 | One model-agnostic core, two agents, headless mode, final eval | `core/`, `agents/`, `models/`, `ui/`, `tools/data.py`, `evals/final_eval.py` | `python -m ui.tui` / `python -m ui.headless` | `tests/test_day14.py` |

Notes:
- Days 1-5 only know OpenRouter (it was the only provider then). The eval runner can still run them on
  other providers by swapping just their HTTP call: `python -m evals.runner --agent day_5_agent --provider groq ...`.
- Later days import earlier ones: day 8 builds on day 7; days 9, 10, 12 and 13 on days 7 and 8.
  `longrun/` was built on day 10 and now runs on the final core (`core/` + `agents/`), like the evaluator.
- The shared packages (`tools/`, `safety/`, `context/`, ...) kept improving after the day that
  introduced them, so an old snapshot runs with today's tools, not the exact tools of its day.
- Evals of the Day 7 text loop: `AGENT_PROTOCOL=text python -m evals.runner --agent day_7_agent ...`.
- Run every test suite: `python -m tests.run_all`.
