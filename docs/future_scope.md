# Future scope

What is deliberately not built yet, roughly in the order I would do it. Sizes are rough: S = an hour or
two, M = a day.

## 1. Finish what exists (no new features)

| Item | Why | Size |
|---|---|---|
| **Final eval table** (`python -m evals.final_eval`, resumes where it stopped) | The README's main claim (the final agent beats the Day 6 baseline, and how it compares with mini-SWE-agent) has 1 of 24 runs behind it. About 360k tokens: two days of one free model's quota, or a small paid budget. | S (just runs) |
| **Day 5 checkpoint in Docker** | The sandbox and the prompt-injection test are written but have never run on this machine (needs the `docker` group). Every eval so far ran unsandboxed. | S |
| **Day 10 checkpoint** | The notes-cli build has 1 of 5 features; resume it (`python -m longrun.session --dir ~/Work/agent-projects/notes-cli --model groq:openai/gpt-oss-120b --sandbox off`). | S |
| **Day 13 comparison** (day 12 vs day 13 on one model) | The "explainable improvement" checkpoint: does overflow recovery + window-aware output cut overflows and tokens per task? | S |
| **Smaller comparisons**: edit formats (day 3), native vs text protocol (day 7), tokens before/after context work (day 8), with/without the evaluator (day 11) | Each turns a design choice into a measured one. | S each |
| **More eval tasks, from your own work** (~20 coding, more data tasks) | With 8 tasks, one task flipping moves the mean by 12 points. | M |
| **Live test of the Anthropic adapter** | Only tested offline against recorded streams; needs an `ANTHROPIC_API_KEY`. | S |
| **CI**: `.github/workflows/tests.yml` running `python -m tests.run_all` on push | Catches breakage and doc drift; no API calls needed. | S |

## 2. Safety and day-to-day use

| Item | Why | Size |
|---|---|---|
| **Undo / checkpoints** | Snapshot files before each write (Claude Code), auto-commit to a shadow branch (Aider), or `git stash create` per turn. The second safety layer next to the sandbox. Until then, git is the undo inside a repo. | M |
| **Plan mode** | A read-only phase that ends with a plan you approve before any edit. Easy on top of the permission modes; teaches steering. | S |
| **Network allowlist for the sandbox** | Network is all or nothing. A proxy that allows only e.g. PyPI is how Codex and Claude Code on the web do it; the "lethal trifecta" made concrete. | M |
| **Model-based permission classifier** | Rules catch only obvious commands. A small model judging each action (the idea behind Claude Code's auto mode) teaches how to evaluate a classifier: false-allow vs false-deny rates. | M |
| **Remaining sandbox gaps** | MCP servers run outside the sandbox; only the top-level `.env` is masked; a hard-killed agent leaves its container running. | S each |

## 3. Models

| Item | Why | Size |
|---|---|---|
| **Reasoning / extended thinking** | `models/sse.py` ignores `reasoning` deltas (gpt-oss, qwen3), so reasoning is neither shown nor logged. The Anthropic adapter has no extended thinking; with it on, thinking blocks must be sent back unchanged with `tool_use`, the same class of problem as Gemini's `thought_signature`. | M |
| **Model routing** | A cheap model for summaries, subagents and the evaluator; a strong one for the main loop. Measure cost vs quality with the evals. | S |

## 4. Developer experience

| Item | Why | Size |
|---|---|---|
| **Session resume** (`--resume`) | Reload a conversation from its JSONL transcript; the history validator already makes reloading safe. | S |
| **In-session commands** (`/clear`, `/compact`, `/model`, `/cost`, `/permissions`) | Lets you poke at the context machinery live. | S |
| **Transcript viewer** | A small HTML page that replays a `.agent/*.jsonl` step by step: prompt, tool calls, results, tokens. Debugging agent behavior is most of the job. | M |
| **Show reasoning and token use live** | Builds on the reasoning item above. | S |

## 5. Agent capability

| Item | Why | Size |
|---|---|---|
| **Automatic test/lint loop after edits** (Aider's "reflection") | Feed failures back without the model deciding to check. The hook system and syntax check are already there; A/B it with the evals. | S |
| **Parallel read-only tools** | Run reads and searches from one response concurrently, writes in order; `asyncio` or a thread pool. A good concurrency exercise. | M |
| **Agents in parallel** | Each agent already has its own workspace, sandbox and read record (`ToolContext`), but some state is still module-level and shared: the todo list, `features.STATE`, the evaluator's `VERDICT` and `ask.HANDLER`. Harmless while agents run one after another; move it into the context before running several at once. | S |
| **Persistent memory** | Let the agent append learnings to `AGENTS.md` on request, so the next session starts smarter. | S |
| **Repo map** (Aider) | `ast`/tree-sitter definitions and references, ranked, packed into a token budget. Compare with search-on-demand on `find-bug-in-package`. | M |
| **LSP diagnostics after edits** | Real type/lint errors instead of a syntax check; JSON-RPC again, like MCP. | M |
| **A browser for the evaluator** | Day 11 mentions running web apps; the evaluator can only run CLI programs. | M |
| **The rest of MCP** | Resources, prompts, sampling, elicitation, OAuth for remote servers. | M |
| Web search / fetch, images | Useful, but teach less about how a harness works. | M |

## 6. Evals

| Item | Why | Size |
|---|---|---|
| **LLM-as-judge grading** | For answers a unit test cannot check (data reports, explanations). | S |
| **A small SWE-bench Lite subset** | Real GitHub issues next to the hand-made tasks. | M |
| **Paired statistics** | Confidence intervals, so "did this change help?" has an answer beyond noise. | S |
