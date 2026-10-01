# A coding agent harness, built from scratch in two weeks

An agent harness in the shape of Claude Code or Codex CLI, written without agent SDKs or frameworks:
raw HTTP to the model, hand-written tools, sandbox, context management, MCP client and evals. One
model-agnostic core runs two different agents (a coding agent and a data-analysis agent) on any
provider. Every part was built first and measured, then compared with production harnesses
(Aider and Codex CLI, see [docs/harness_comparison.md](docs/harness_comparison.md)).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
# .env: at least one of GROQAPI_KEY, OPENROUTER_KEY, GEMINIAPI_KEY, ANTHROPIC_API_KEY
python -m ui.tui                                          # interactive coding agent (Docker sandbox)
python -m ui.tui --agent data --model groq:qwen/qwen3.8-27b
python -m ui.headless "Fix the failing tests" --workspace ./repo --json    # scripts, CI, cron
python -m longrun.session --dir ./app --goal "..."        # build a project across many sessions
python -m evals.runner --agent core:coding --model groq:openai/gpt-oss-120b --trials 3
python -m tests.run_all                                   # all 14 test suites (no API calls)
```

Models are named `provider:model` (`groq:qwen/qwen3.8-27b`, `anthropic:claude-haiku-4-5`,
`ollama:qwen3:8b`), with `+text` to use the text protocol instead of native tool calling. Aliases go in
`models.json` or `~/.agent/models.json`. The Docker sandbox needs your user in the `docker` group.

## Layout

| Layer | Where | Built on |
|---|---|---|
| Interfaces | `ui/tui.py` (interactive), `ui/headless.py` (scripts) | Days 7, 14 |
| Agent core | `core/agent.py`: the loop, retries, streaming, interrupts, subagents | Days 2, 7, 9, 14 |
| Agents | `agents/coding.py`, `agents/data.py`: a tool set + a system prompt each | Day 14 |
| Model layer | `models/`: `openai_compat.py`, `anthropic_http.py`, `text_protocol.py`, `registry.py`, `sse.py` | Days 1-2, 7, 14 |
| Tools | `tools/`: files, edit, bash, search, todo, ask_user, data; `tools/registry.py` | Days 3-4, 10, 13, 14 |
| Context | `context/`: token budget, system prompt + AGENTS.md, compaction + history validator | Days 8-9, 13 |
| Safety | `safety/permissions.py`, `safety/sandbox.py`, `sandbox/Dockerfile`, `ext/trust.py` | Days 5, 12 |
| Long-running | `longrun/`: feature list, multi-session harness, evaluator agent | Days 10-11 |
| Extensibility | `ext/`: MCP client (from the spec), MCP tools bridge, hooks, skills | Day 12 |
| Evals | `evals/`: tasks (coding + data), runner, report, failure analysis, final eval | Days 6, 13, 14 |
| History | `days/`: each day's agent as a snapshot, and [days/README.md](days/README.md), an index of every day's code | every day |
| Tests | `tests/test_dayN.py` (one suite per day), `tests/fixtures/` (fake MCP servers); `python -m tests.run_all` | every day |

## Design choices, and why

### The loop
The model is called with the conversation and the tool list; if it asks for tools, the harness runs
them and sends the results back; repeat until it answers without a tool call. Everything else is
policy around this loop. Choices:
- **A step limit (25) per user message**, counting model calls, not user turns. The first version
  counted user messages, so a model that kept calling tools could loop forever.
- **Every tool call gets a result, always.** The API rejects a history with an unanswered tool call.
  Interrupts (Ctrl-C), crashes and compaction all have to respect this, so a validator checks the
  history before every request and repairs it if needed (Days 7, 9).
- **Retries with exponential backoff and jitter**, honoring `Retry-After` and Gemini's `RetryInfo`,
  only for errors that can succeed later (429, 5xx, timeouts); the HTTP status decides, not the body
  (Groq's body code is the string `rate_limit_exceeded`).

### A model-agnostic core
One canonical message shape (OpenAI-style chat messages) everywhere; adapters translate at the edge.
The core never knows which provider it talks to. Three adapters cover almost every model:
- **OpenAI-compatible** (OpenRouter, Groq, Gemini's compatibility endpoint, OpenAI, Ollama, vLLM).
  Provider quirks found by testing live: Gemini sends parallel tool calls without an `index` and needs
  its `thought_signature` sent back; Gemini wraps errors in a list; Groq validates tool-call JSON
  server-side and returns `tool_use_failed`, which is the model's mistake, so it is sent back to the
  model as a format error instead of ending the run.
- **Native Anthropic Messages API** (`tool_use` / `tool_result` blocks, strict role alternation, its
  own event stream, explicit cache breakpoints). Tested offline against recorded streams only: no
  Anthropic key was available.
- **Text protocol** for models without reliable tool calling: tools are described in the prompt and
  called as fenced JSON blocks (or a `bash` block), parsed back into tool calls. The core cannot tell
  the difference, so everything else (permissions, compaction, evals) works unchanged.

### Tools
- **A registry with the schema next to the function** (`@tool(...)`), so adding a tool is one
  function. Arguments are validated against the schema before the call, with a message listing the
  valid parameters: models invent parameters (`line_start`/`line_end`, `depth`), and a clear error
  lets them recover in one step.
- **`read_file` numbers lines and records what was read** (a content hash). Edits and overwrites
  are refused for files the agent has not read, or that changed on disk since (e.g. via bash).
- **`str_replace` requires exactly one exact match.** On failure it reports why: zero or several
  matches (with line numbers), tabs vs spaces, `\r\n`, trailing whitespace. Aider and Codex both
  repair some whitespace mismatches automatically; I kept exact matching plus diagnostics and measured
  instead (see the comparison doc).
- **`bash` can never hang**: stdin closed, no terminal, pagers/editors/git prompts disabled, its own
  process group killed on timeout or Ctrl-C, ANSI codes stripped, output truncated in the middle
  (the end matters most). `PYTHONDONTWRITEBYTECODE=1`, because an agent that edits `a + b` into
  `a * b` within the same second as the last run otherwise gets Python's stale compiled code (found
  by the Day 10 tests).
- **Tool output entering the history is capped by the window**: min(10k characters, 15% of the
  working window), as Codex does. A flat 30k cap let a 7k-token provider limit fill up in six steps.

### Safety
- **Two layers: permissions decide what may be tried; the sandbox limits what can happen.** A deny
  list catches obvious mistakes but is trivially bypassed (`python -c "shutil.rmtree(...)"`), so
  commands also run in a Docker container: only the project mounted, `.env` masked, no network by
  default, memory/CPU/process limits, no capabilities, your user id.
- **Permission modes** ask / auto-read / auto-edit / auto; file tools are confined to the workspace
  (symlinks and `..` resolved). Unattended runs (evals, headless, long-running) have nobody to ask,
  so anything needing approval is denied.
- **Config that ships with a repo is untrusted.** A cloned repo's `.agent/mcp.json` or `hooks.json`
  could run anything, so it is used only after you approve it (remembered per content hash).
- **Secrets never leave**: MCP servers get a minimal environment (verified with `server-everything`'s
  `get-env`), the sandbox masks `.env`, and prompts tell the model that file contents are data, not
  instructions. The Day 5 checkpoint plays a model that obeys a malicious README; the attacks fail.

### Evals
- **Hidden graders**: each task has a starting repo, a prompt, a grading test copied in only after
  the agent finishes, and a reference solution. `evals/validate.py` checks the unsolved repo fails and
  the solution passes (it caught a wrong grader on day one).
- **pass@k and pass^k**, tokens, steps, cost; spending cap; `evals/failures.py` shows *why* trials
  failed (edit errors, no verification after the last edit, repeated calls).
- **Infrastructure failures are not scored as model failures** (quota, outage, bad key): the run
  stops. A context overflow *is* the agent's failure and is scored as such.
- Measuring changed the design twice: the Day 13 ports were chosen from baseline failures, and the
  free tier's quotas (7k tokens per request, 200k per day) made tokens-per-task the metric that matters.

### Streaming and interrupts
The stream is parsed by hand (SSE: `data:` lines, comments, multi-line events, `[DONE]`) and rebuilt
into the same response shape. Ctrl-C while streaming keeps the partial text marked as interrupted;
Ctrl-C during tools gives every unfinished call an "interrupted" result and kills its processes
(inside Docker too: killing `docker exec` does not stop the command, so the container's processes are
killed explicitly).

### Context
- **A stable system prompt** (no timestamps) so providers can cache the prefix; project notes from
  `AGENTS.md` are appended; token usage (including cached tokens) is shown after every turn.
- **Compaction at 70% of the window**: the old middle of the conversation is summarized; the system
  prompt, the original request and the most recent quarter are kept verbatim. A tool call and its
  results are one unit and are never split (tested on 300 random histories). It skips compacting when
  only an earlier summary is left, which otherwise re-summarizes a summary on every step.
- **Overflow recovery** (from Codex): when the provider says a request is too large, read the real
  limit from the error, adopt it, compact or blank old tool output, retry. The baseline's only real
  failure was exactly this: the agent assumed 131k tokens, the provider allowed 7k.
- **Subagents** (`delegate`) run a subtask in a fresh context and return only their report.
- **Retrieval on demand**: nothing is loaded up front; the agent searches and reads what it needs.
  (Aider's repo map is the main alternative; the evals did not show search-related failures.)

### Long-running work
Continuity lives in files, not chat memory: `feature_list.json`, `progress.md`, git history. The
**harness**, not the model, picks the next feature, runs the tests before and after each session,
decides the real status (a model's "done" is only a claim), writes the progress note and commits. An
**independent evaluator agent** (fresh context, skeptical prompt, no edit tools, changes reverted)
reviews every "done" claim and sends reproduced problems back. Live, it caught five real bugs in code
the generator called finished, two of which had not been planted. Provider quota failures are not
counted as attempts (they had falsely consumed one).

### Extensibility
- **MCP client written from the spec** (JSON-RPC over stdio and Streamable HTTP; initialize,
  tools/list with pagination, tools/call, cancellation, server-to-client ping and roots). Tested
  against a deliberately awkward fake server and against real third-party servers (`mcp-server-git`,
  `server-everything`). Third-party tools count as "exec" unless marked read-only.
- **Hooks** before/after every tool call: Python callbacks or shell commands (exit 2 blocks), plus a
  built-in syntax check after edits to `.py` files.
- **Skills**: `SKILL.md` folders; only names and descriptions are in the prompt, `load_skill` brings
  in the rest when a task matches.
- **`ask_user`** (from Claude Code): a multiple-choice question mid-task; the answer returns as the
  tool result. Unattended runs are told to choose and say so. Live, `gpt-oss-20b` preferred to decide
  on its own even for ambiguous requests.

### Two agents on one core
An agent is an `AgentConfig`: a tool list and a system prompt. The **data-analysis agent** answers
questions about CSV files with read-only SQL (every CSV becomes an SQLite table; types inferred;
`PRAGMA query_only`), and writes Markdown reports. It shares everything else with the coding agent:
loop, permissions, budget, compaction, subagents, streaming, headless mode, evals (a 3-task data
suite). It has no command tools, so it needs no sandbox.

### Headless mode
`python -m ui.headless`: the task from an argument, a file or stdin; the answer (or JSON with status,
steps, usage, changed files, transcript path) on stdout, progress on stderr; exit codes 0 done,
1 stopped, 2 usage error, 130 interrupted. It refuses to run commands unattended outside the sandbox
unless told explicitly.

## Two weeks of progress

See [docs/progress.md](docs/progress.md) for the score table (Day 6 baseline vs final vs
mini-SWE-agent) and what the numbers do and do not show.

## Known limitations

- The Docker sandbox and the Day 5 prompt-injection checkpoint have not run on this machine yet
  (the user is not in the `docker` group). All evals so far ran with `--sandbox off`.
- The Anthropic adapter is tested offline only.
- MCP servers run on the host, outside the sandbox; the client supports tools only (no resources,
  prompts, sampling or elicitation), and no OAuth for remote servers.
- The free-tier quotas limit how many eval trials fit in a day, so the score table is small; rerun
  `python -m evals.final_eval` with more quota for tighter numbers.
- No git history for the harness itself yet; the `days/` files are the snapshots (`.gitignore` is ready).
