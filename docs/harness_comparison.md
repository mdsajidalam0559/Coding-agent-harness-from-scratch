# How our harness compares with Aider and Codex CLI

Studied from source on 2026-09-29: Aider `5dc9490` (Python) and OpenAI Codex CLI `c248f6d` (Rust), cloned in
`~/Work/harness-study/`. File references are relative to those clones.

## The comparison

| | **Ours (day_12_agent)** | **Aider** | **Codex CLI** |
|---|---|---|---|
| **Loop** | Native tool calling. Model calls tools until it replies without one; 25-step cap; retries with backoff; Ctrl-C leaves a valid history. | **No tool calling** in the main path: the model answers in text with edit blocks; the harness parses and applies them, then runs a *reflection* loop: lint and test failures become the next user message, up to 3 times (`coders/base_coder.py` `run_one`, `send_message` L1585-1623). | Native tool calling (Responses API) inside a session/turn engine (`core/src/session/turn.rs`); many tools, sub-agents, plugins. |
| **Tools** | read_file, list_dir, search (ripgrep), write_file, str_replace / apply_edits, bash, todo_write, delegate, load_skill, MCP tools. | No tools for the model. The *user* adds files to the chat (`/add`); the harness adds a **repo map** so the model knows what else exists; the model can ask for more files by naming them (`check_for_file_mentions`). | shell (with an "exec policy" of allowed commands), `apply_patch`, `update_plan`, web search, MCP, image viewing, sub-agents. |
| **Edit format** | `str_replace`: exact, unique match; on failure, diagnostics (tabs vs spaces, CRLF, trailing whitespace, near-matches). Alternative: SEARCH/REPLACE blocks, same strictness. | SEARCH/REPLACE blocks (plus whole-file, unified diff and others per model). Matching: exact, then **uniform leading-whitespace repair**, then skip a spurious blank first line, then `...` elision. Fuzzy edit-distance matching exists but is **switched off** (`editblock_coder.py` `replace_most_similar_chunk`, the bare `return` before it). | Its own `apply_patch` format: `*** Update File:` with `@@` context and `+`/`-`/` ` lines, no line numbers; add/delete/move files (`apply-patch/src/parser.rs`). Matching with decreasing strictness: exact, then ignore trailing whitespace, then ignore both sides, then normalize Unicode punctuation (`apply-patch/src/seek_sequence.rs`). Models are trained on this format. |
| **Verification** | Model is *asked* to run tests (system prompt). Built-in hook: Python syntax check after each edit. Long runs: the harness runs tests and an evaluator agent reviews. | **Harness-driven**: after every edit, lint automatically (`auto_lint=True`) and optionally run the test command (`auto_test`); failures are fed back without the model having to decide to check. | Model-driven, guided by a long "Validating your work" section in the prompt (`protocol/src/prompts/base_instructions/default.md` L149-165): start with the most specific test, widen after. |
| **Permissions / sandbox** | Modes ask / auto-read / auto-edit / auto; deny rules; path confinement; Docker container per session, network off, resource limits; project config needs approval. | Asks the user before running shell commands the model suggests; no sandbox. | Two independent axes: **sandbox mode** (read-only / workspace-write / danger-full-access, network off by default; Linux: bubblewrap + Landlock + seccomp, `linux-sandbox/src/`) and **approval policy** (untrusted / on-request / never / granular, `protocol/src/protocol.rs` `AskForApproval`). In on-request mode the model may ask to run a command *outside* the sandbox, with a reason. |
| **Compaction** | At 70% of the window: summarize the old middle; keep system prompt, first request and the most recent quarter of the window (tool calls kept with their results). | Summarizes chat history in a background thread once it passes a token budget; splits at an assistant message; recursive; the summary is written **in the user's voice** ("I asked you...") (`history.py`, `prompts.py summarize`). | "Checkpoint compaction": a handoff summary for "another LLM", then the new history is **the user's messages verbatim (newest first, up to 20k tokens) + the summary**; assistant turns and tool calls are dropped (`core/src/compact.rs` `build_compacted_history`, `prompts/templates/compact/`). If compaction itself overflows, drop the oldest item and retry. |
| **System prompt** | ~40 lines: workflow (understand, targeted edits, verify), environment, safety; + AGENTS.md, skills list, MCP notes. | Per edit format; very specific about the block syntax, with worked examples; tells the model to ask for files it needs. | ~275 lines: personality, AGENTS.md scoping rules, preamble messages before tool calls, when to plan, task execution, validation, final-answer formatting. |
| **Project memory** | AGENTS.md / CLAUDE.md in the root, capped at 8k characters. | Conventions file you add with `/read`. | AGENTS.md anywhere in the tree, scoped to its directory (`core/src/agents_md.rs`). |
| **Context retrieval** | On demand: search + read_file; nothing up front. | **Repo map** up front: tree-sitter tags, a PageRank over the definition/reference graph personalized toward the files and identifiers in the chat, packed into ~1k tokens (`repomap.py`). | On demand via shell (`rg`, `sed -n`), guided by the prompt. |

## What each does better

- **Aider**: verification the model cannot skip; pragmatic edit matching that repairs the most common model
  error (uniform indentation) without fuzzy guessing; the repo map gives the model the shape of the codebase
  for ~1k tokens.
- **Codex**: a sandbox that is cheap enough to be always on, with a principled escape hatch; a patch format
  co-designed with the models; compaction that guarantees the user's own words survive.
- **Ours**: strictest safety defaults of the three (container, deny rules, project-config trust), a
  harness-verified long-running mode with an independent evaluator, and evals built in from day 6.

## Candidate ideas to port

1. Harness-run verification after edits (Aider's reflection).
2. Tolerant edit matching: uniform indentation (Aider) and trailing whitespace (Codex), still requiring a unique match.
3. A repo map for up-front orientation (Aider).
4. Keep user messages verbatim through compaction (Codex).
5. Sandbox escalation "on request" (Codex).

Which two to port is decided by the baseline eval's failures: see the results section below.

## Baseline results and what was ported (Day 13)

**Baseline:** `day_12_agent`, Groq `qwen/qwen3.8-27b`, sandbox off, results in `evals/results/day12-baseline*.jsonl`.
Groq's free tier cut both runs short, so the evidence is small: 6 of 6 completed trials passed
(add-retry-backoff x2, cli-json-flag x2, find-bug-in-package, fix-failing-tests), 1 trial failed, and
3 tasks never ran.

| What happened | Evidence | Root cause |
|---|---|---|
| **Context overflow** on `implement-from-spec` | HTTP 413: "Limit 7000, Requested 7111" (input tokens per request) | The agent assumed a 131k window, so compaction never fired. The real limit was the provider's 7k, and each request grew 4.3k → 7.1k in 6 steps (whole-file rewrites and long tool output kept verbatim). |
| **Daily quota exhausted** (both runs) | HTTP 429: "tokens per day (TPD): Limit 200000" | About 20-30k tokens per task; a request costs about 2k tokens before any work (system prompt + tools). |
| 1 of 6 passes never ran anything after its last edit | `python -m evals.failures ... --all` | The model skipped verification and passed by luck. |

Both failures are the same problem: **tokens per task** measured against a provider limit the agent did
not know about. So both ports target that:

1. **Context-overflow recovery (Codex, `core/src/compact.rs`):** on "request too large", read the real
   limit from the error, use it as the working window, compact (or blank old tool output when there is
   nothing to summarize, like Codex dropping its oldest items), and retry. At most 3 recoveries per step.
2. **Window-aware tool-output truncation (Codex, `core/src/context_manager/history.rs`, 10 KB default,
   `tool_output_token_limit` per model):** every tool result is capped at min(10k characters, 15% of the
   working window) as it enters the history. Before: a flat 30k characters.

Also added: **`ask_user`** (from Claude Code): a multiple-choice question in the middle of a task.

Not ported, and why: harness-run verification (Aider) does not help here, since the eval tasks' tests are
hidden and only one task has visible tests; the repo map (Aider) targets a problem the data did not show
(no search-heavy failures); keeping user messages through compaction (Codex) matters only once compaction
happens, which it never did in the baseline.

**Pending: the before/after comparison.** Both Groq models used today hit their daily quota (200k
tokens/day each). Once it refills (a rolling 24-hour window), run the same tasks on the same model:

    python -m evals.runner --agent day_12_agent --provider groq --model qwen/qwen3.8-27b --trials 2 --sandbox off --label d12
    python -m evals.runner --agent day_13_agent --provider groq --model qwen/qwen3.8-27b --trials 2 --sandbox off --label d13
    python -m evals.report evals/results/d12.jsonl evals/results/d13.jsonl

Expected: 0 context overflows for day 13 (vs 1 in 7 trials), fewer prompt tokens per task, and the
same pass rate on tasks that fit. A full run of both agents needs more than one model's daily quota, so
split it over two days, or use `--tasks` to run the long tasks (implement-from-spec, find-bug-in-package,
add-retry-backoff) first.
