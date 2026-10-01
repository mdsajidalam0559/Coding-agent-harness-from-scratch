"""The context window is a budget: measure what each turn uses and keep tool output from blowing it."""
import json
import os
import re

DEFAULT_CONTEXT_WINDOW = 128_000
CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": 16_385,
    "openai/gpt-oss-120b": 131_072,
    "openai/gpt-oss-20b": 131_072,
    "qwen/qwen3.8-27b": 131_072,
    "anthropic/claude-haiku-4.5": 200_000,
}
MAX_TOOL_RESULT_CHARS = 30_000  # ~8k tokens; one tool result should never take over the context
CHARS_PER_TOKEN = 3.5  # rough; code and JSON tokenize worse than English prose


def context_window(model):
    """Tokens the model can take. AGENT_CONTEXT_WINDOW overrides (useful to force compaction in tests)."""
    if os.getenv("AGENT_CONTEXT_WINDOW"):
        return int(os.getenv("AGENT_CONTEXT_WINDOW"))
    if model in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[model]
    if model.startswith("gemini"):
        return 1_048_576
    return DEFAULT_CONTEXT_WINDOW


def estimate_tokens(obj):
    """Cheap estimate for anything JSON-serializable. Use real usage numbers from the API when you have them."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return int(len(text) / CHARS_PER_TOKEN) + 1


def clip_tool_result(text, limit=MAX_TOOL_RESULT_CHARS):
    """Keep the start and the end of an oversized tool result, and tell the model how to get the rest."""
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.4), int(limit * 0.6)
    return (f"{text[:head]}\n\n[... {len(text) - head - tail} characters omitted from the middle of this result "
            f"to save context. Use a narrower command (grep, head, tail, read_file with offset/limit) "
            f"to see the part you need ...]\n\n{text[-tail:]}")


def cached_tokens(usage):
    """Prompt tokens served from the provider's cache (field names differ between APIs)."""
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0


def _k(n):
    return f"{n / 1000:.1f}k"


class TokenBudget:
    """Running token/cost totals for the current turn and the whole session."""

    def __init__(self, window):
        self.window = window
        self.context_tokens = 0  # size of the conversation as of the last API call
        self.session = self._zero()
        self.turn = self._zero()

    @staticmethod
    def _zero():
        return {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "cost": 0.0}

    def start_turn(self):
        self.turn = self._zero()

    def record(self, usage):
        usage = usage or {}
        numbers = {"calls": 1, "prompt": usage.get("prompt_tokens") or 0, "cached": cached_tokens(usage),
                   "completion": usage.get("completion_tokens") or 0, "cost": usage.get("cost") or 0.0}
        for totals in (self.turn, self.session):
            for key, value in numbers.items():
                totals[key] += value
        if numbers["prompt"]:
            self.context_tokens = numbers["prompt"] + numbers["completion"]
        return numbers

    def status(self):
        t = self.turn
        pct = 100 * self.context_tokens / self.window if self.window else 0
        return (f"context {_k(self.context_tokens)}/{_k(self.window)} ({pct:.0f}%) · "
                f"this turn: {t['calls']} calls, {_k(t['prompt'])} in ({_k(t['cached'])} cached), "
                f"{_k(t['completion'])} out, ${t['cost']:.4f} · session ${self.session['cost']:.4f}")


OVERFLOW_PHRASES = ("request too large", "context_length_exceeded", "context length", "maximum context",
                    "too many tokens", "prompt is too long")
LIMIT_PATTERNS = (r"Limit (\d+), Requested \d+", r"maximum context length is (\d+)", r"context window of (\d+)",
                  r"prompt is too long: \d+ tokens > (\d+)")


def context_overflow(error_event):
    """(is_overflow, limit_or_None) for a logged api_error: did the request not fit the model/provider?"""
    text = json.dumps((error_event or {}).get("error") or "")
    overflow = (error_event or {}).get("status") == 413 or any(p in text.lower() for p in OVERFLOW_PHRASES)
    if not overflow:
        return False, None
    for pattern in LIMIT_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return True, int(match.group(1))
    return True, None
