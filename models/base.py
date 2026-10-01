"""The model layer's contract. Nothing outside models/ knows which provider or API it talks to.

The conversation is kept in one canonical shape everywhere (OpenAI-style chat messages):
    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [{"id", "type": "function", "function": {"name", "arguments": json-string}}],  # assistant
     "tool_call_id": str}                                                                           # tool
Tools are OpenAI-style function schemas. Each adapter translates to and from its own API.
"""
import random
import time
from dataclasses import dataclass, field

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)   # canonical tool_calls (see module docstring)
    usage: dict = field(default_factory=dict)         # prompt_tokens, completion_tokens, cached tokens, cost
    finish_reason: str | None = None
    format_error: str | None = None                   # text protocol: the reply could not be parsed
    extra: dict = field(default_factory=dict)         # provider fields to keep on the assistant message

    def as_message(self):
        """The assistant message to append to the conversation."""
        message = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        message.update(self.extra)
        return message


class ModelError(Exception):
    """A request that failed for good (after retries)."""

    def __init__(self, message, status=None, retryable=False, body=None):
        super().__init__(message)
        self.status, self.retryable, self.body = status, retryable, body


class ModelAdapter:
    """What the core needs from a model."""
    name = "model"
    model = ""
    context_window = 128_000
    native_tools = True
    max_retries = 4

    def __init__(self, log=None):
        self.log = log or (lambda event, data: None)  # set by the agent to its transcript logger

    def complete(self, messages, tools, on_text=None, max_tokens=4000):
        """One model call. messages/tools in canonical shape. Raises ModelError on final failure."""
        raise NotImplementedError


def backoff_delay(attempt, retry_after=None, base=1.0, cap=30.0, server_cap=120.0):
    """Exponential backoff with full jitter; the server's requested delay wins when it sends one."""
    if retry_after is not None:
        try:
            return min(float(retry_after) + 1, server_cap)
        except (TypeError, ValueError):
            pass
    return random.uniform(0, min(cap, base * 2 ** attempt))


def with_retries(adapter, attempt_once):
    """Run attempt_once() -> ModelResponse, retrying ModelErrors marked retryable.

    attempt_once raises ModelError(retryable=..., body={"retry_after": seconds or None, ...}).
    Every failure is logged as api_error (the transcript format the evals read).
    """
    for attempt in range(adapter.max_retries + 1):
        try:
            return attempt_once()
        except ModelError as e:
            final = not e.retryable or attempt == adapter.max_retries
            error = (e.body or {}).get("error", str(e))
            adapter.log("api_error", {"attempt": attempt, "status": e.status, "error": error,
                                      "retryable": e.retryable, "final": final})
            if final:
                raise
            time.sleep(backoff_delay(attempt, (e.body or {}).get("retry_after")))
