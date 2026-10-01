"""Adapter for OpenAI-compatible chat completions: OpenRouter, Groq, Gemini's compatibility endpoint,
OpenAI itself, and local servers (Ollama, vLLM, llama.cpp). Streams, and retries transient failures."""
import os

import requests

from models.base import RETRYABLE_STATUS, ModelAdapter, ModelError, ModelResponse, with_retries
from models.sse import StreamError, assemble, iter_events


def retry_delay_hint(error):
    """Seconds to wait when the error body says so (Gemini: details[].RetryInfo.retryDelay = "27s")."""
    if not isinstance(error, dict):
        return None
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and detail.get("@type", "").endswith("RetryInfo"):
            return (detail.get("retryDelay") or "").rstrip("s") or None
    return None


class OpenAICompatAdapter(ModelAdapter):
    def __init__(self, url, model, key_env=None, context_window=128_000, name=None, headers=None, log=None):
        super().__init__(log)
        self.url, self.model, self.key_env = url.rstrip("/"), model, key_env
        self.context_window = context_window
        self.name = name or model
        self.headers = headers or {}

    def _key(self):
        if not self.key_env:
            return None  # local servers
        key = (os.getenv(self.key_env) or "").strip()
        if not key:
            raise ModelError(f"{self.name} needs {self.key_env} in .env", status=None, retryable=False)
        return key

    def complete(self, messages, tools, on_text=None, max_tokens=4000):
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                   "stream_options": {"include_usage": True}}
        if tools:
            payload["tools"] = tools
        key = self._key()
        headers = {**self.headers, **({"Authorization": f"Bearer {key}"} if key else {})}
        return with_retries(self, lambda: self._attempt(payload, headers, on_text))

    def _attempt(self, payload, headers, on_text):
        try:
            with requests.post(f"{self.url}/chat/completions", json=payload, headers=headers, stream=True,
                               timeout=(10, 120)) as response:
                if response.status_code == 200:
                    response.encoding = "utf-8"
                    body = assemble(iter_events(response.iter_lines(decode_unicode=True)), on_text)
                    return self._to_response(body)
                try:
                    body = response.json()
                except ValueError:
                    body = {"error": {"message": response.text[:500]}}
                if isinstance(body, list):  # Gemini wraps errors in a list
                    body = body[0] if body and isinstance(body[0], dict) else {}
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict) and error.get("code") == "tool_use_failed":
                    # The provider rejected the model's own malformed tool call (e.g. Groq validating JSON
                    # arguments). That is a model mistake, not an outage: send it back so the model can fix it.
                    self.log("format_error", {"provider_error": error.get("message"),
                                              "failed_generation": str(error.get("failed_generation", ""))[:2000]})
                    return ModelResponse(text="", format_error=(
                        "Your last tool call was rejected because its arguments were not valid JSON. What you sent "
                        f"began with:\n{str(error.get('failed_generation', ''))[:600]}\n"
                        "Call the tool again with a single valid JSON object as arguments (escape quotes and newlines "
                        "inside strings). For multi-line edits, prefer str_replace or write_file over shell heredocs."))
                raise ModelError(f"HTTP {response.status_code}: {(error or {}).get('message', '') if isinstance(error, dict) else error}",
                                 status=response.status_code, retryable=response.status_code in RETRYABLE_STATUS,
                                 body={"error": error, "retry_after": response.headers.get("Retry-After")
                                       or retry_delay_hint(error)})
        except StreamError as e:  # an error sent in the middle of a stream
            code = e.code if isinstance(e.code, int) else None
            raise ModelError(f"stream error: {e}", status=code, retryable=code is None or code in RETRYABLE_STATUS,
                             body={"error": e.error, "retry_after": retry_delay_hint(e.error)}) from None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            raise ModelError(f"network error: {e!r}", status=None, retryable=True, body={"error": repr(e)}) from None

    @staticmethod
    def _to_response(body):
        choice = body["choices"][0]
        message = choice["message"]
        extra = {}
        return ModelResponse(text=message.get("content") or "", tool_calls=message.get("tool_calls") or [],
                             usage=body.get("usage") or {}, finish_reason=choice.get("finish_reason"), extra=extra)
