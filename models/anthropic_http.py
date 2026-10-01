"""Native adapter for Anthropic's Messages API (raw HTTP, streaming, prompt caching).

Canonical (OpenAI-style) conversation -> Messages API:
- system messages become the top-level `system` parameter;
- an assistant message's tool_calls become `tool_use` content blocks;
- `tool` messages become `tool_result` blocks inside a user message;
- consecutive messages with the same role are merged (the API requires user/assistant alternation).
Streaming events: message_start, content_block_start/delta/stop (text_delta, input_json_delta),
message_delta (stop_reason, output usage), message_stop, ping, error.
"""
import json
import os

import requests

from models.base import RETRYABLE_STATUS, ModelAdapter, ModelError, ModelResponse, with_retries
from models.sse import iter_events

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
STOP_REASONS = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}
CACHE = {"type": "ephemeral"}


def to_anthropic(messages, tools):
    """(system_blocks, messages, tools) in Messages API shape, with cache breakpoints."""
    system = [{"type": "text", "text": m["content"]} for m in messages if m["role"] == "system" and m.get("content")]
    converted = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "tool":
            blocks, role = [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content") or ""}], "user"
        elif role == "assistant":
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for call in m.get("tool_calls") or []:
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {"_unparseable_arguments": call["function"]["arguments"]}
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": arguments})
            if not blocks:
                blocks = [{"type": "text", "text": "(no content)"}]
        else:
            content = m.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] += blocks
        else:
            converted.append({"role": role, "content": blocks})
    # tool results must come first in their user message
    for m in converted:
        if m["role"] == "user":
            m["content"].sort(key=lambda b: b.get("type") != "tool_result")
    if system:
        system[-1]["cache_control"] = CACHE
    if converted and converted[-1]["content"]:
        converted[-1]["content"][-1] = {**converted[-1]["content"][-1], "cache_control": CACHE}
    anthropic_tools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                        "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                       for t in tools or []]
    return system, converted, anthropic_tools


def assemble_stream(events, on_text=None):
    """ModelResponse from Anthropic stream events."""
    text, blocks, usage, stop = [], {}, {}, None
    for event in events:
        kind = event.get("type")
        if kind == "error":
            error = event.get("error") or {}
            overloaded = error.get("type") in ("overloaded_error", "api_error", "rate_limit_error")
            raise ModelError(f"stream error: {error.get('message')}", status=529 if overloaded else None,
                             retryable=overloaded, body={"error": error})
        if kind == "message_start":
            usage.update((event.get("message") or {}).get("usage") or {})
        elif kind == "content_block_start":
            block = dict(event.get("content_block") or {})
            if block.get("type") == "tool_use":
                block["partial_json"] = ""
            blocks[event["index"]] = block
        elif kind == "content_block_delta":
            delta, block = event.get("delta") or {}, blocks.get(event["index"], {})
            if delta.get("type") == "text_delta":
                text.append(delta["text"])
                if on_text:
                    on_text(delta["text"])
            elif delta.get("type") == "input_json_delta":
                block["partial_json"] = block.get("partial_json", "") + delta.get("partial_json", "")
        elif kind == "message_delta":
            stop = (event.get("delta") or {}).get("stop_reason") or stop
            usage.update(event.get("usage") or {})
    tool_calls = []
    for index in sorted(blocks):
        block = blocks[index]
        if block.get("type") == "tool_use":
            raw = block.get("partial_json") or json.dumps(block.get("input") or {})
            tool_calls.append({"id": block["id"], "type": "function",
                               "function": {"name": block["name"], "arguments": raw or "{}"}})
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    canonical_usage = {"prompt_tokens": (usage.get("input_tokens") or 0) + cache_read + cache_write,
                       "completion_tokens": usage.get("output_tokens") or 0,
                       "prompt_tokens_details": {"cached_tokens": cache_read}, "cache_write_tokens": cache_write}
    return ModelResponse(text="".join(text), tool_calls=tool_calls, usage=canonical_usage,
                         finish_reason=STOP_REASONS.get(stop, stop))


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model, key_env="ANTHROPIC_API_KEY", context_window=200_000, name=None, url=API_URL, log=None):
        super().__init__(log)
        self.model, self.key_env, self.context_window, self.url = model, key_env, context_window, url
        self.name = name or model

    def complete(self, messages, tools, on_text=None, max_tokens=4000):
        key = (os.getenv(self.key_env) or "").strip()
        if not key:
            raise ModelError(f"{self.name} needs {self.key_env} in .env", retryable=False)
        system, converted, anthropic_tools = to_anthropic(messages, tools)
        payload = {"model": self.model, "max_tokens": max_tokens, "messages": converted, "stream": True}
        if system:
            payload["system"] = system
        if anthropic_tools:
            payload["tools"] = anthropic_tools
        headers = {"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"}
        return with_retries(self, lambda: self._attempt(payload, headers, on_text))

    def _attempt(self, payload, headers, on_text):
        try:
            with requests.post(self.url, json=payload, headers=headers, stream=True, timeout=(10, 120)) as response:
                if response.status_code == 200:
                    response.encoding = "utf-8"
                    return assemble_stream(iter_events(response.iter_lines(decode_unicode=True)), on_text)
                try:
                    error = response.json().get("error") or {}
                except ValueError:
                    error = {"message": response.text[:500]}
                raise ModelError(f"HTTP {response.status_code}: {error.get('message', '')}", status=response.status_code,
                                 retryable=response.status_code in RETRYABLE_STATUS,
                                 body={"error": error, "retry_after": response.headers.get("retry-after")})
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            raise ModelError(f"network error: {e!r}", retryable=True, body={"error": repr(e)}) from None
