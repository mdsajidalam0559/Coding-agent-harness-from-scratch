"""Server-Sent Events parsing for OpenAI-compatible streaming chat completions.

A stream is plain text: events separated by blank lines, each event made of `data: ...` lines.
Lines starting with ':' are comments (OpenRouter sends ': OPENROUTER PROCESSING' keep-alives).
The stream ends with `data: [DONE]`.
"""
import json


class StreamError(Exception):
    """The server reported an error in the middle of the stream."""

    def __init__(self, error):
        super().__init__(error.get("message", str(error)))
        self.error = error
        self.code = error.get("code")


def iter_events(lines):
    """Yield the JSON payload of each event from an iterable of text lines."""
    data = []
    for line in lines:
        if line is None:
            continue
        line = line.rstrip("\r")
        if line == "":  # blank line = end of one event
            if data:
                payload = "\n".join(data)
                data = []
                if payload == "[DONE]":
                    return
                yield json.loads(payload)
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data and "\n".join(data) != "[DONE]":  # stream ended without a final blank line
        yield json.loads("\n".join(data))


def _slot(calls, piece):
    """Which tool call a streamed fragment belongs to.

    OpenAI-style streams number calls with `index` and send later fragments without an id.
    Gemini sends each call whole, with an id but no index, so a new id means a new call.
    """
    if "index" in piece:
        return piece["index"]
    if piece.get("id"):
        for index, call in calls.items():
            if call["id"] == piece["id"]:
                return index
        return max(calls, default=-1) + 1
    return max(calls, default=0)  # no index and no id: continues the latest call


def assemble(events, on_text=None):
    """Rebuild a normal (non-streaming) response body from streamed chunks.

    Text arrives in pieces in `delta.content`; tool calls arrive in pieces too, keyed by `index`,
    with the id and name in the first piece and the JSON arguments spread over many.
    """
    text, calls, finish_reason, usage = [], {}, None, None
    for chunk in events:
        if "error" in chunk:
            raise StreamError(chunk["error"])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text.append(delta["content"])
                if on_text:
                    on_text(delta["content"])
            for piece in delta.get("tool_calls") or []:
                call = calls.setdefault(_slot(calls, piece),
                                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if piece.get("id"):
                    call["id"] = piece["id"]
                function = piece.get("function") or {}
                call["function"]["name"] += function.get("name") or ""
                call["function"]["arguments"] += function.get("arguments") or ""
                # provider-specific fields (e.g. Gemini's extra_content.google.thought_signature) must be
                # sent back unchanged with the next request, so keep them
                for key, value in piece.items():
                    if key not in ("index", "id", "type", "function"):
                        call[key] = value
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    message = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {"choices": [{"finish_reason": finish_reason, "message": message}], "usage": usage or {}}
