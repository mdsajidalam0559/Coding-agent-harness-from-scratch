"""Text-protocol adapter: tool use for any instruction-following model, no tool-calling API needed.

The model is told to write tool calls as fenced blocks, which this adapter parses into canonical
tool_calls, so the core loop cannot tell the difference:

    ```tool
    {"name": "read_file", "arguments": {"path": "app.py"}}
    ```
    ```bash
    python -m pytest -q          <- shorthand for the bash tool (mini-SWE-agent style)
    ```

Going the other way, earlier tool calls stay as the text the model wrote, and tool results are sent
back as user messages. A reply without blocks is the final answer; an unparseable block becomes a
format error that the core sends back to the model.
"""
import itertools
import json
import re

from models.base import ModelAdapter

BLOCK_RE = re.compile(r"```(tool|bash|sh)[ \t]*\n(.*?)```", re.S)

INSTRUCTIONS = """

# How to use tools
You call tools by writing fenced blocks in your reply; the results come back in the next message.
For any tool:
```tool
{{"name": "<tool name>", "arguments": {{<arguments as JSON>}}}}
```
For shell commands you may use the shorthand:
```bash
<command>
```
You can put several blocks in one reply; they run in order. Write NO block when you are finished: that reply is your final answer.

# Available tools
{tools}"""


def describe_tools(tools):
    lines = []
    for t in tools:
        f = t["function"]
        params = f.get("parameters") or {}
        required = set(params.get("required") or [])
        args = ", ".join(f"{name}{'' if name in required else '?'}: {spec.get('type', 'any')}"
                         for name, spec in (params.get("properties") or {}).items())
        lines.append(f"- {f['name']}({args}): {f.get('description', '')}")
    return "\n".join(lines)


class TextProtocolAdapter(ModelAdapter):
    native_tools = False

    def __init__(self, inner):
        super().__init__(inner.log)
        self.inner = inner
        self.name, self.model, self.context_window = f"{inner.name}+text", inner.model, inner.context_window
        self._ids = itertools.count(1)

    @property
    def log(self):
        return self.inner.log

    @log.setter
    def log(self, value):
        if hasattr(self, "inner"):
            self.inner.log = value

    def translate(self, messages, tools):
        """Canonical conversation -> plain chat with the protocol in the system prompt."""
        out = []
        for m in messages:
            role = m["role"]
            if role == "system":
                content = m["content"] + (INSTRUCTIONS.format(tools=describe_tools(tools)) if tools else "")
                out.append({"role": "system", "content": content})
            elif role == "tool":
                result = f"[result of call {m['tool_call_id']}]\n{m.get('content') or ''}"
                if out and out[-1]["role"] == "user" and out[-1]["content"].startswith("[result of call"):
                    out[-1]["content"] += "\n\n" + result
                else:
                    out.append({"role": "user", "content": result})
            elif role == "assistant":
                out.append({"role": "assistant", "content": m.get("content") or "(no text)"})
            else:
                out.append({"role": role, "content": m.get("content") or ""})
        if tools and not any(m["role"] == "system" for m in out):
            out.insert(0, {"role": "system", "content": INSTRUCTIONS.format(tools=describe_tools(tools)).strip()})
        return out

    def parse(self, text):
        """(tool_calls, format_error) from a reply."""
        calls = []
        for n, (kind, body) in enumerate(BLOCK_RE.findall(text or "")):
            call_id = f"txt{next(self._ids)}"
            if kind in ("bash", "sh"):
                calls.append({"id": call_id, "type": "function",
                              "function": {"name": "bash", "arguments": json.dumps({"command": body.strip()})}})
                continue
            try:
                data = json.loads(body)
                name, arguments = data["name"], data.get("arguments") or {}
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise ValueError("name must be a string and arguments an object")
            except (ValueError, KeyError, TypeError) as e:
                return [], (f"Format error in tool block {n + 1}: {e}. Write exactly one JSON object with \"name\" "
                            f"and \"arguments\", e.g. {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"a.py\"}}}}.")
            calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}})
        return calls, None

    def complete(self, messages, tools, on_text=None, max_tokens=4000):
        response = self.inner.complete(self.translate(messages, tools), tools=None, on_text=on_text,
                                       max_tokens=max_tokens)
        response.tool_calls, response.format_error = self.parse(response.text) if tools else ([], None)
        if response.tool_calls:
            response.finish_reason = "tool_calls"
        return response
