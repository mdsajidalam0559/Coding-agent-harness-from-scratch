"""Day 7: streaming, safe interrupts, providers, and two agent loops.

- Tool protocol (default): native tool calling over a hand-parsed SSE stream.
- Text protocol (--text-protocol, or AGENT_PROTOCOL=text): no tool API at all, mini-SWE-agent style. The
  model writes exactly one ```bash block per reply; the harness runs it and sends the output back.

    python -m days.day_7_agent --provider groq
    python -m days.day_7_agent --provider groq --text-protocol
"""
import argparse
import re
import requests
import json
import time
import random
from datetime import datetime
import os
from dotenv import load_dotenv

from tools import tool_schemas, execute_tool
import tools.shell
from safety.permissions import Policy
from safety.sandbox import DockerSandbox
from models.sse import StreamError, assemble, iter_events

load_dotenv()

# OpenAI-compatible chat APIs. The key is read from the named variable in .env; it is only ever sent
# to that provider's URL.
PROVIDERS = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "key_env": "OPENROUTER_KEY", "default_model": "gpt-3.5-turbo"},
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions",
             "key_env": "GROQAPI_KEY", "default_model": "openai/gpt-oss-120b"},
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "key_env": "GEMINIAPI_KEY", "default_model": "gemini-flash-latest"},
}
PROVIDER = os.getenv("AGENT_PROVIDER", "openrouter")

messages = []
log_file = "agent_transcript.jsonl"
MAX_STEPS = 25  # max model calls per user message
MODEL = os.getenv("AGENT_MODEL") or PROVIDERS[PROVIDER]["default_model"]
# which edit tool the model gets: "str_replace" or "apply_edits" (SEARCH/REPLACE blocks)
EDIT_FORMAT = os.getenv("AGENT_EDIT_FORMAT", "str_replace")
EDIT_TOOLS = {"str_replace", "apply_edits"}
# ask | auto-read | auto-edit | auto  (see safety/permissions.py)
PERMISSION_MODE = os.getenv("AGENT_PERMISSION_MODE", "auto-read")
# docker = run bash in a container; off = run on this machine (not recommended)
SANDBOX_MODE = os.getenv("AGENT_SANDBOX", "docker")
NETWORK = os.getenv("AGENT_NETWORK", "off") == "on"
# tools = native tool calling; text = bash code blocks parsed from plain replies (the eval runner reads this too)
PROTOCOL = os.getenv("AGENT_PROTOCOL", "tools")
policy = None  # created on first use, for the current directory


def tool_names():
    # a function, not a constant, because EDIT_FORMAT can be changed at runtime (e.g. by the eval runner)
    return ["read_file", "list_dir", "search", "write_file", EDIT_FORMAT, "bash"]

turn_count = 0

def log_to_jsonl(event_type, data):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "data": data
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}


def backoff_delay(attempt, retry_after=None, base=1.0, cap=30.0, server_cap=120.0):
    """Exponential backoff with full jitter; honor the server's requested delay when it sends one."""
    if retry_after is not None:
        try:
            return min(float(retry_after) + 1, server_cap)  # +1s so we don't arrive a moment too early
        except ValueError:
            pass
    return random.uniform(0, min(cap, base * 2 ** attempt))


def retry_delay_hint(error):
    """Seconds to wait if the error body says so (Gemini: details[].RetryInfo.retryDelay = "27s")."""
    if not isinstance(error, dict):
        return None
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and detail.get("@type", "").endswith("RetryInfo"):
            return (detail.get("retryDelay") or "").rstrip("s") or None
    return None


INTERRUPTED = "Interrupted by the user before this finished. Wait for the user's next instruction."


def provider_settings(provider):
    """(url, api_key) for a provider; raises SystemExit with a clear message if it is not usable."""
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}")
    config = PROVIDERS[provider]
    key = (os.getenv(config["key_env"]) or "").strip()
    if not key:
        raise SystemExit(f"Provider {provider!r} needs {config['key_env']} in .env")
    return config["url"], key


def call_model(payload, on_text=None, log=None, max_retries=4, provider=None):
    """Stream a completion, retrying transient failures. Returns a normal response body, or None.

    on_text(piece) is called with each piece of text as it arrives.
    """
    log = log or log_to_jsonl
    url, api_key = provider_settings(provider or PROVIDER)
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    for attempt in range(max_retries + 1):
        retry_after = None
        try:
            with requests.post(
                url=url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                stream=True,
                timeout=(10, 120),  # (connect, longest silence between chunks)
            ) as response:
                if response.status_code == 200:
                    response.encoding = "utf-8"
                    return assemble(iter_events(response.iter_lines(decode_unicode=True)), on_text)
                try:
                    body = response.json()
                except ValueError:
                    body = {"error": {"message": response.text[:500]}}
                if isinstance(body, list):  # Gemini wraps errors in a list: [{"error": {...}}]
                    body = body[0] if body and isinstance(body[0], dict) else {}
                # the HTTP status decides retrying; body codes vary (Groq: "rate_limit_exceeded")
                status = response.status_code
                error = body.get("error")
                retry_after = response.headers.get("Retry-After") or retry_delay_hint(error)
        except StreamError as e:  # the server reported an error mid-stream
            status, error = (e.code if isinstance(e.code, int) else None), e.error  # unknown code: retry
            retry_after = retry_delay_hint(error)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            status, error = None, repr(e)

        retryable = status is None or status in RETRYABLE_STATUS
        final = not retryable or attempt == max_retries  # the call failed for good
        log("api_error", {"attempt": attempt, "status": status, "error": error, "retryable": retryable, "final": final})
        if not retryable:
            print(f"\nAPI error {status}: {error}")
            return None
        if attempt == max_retries:
            print(f"\nGiving up after {max_retries + 1} attempts (last error: {status or error})")
            return None
        delay = backoff_delay(attempt, retry_after)
        print(f"\n {status or 'network error'}, retry {attempt + 1}/{max_retries} in {delay:.1f}s...")
        time.sleep(delay)


def repair_history(messages):
    """Give every tool call a result; the API rejects a history with an unanswered tool call.

    Returns how many results had to be filled in.
    """
    repaired, i = 0, 0
    while i < len(messages):
        message = messages[i]
        i += 1
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        answered = set()
        while i < len(messages) and messages[i].get("role") == "tool":
            answered.add(messages[i]["tool_call_id"])
            i += 1
        for call in message["tool_calls"]:
            if call["id"] not in answered:
                messages.insert(i, {"role": "tool", "tool_call_id": call["id"], "content": INTERRUPTED})
                i += 1
                repaired += 1
    return repaired


def run_tool_call(tool_call):
    """Parse, permission-check and run one tool call. Returns the result text for the model."""
    name, raw_args = tool_call["function"]["name"], tool_call["function"]["arguments"]
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        result = f"Error: arguments were not valid JSON ({e}). Raw arguments: {raw_args}"
    else:
        allowed, reason = policy.check(name, args)
        log_to_jsonl("permission", {"call_id": tool_call["id"], "tool": name, "allowed": allowed, "reason": reason})
        if allowed:
            print(f"  → {name} {json.dumps(args)[:120]}")
            result = execute_tool(name, args)
        else:
            print(f"  ⛔ {name} blocked: {reason}")
            result = (f"Permission denied: {reason}. Do not retry the same action; "
                      f"find another way to do the task, or explain to the user why you cannot.")
    print(f"    {result.splitlines()[0][:100] if result else '(empty)'}")
    log_to_jsonl("tool_result", {"call_id": tool_call["id"], "tool": name, "args": raw_args, "result": result})
    return result


def agentic_loop(user_message):
    """One user turn, with whichever protocol is selected."""
    return text_loop(user_message) if PROTOCOL == "text" else tool_loop(user_message)


def tool_loop(user_message):
    global turn_count, messages, policy
    if policy is None:
        policy = Policy(os.getcwd(), PERMISSION_MODE)

    turn_count += 1
    messages.append({"role": "user", "content": user_message})
    log_to_jsonl("user_message", {"turn": turn_count, "content": user_message})

    for step in range(1, MAX_STEPS + 1):
        repaired = repair_history(messages)
        if repaired:
            log_to_jsonl("history_repaired", {"missing_tool_results": repaired})

        payload = {"model": MODEL, "messages": messages, "tools": tool_schemas(tool_names()), "max_tokens": 2000}
        log_to_jsonl("request", {"turn": turn_count, "step": step, "provider": PROVIDER, "payload": payload})

        streamed = []

        def show(piece):
            if not streamed:
                print("\nAssistant: ", end="", flush=True)
            streamed.append(piece)
            print(piece, end="", flush=True)

        try:
            response_data = call_model(payload, on_text=show)
        except KeyboardInterrupt:
            partial = "".join(streamed)
            if partial:  # keep what the user already saw, so the model knows where it was cut off
                messages.append({"role": "assistant", "content": partial + " [interrupted by the user]"})
            log_to_jsonl("interrupted", {"turn": turn_count, "step": step, "during": "model", "partial": partial})
            print("\n⏹  Interrupted.")
            return
        if streamed:
            print()
        if response_data is None:
            return
        log_to_jsonl("response", {"turn": turn_count, "step": step, "body": response_data})
        message = response_data["choices"][0]["message"]

        if not message.get("tool_calls"):
            messages.append({"role": "assistant", "content": message.get("content") or ""})
            return

        message["content"] = message.get("content") or ""
        messages.append(message)
        try:
            for tool_call in message["tool_calls"]:
                result = run_tool_call(tool_call)
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
        except KeyboardInterrupt:
            filled = repair_history(messages)
            log_to_jsonl("interrupted", {"turn": turn_count, "step": step, "during": "tools", "unfinished": filled})
            print(f"\n⏹  Interrupted. {filled} unfinished tool call(s) marked as interrupted.")
            return

    print(f"Reached max steps ({MAX_STEPS}) for this turn, stopping")
    log_to_jsonl("max_steps_reached", {"turn": turn_count, "steps": MAX_STEPS})


TEXT_SYSTEM_PROMPT = """You are a coding agent working in a project in the current directory. You act only by running bash commands.

Every reply must be: a short explanation of what you will do and why, then exactly ONE code block:

```bash
<command>
```

You will then see the command's exit code and output. Rules:
- Exactly one ```bash block per reply. To run several commands, chain them with && or ; in that block.
- Each command runs in a fresh shell in the project directory: `cd` does not carry over; use `cd dir && cmd`.
- There is no terminal and no input: never use interactive programs (vim, nano, less, python without a script).
- Create or overwrite files with `cat > path <<'EOF'` ... `EOF`; make small edits with `sed -i` or a python one-liner. Look at the file before and after editing.
- Verify your work (run the code or the tests) before finishing.
- When the task is finished, or you only need to answer the user, reply WITHOUT any code block; that ends your turn."""

BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```", re.S)
FORMAT_ERROR = ("Format error: your reply contained {n} code blocks. Reply with exactly ONE ```bash block "
                "(chain commands with && inside it), or with no code block at all if you are finished.")


def parse_commands(reply):
    """All bash code blocks in a reply (the text protocol expects zero or one)."""
    return [block.strip() for block in BLOCK_RE.findall(reply or "")]


def run_command(command):
    """Text protocol: permission-check and run one command; returns its output for the model."""
    allowed, reason = policy.check("bash", {"command": command})
    log_to_jsonl("permission", {"tool": "bash", "allowed": allowed, "reason": reason})
    if not allowed:
        print(f"  ⛔ blocked: {reason}")
        result = f"Permission denied: {reason}. Do not retry the same command; find another way."
    else:
        more = " ..." if len(command.splitlines()) > 1 else ""
        print(f"  $ {command.splitlines()[0][:120]}{more}")
        result = execute_tool("bash", {"command": command})
    print(f"    {result.splitlines()[0][:100]}")
    log_to_jsonl("tool_result", {"tool": "bash", "args": json.dumps({"command": command}), "result": result})
    return result


def text_loop(user_message):
    """The text-protocol turn: no tools in the request; bash blocks are parsed out of the reply."""
    global turn_count, policy
    if policy is None:
        policy = Policy(os.getcwd(), PERMISSION_MODE)
    if not messages:
        messages.append({"role": "system", "content": TEXT_SYSTEM_PROMPT})

    turn_count += 1
    messages.append({"role": "user", "content": user_message})
    log_to_jsonl("user_message", {"turn": turn_count, "content": user_message})

    for step in range(1, MAX_STEPS + 1):
        payload = {"model": MODEL, "messages": messages, "max_tokens": 2000}
        log_to_jsonl("request", {"turn": turn_count, "step": step, "payload": payload})
        streamed = []

        def show(piece):
            if not streamed:
                print("\nAssistant: ", end="", flush=True)
            streamed.append(piece)
            print(piece, end="", flush=True)

        try:
            body = call_model(payload, on_text=show, provider=PROVIDER)
        except KeyboardInterrupt:
            if streamed:
                messages.append({"role": "assistant", "content": "".join(streamed) + " [interrupted by the user]"})
            log_to_jsonl("interrupted", {"turn": turn_count, "step": step, "during": "model"})
            print("\n⏹  Interrupted.")
            return
        if streamed:
            print()
        if body is None:
            return
        log_to_jsonl("response", {"turn": turn_count, "step": step, "body": body})
        reply = body["choices"][0]["message"].get("content") or ""
        messages.append({"role": "assistant", "content": reply})

        commands = parse_commands(reply)
        if not commands:
            return  # no code block = the agent is done or is just answering
        if len(commands) > 1:
            log_to_jsonl("format_error", {"blocks": len(commands)})
            messages.append({"role": "user", "content": FORMAT_ERROR.format(n=len(commands))})
            continue
        try:
            output = run_command(commands[0])
        except KeyboardInterrupt:
            # no tool-call ids to pair up in this protocol: just tell the model what happened
            messages.append({"role": "user", "content": "The command was interrupted by the user. "
                                                        "Wait for the user's next instruction."})
            log_to_jsonl("interrupted", {"turn": turn_count, "step": step, "during": "command"})
            print("\n⏹  Interrupted.")
            return
        messages.append({"role": "user", "content": f"Command output:\n{output}"})

    print(f"Reached max steps ({MAX_STEPS}) for this turn, stopping")
    log_to_jsonl("max_steps_reached", {"turn": turn_count, "steps": MAX_STEPS})


def parse_args(description, text_protocol_option=False):
    """--provider/--model override AGENT_PROVIDER/AGENT_MODEL for this run (later days reuse this)."""
    global PROVIDER, MODEL, PROTOCOL
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--provider", default=PROVIDER, choices=list(PROVIDERS))
    parser.add_argument("--model", help="model id for that provider (default: AGENT_MODEL, else the provider's default)")
    if text_protocol_option:
        parser.add_argument("--text-protocol", action="store_true",
                            help="no tool API: the model writes bash code blocks (mini-SWE-agent style)")
    args = parser.parse_args()
    if text_protocol_option and args.text_protocol:
        PROTOCOL = "text"
    PROVIDER = args.provider
    MODEL = args.model or os.getenv("AGENT_MODEL") or PROVIDERS[PROVIDER]["default_model"]
    provider_settings(PROVIDER)  # fail now, not on the first message, if the key is missing
    return PROVIDER, MODEL


def main():
    parse_args("Day 7 streaming coding agent", text_protocol_option=True)
    kind = "Text-Protocol Agent (bash code blocks, no tool API)" if PROTOCOL == "text" else "Streaming Agent"
    print(f"Starting Day 7: {kind} (Ctrl-C interrupts the agent; Ctrl-C at the prompt exits)\n")
    sandbox = None
    if SANDBOX_MODE == "docker":
        sandbox = DockerSandbox(os.getcwd(), network=NETWORK).start()
        tools.shell.SANDBOX = sandbox
        print(f"🔒 bash runs in container {sandbox.name} (network {'on' if NETWORK else 'off'})")
    else:
        print("⚠️  No sandbox: bash runs directly on this machine")
    print(f"Provider: {PROVIDER}   Model: {MODEL}   Permission mode: {PERMISSION_MODE}\n")
    try:
        chat()
    finally:
        if sandbox:
            sandbox.stop()


def chat():
    print("Type your message. 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input:
            try:
                agentic_loop(user_input)
            except KeyboardInterrupt:  # e.g. during a retry wait; the next request repairs the history
                print("\n⏹  Interrupted.")


if __name__ == "__main__":
    main()
