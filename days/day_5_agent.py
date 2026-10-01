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

# Load API key first
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("Error: OPENROUTER_KEY not found in .env file")

messages = []
log_file = "agent_transcript.jsonl"
MAX_STEPS = 25  # max model calls per user message
MODEL = os.getenv("AGENT_MODEL", "gpt-3.5-turbo")
# which edit tool the model gets: "str_replace" or "apply_edits" (SEARCH/REPLACE blocks)
EDIT_FORMAT = os.getenv("AGENT_EDIT_FORMAT", "str_replace")
EDIT_TOOLS = {"str_replace", "apply_edits"}
# ask | auto-read | auto-edit | auto  (see safety/permissions.py)
PERMISSION_MODE = os.getenv("AGENT_PERMISSION_MODE", "auto-read")
# docker = run bash in a container; off = run on this machine (not recommended)
SANDBOX_MODE = os.getenv("AGENT_SANDBOX", "docker")
NETWORK = os.getenv("AGENT_NETWORK", "off") == "on"
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


def backoff_delay(attempt, retry_after=None, base=1.0, cap=30.0):
    """Exponential backoff with full jitter; honor Retry-After when the server sends it."""
    if retry_after is not None:
        try:
            return min(float(retry_after), cap)
        except ValueError:
            pass
    return random.uniform(0, min(cap, base * 2 ** attempt))


def call_model(payload, max_retries=4):
    """POST to the API, retrying transient failures. Returns the response body or None."""
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json=payload,
                timeout=60,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            status, retry_after, error = None, None, repr(e)
        else:
            try:
                body = response.json()
            except ValueError:
                body = {"error": {"message": response.text[:500]}}
            # OpenRouter can also return an error inside a 200 body
            if response.status_code == 200 and "error" not in body:
                return body
            status = body.get("error", {}).get("code", response.status_code)
            retry_after = response.headers.get("Retry-After")
            error = body.get("error")

        retryable = status is None or status in RETRYABLE_STATUS
        log_to_jsonl("api_error", {"attempt": attempt, "status": status, "error": error, "retryable": retryable})

        if not retryable:
            print(f"API error {status}: {error}")
            return None
        if attempt == max_retries:
            print(f"Giving up after {max_retries + 1} attempts (last error: {status or error})")
            return None

        delay = backoff_delay(attempt, retry_after)
        print(f" {status or 'network error'}, retry {attempt + 1}/{max_retries} in {delay:.1f}s...")
        time.sleep(delay)


def agentic_loop(user_message):
    global turn_count, messages, policy
    if policy is None:
        policy = Policy(os.getcwd(), PERMISSION_MODE)

    turn_count += 1
    print(f"\n=== Turn {turn_count} ===")

    messages.append({"role": "user", "content": user_message})
    log_to_jsonl("user_message", {"turn": turn_count, "content": user_message})
    print(f" User: {user_message}")

    steps = 0

    while True:
        if steps >= MAX_STEPS:
            print(f"Reached max steps ({MAX_STEPS}) for this turn, stopping")
            log_to_jsonl("max_steps_reached", {"turn": turn_count, "steps": steps})
            return
        steps += 1

        payload = {
            "model": MODEL,
            "messages": messages,
            "tools": tool_schemas(tool_names()),
            "max_tokens": 2000
        }
        print(f"Calling API (step {steps})...")
        log_to_jsonl("request", {"turn": turn_count, "step": steps, "payload": payload})
        response_data = call_model(payload)
        if response_data is None:
            return

        log_to_jsonl("response", {"turn": turn_count, "step": steps, "body": response_data})
        choice = response_data['choices'][0]
        finish_reason = choice['finish_reason']
        message = choice['message']

        print(f"✓ Got response, finish_reason: {finish_reason}")

        if message.get('tool_calls'):
            # Ensure assistant message has content field
            if 'content' not in message or message['content'] is None:
                message['content'] = ''
            messages.append(message)
            print(f"  Assistant requested tools: {[tc['function']['name'] for tc in message['tool_calls']]}")

            for tool_call in message['tool_calls']:
                tool_name = tool_call['function']['name']
                raw_args = tool_call['function']['arguments']
                try:
                    tool_input = json.loads(raw_args or "{}")
                except json.JSONDecodeError as e:
                    tool_input = None
                    result = f"Error: arguments were not valid JSON ({e}). Raw arguments: {raw_args}"

                if tool_input is not None:
                    allowed, reason = policy.check(tool_name, tool_input)
                    log_to_jsonl("permission", {"call_id": tool_call['id'], "tool": tool_name,
                                                "allowed": allowed, "reason": reason})
                    if allowed:
                        print(f"  → Executing {tool_name}({tool_input})")
                        result = execute_tool(tool_name, tool_input)
                    else:
                        print(f"  ⛔ Blocked {tool_name}: {reason}")
                        result = (f"Permission denied: {reason}. Do not retry the same action; "
                                  f"find another way to do the task, or explain to the user why you cannot.")
                print(f"    ✓ Result: {result[:80]}")
                log_to_jsonl("tool_result", {
                    "call_id": tool_call['id'], "tool": tool_name,
                    "args": raw_args, "result": result,
                })

                # OpenAI format: one role="tool" message per tool call
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": result
                })

            print("Sending tool results back to model...")

        else:
            response_text = message.get('content', '')
            messages.append({"role": "assistant", "content": response_text})
            print(f"Assistant: {response_text}")
            break

def main():
    print("Starting Day 5: Sandboxed Agent\n")
    sandbox = None
    if SANDBOX_MODE == "docker":
        sandbox = DockerSandbox(os.getcwd(), network=NETWORK).start()
        tools.shell.SANDBOX = sandbox
        print(f"🔒 bash runs in container {sandbox.name} (network {'on' if NETWORK else 'off'})")
    else:
        print("⚠️  No sandbox: bash runs directly on this machine")
    print(f"Permission mode: {PERMISSION_MODE}\n")
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
            agentic_loop(user_input)


if __name__ == "__main__":
    main()
