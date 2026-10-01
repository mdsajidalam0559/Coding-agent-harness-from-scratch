"""Day 1: what an LLM API actually is. A multi-turn chat over raw HTTP: no SDK, no tools.

Models are stateless: every request sends the whole conversation, and "memory" is us resending the
history. Every request and response is logged to JSONL with its token counts.

    python -m days.day_1_agent
"""
import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "gpt-3.5-turbo"
CONTEXT_WINDOW = 16_385  # gpt-3.5-turbo
SYSTEM_PROMPT = "You are a helpful coding assistant. Be concise."
TEMPERATURE = 0.7  # 0 = focused and repeatable, 1 = more varied

log_file = "agent_transcript.jsonl"
messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def log_to_jsonl(event_type, data):
    with open(log_file, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(), "type": event_type, "data": data}) + "\n")


def chat(user_message):
    """Send the whole history plus the new message; return the reply (or None on error)."""
    messages.append({"role": "user", "content": user_message})
    log_to_jsonl("user_message", {"content": user_message})
    try:
        response = requests.post(URL, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, timeout=60,
                                 json={"model": MODEL, "messages": messages, "temperature": TEMPERATURE})
    except requests.exceptions.RequestException as e:
        log_to_jsonl("api_error", {"type": "network", "error": repr(e)})
        print(f"Network error: {e}")
        messages.pop()  # keep the history consistent: the message was never answered
        return None

    data = response.json()
    if response.status_code != 200:
        log_to_jsonl("api_error", {"status_code": response.status_code, "error": data.get("error")})
        print(f"API error {response.status_code}: {data.get('error', {}).get('message', data)}")
        messages.pop()
        return None

    usage = data.get("usage", {})
    choice = data["choices"][0]
    log_to_jsonl("api_response", {
        "status_code": response.status_code, "model": data.get("model"),
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"), "cost": usage.get("cost"),
        "finish_reason": choice.get("finish_reason"), "message_count": len(messages),
    })
    if choice.get("finish_reason") == "length":  # the reply was cut off by the token limit
        print("⚠️  The reply hit the token limit and is incomplete.")

    reply = choice["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    log_to_jsonl("assistant_message", {"content": reply})
    used = usage.get("total_tokens") or 0
    print(f"  [tokens: {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out · "
          f"context {100 * used / CONTEXT_WINDOW:.1f}% used · cost ${usage.get('cost') or 0:.6f}]")
    return reply


def main():
    if not OPENROUTER_API_KEY:
        raise SystemExit("Error: OPENROUTER_KEY not found in .env file")
    print("Day 1 chat (raw HTTP, multi-turn). Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input:
            reply = chat(user_input)
            if reply is not None:
                print(f"AI: {reply}\n")


if __name__ == "__main__":
    main()
