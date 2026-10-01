"""Day 1: a stateless API, multi-turn by resending history, token usage logged. Run: python -m tests.test_day1"""
import json
import os
import tempfile
from unittest import mock

import requests

from days import day_1_agent as agent


def check(label, condition, detail=""):
    assert condition, f"{label}\n{detail}"
    print(f"✅ {label}")


def reply(text, prompt_tokens, finish="stop"):
    r = mock.Mock(status_code=200)
    r.json.return_value = {"model": "openai/gpt-3.5-turbo", "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": text}}],
                           "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5, "total_tokens": prompt_tokens + 5, "cost": 0.0001}}
    return r


def main():
    agent.log_file = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    replies, sent_requests = [reply("Hi Sam!", 20), reply("Your name is Sam.", 40)], []

    def post(*args, **kwargs):  # snapshot each request: the agent keeps appending to the same list
        sent_requests.append(json.loads(json.dumps(kwargs["json"])))
        return replies.pop(0)

    with mock.patch.object(requests, "post", side_effect=post), mock.patch("builtins.print"):
        agent.chat("Hi, I'm Sam")
        second = agent.chat("What is my name?")
    sent = sent_requests[1]["messages"]
    check("the second request resends the whole conversation (that is the model's only memory)",
          [m["role"] for m in sent] == ["system", "user", "assistant", "user"] and sent[1]["content"] == "Hi, I'm Sam")
    check("the reply comes back and is added to the history", second == "Your name is Sam." and agent.messages[-1]["content"] == second)
    events = [json.loads(line) for line in open(agent.log_file)]
    usage = [e["data"] for e in events if e["type"] == "api_response"]
    check("every response is logged with its token counts", [u["prompt_tokens"] for u in usage] == [20, 40]
          and all(u["completion_tokens"] == 5 and u["cost"] for u in usage))
    check("prompt tokens grow each turn, because the history is resent", usage[1]["prompt_tokens"] > usage[0]["prompt_tokens"])

    error = mock.Mock(status_code=401)
    error.json.return_value = {"error": {"message": "Missing Authentication header"}}
    before = len(agent.messages)
    with mock.patch.object(requests, "post", return_value=error), mock.patch("builtins.print"):
        check("an API error returns None instead of crashing", agent.chat("hello?") is None)
    check("...and leaves the history unchanged", len(agent.messages) == before)
    with mock.patch.object(requests, "post", return_value=reply("Once upon a", 50, finish="length")), \
            mock.patch("builtins.print") as printed:
        agent.chat("tell me a story")
    check("finish_reason 'length' (a cut-off reply) is flagged", any("token limit" in str(c) for c in printed.call_args_list))
    print("\nDay 1 tests passed.")


if __name__ == "__main__":
    main()
