"""Day 2 checkpoint: the agent survives simulated API errors. Run: python -m tests.test_day2"""
import os
import tempfile
from unittest import mock

import requests

from days import day_2_agent as agent


def fake_response(status, body, headers=None):
    r = mock.Mock(status_code=status, headers=headers or {}, text=str(body))
    r.json.return_value = body
    return r


def text_reply(text):
    return fake_response(200, {"choices": [{"finish_reason": "stop",
                                            "message": {"role": "assistant", "content": text}}]})


def tool_reply(call_id, name, args):
    return fake_response(200, {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": args}}]}}]})


def run(case):
    with mock.patch.object(agent.time, "sleep") as sleep:
        case(sleep)
    print(f"✅ {case.__name__}")


def test_429_then_success(sleep):
    with mock.patch.object(requests, "post", side_effect=[
        fake_response(429, {"error": {"message": "rate limited", "code": 429}}, {"Retry-After": "7"}),
        text_reply("ok"),
    ]):
        assert agent.call_model({})["choices"][0]["message"]["content"] == "ok"
    sleep.assert_called_once_with(7.0)


def test_5xx_backs_off_then_succeeds(sleep):
    with mock.patch.object(requests, "post", side_effect=[
        fake_response(503, {"error": {"message": "overloaded", "code": 503}}),
        fake_response(502, {"error": {"message": "bad gateway", "code": 502}}),
        text_reply("ok"),
    ]):
        assert agent.call_model({}) is not None
    assert sleep.call_count == 2


def test_timeout_and_connection_error_are_retried(sleep):
    with mock.patch.object(requests, "post", side_effect=[
        requests.exceptions.Timeout(),
        requests.exceptions.ConnectionError(),
        text_reply("ok"),
    ]):
        assert agent.call_model({}) is not None


def test_error_inside_200_body_is_retried(sleep):
    with mock.patch.object(requests, "post", side_effect=[
        fake_response(200, {"error": {"message": "provider overloaded", "code": 502}}),
        text_reply("ok"),
    ]):
        assert agent.call_model({}) is not None


def test_400_is_not_retried(sleep):
    post = mock.Mock(return_value=fake_response(400, {"error": {"message": "bad request", "code": 400}}))
    with mock.patch.object(requests, "post", post):
        assert agent.call_model({}) is None
    assert post.call_count == 1
    sleep.assert_not_called()


def test_gives_up_after_max_retries(sleep):
    post = mock.Mock(return_value=fake_response(503, {"error": {"message": "down", "code": 503}}))
    with mock.patch.object(requests, "post", post):
        assert agent.call_model({}, max_retries=3) is None
    assert post.call_count == 4


def test_agent_survives_error_mid_task(sleep):
    """Multi-step task with an outage between tool call and final answer."""
    agent.messages.clear()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "note.txt")
        with mock.patch.object(requests, "post", side_effect=[
            tool_reply("c1", "write_file", f'{{"path": "{path}", "content": "hi"}}'),
            fake_response(503, {"error": {"message": "overloaded", "code": 503}}),
            fake_response(429, {"error": {"message": "slow down", "code": 429}}),
            text_reply("done"),
        ]):
            agent.agentic_loop("write hi to note.txt")
        assert open(path).read() == "hi"
    assert agent.messages[-1] == {"role": "assistant", "content": "done"}


if __name__ == "__main__":
    agent.log_file = os.devnull
    for case in [test_429_then_success, test_5xx_backs_off_then_succeeds,
                 test_timeout_and_connection_error_are_retried, test_error_inside_200_body_is_retried,
                 test_400_is_not_retried, test_gives_up_after_max_retries,
                 test_agent_survives_error_mid_task]:
        run(case)
    print("\nDay 2 checkpoint passed.")
