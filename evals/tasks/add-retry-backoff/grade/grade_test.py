import unittest
from http_client import TransientError, fetch


def flaky(failures, error=TransientError):
    calls = []

    def transport(url):
        calls.append(url)
        if len(calls) <= failures:
            raise error("boom")
        return f"ok:{url}"
    return transport, calls


class Grade(unittest.TestCase):
    def test_success_first_try(self):
        transport, calls = flaky(0)
        self.assertEqual(fetch("u", transport), "ok:u")
        self.assertEqual(len(calls), 1)

    def test_retries_with_backoff(self):
        transport, calls = flaky(2)
        sleeps = []
        self.assertEqual(fetch("u", transport, sleep=sleeps.append), "ok:u")
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_gives_up(self):
        transport, calls = flaky(10)
        sleeps = []
        with self.assertRaises(TransientError):
            fetch("u", transport, max_retries=3, sleep=sleeps.append)
        self.assertEqual(len(calls), 4)
        self.assertEqual(sleeps, [0.5, 1.0, 2.0])

    def test_other_errors_not_retried(self):
        transport, calls = flaky(5, error=KeyError)
        with self.assertRaises(KeyError):
            fetch("u", transport, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)
