import time


class TransientError(Exception):
    """A temporary failure (timeout, 503, ...) that is worth retrying."""


def fetch(url, transport, max_retries=3, sleep=time.sleep):
    """Fetch url using transport(url) and return its result."""
    for attempt in range(max_retries + 1):
        try:
            return transport(url)
        except TransientError:
            if attempt == max_retries:
                raise
            sleep(0.5 * 2 ** attempt)
