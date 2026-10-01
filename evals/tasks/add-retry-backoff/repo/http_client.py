class TransientError(Exception):
    """A temporary failure (timeout, 503, ...) that is worth retrying."""


def fetch(url, transport):
    """Fetch url using transport(url) and return its result."""
    return transport(url)
