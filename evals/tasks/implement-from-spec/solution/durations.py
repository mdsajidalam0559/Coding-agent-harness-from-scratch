def parse_duration(text):
    """Convert a duration like "1h30m", "45s", "2h" or "1h 5m 10s" into a number of seconds.

    Units are h (hours), m (minutes) and s (seconds). Units may appear at most once each and must
    be in the order h, m, s. Spaces between parts are allowed. Raise ValueError for anything else,
    including an empty string, unknown units, a number without a unit, or negative numbers.
    """
    import re
    match = re.fullmatch(r"\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?\s*", text)
    if not text.strip() or not match or not any(match.groups()):
        raise ValueError(f"invalid duration: {text!r}")
    h, m, s = (int(g or 0) for g in match.groups())
    return h * 3600 + m * 60 + s
