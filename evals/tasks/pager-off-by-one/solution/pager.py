def paginate(items, page, per_page):
    """Return the items on the given page. Pages start at 1."""
    start = (page - 1) * per_page
    return items[start:start + per_page]


def page_count(total, per_page):
    """Number of pages needed to show `total` items."""
    return (total + per_page - 1) // per_page
