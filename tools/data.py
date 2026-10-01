"""Tools for a data-analysis agent: explore CSV files and query them with read-only SQL.

Every .csv in <workspace>/data/ and in the workspace root becomes an SQLite table (named after the file)
in an in-memory database, rebuilt whenever a file changes. Column types are inferred from the values.
"""
import csv
import os
import re
import sqlite3

from .registry import TOOL_CATEGORIES, tool, workspace as current_workspace

MAX_ROWS = 100
SAMPLE_FOR_TYPES = 500
_cache = {"key": None, "connection": None, "tables": {}}
for _name, _category in (("list_tables", "read"), ("describe_table", "read"), ("query", "read"),
                         ("write_report", "write")):
    TOOL_CATEGORIES[_name] = _category


def _csv_files(workspace):
    found = []
    for folder in (os.path.join(workspace, "data"), workspace):
        if os.path.isdir(folder):
            found += sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".csv"))
    return found


def _table_name(path, taken):
    name = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(path))[0]).strip("_").lower() or "table"
    if name[0].isdigit():
        name = "t_" + name
    base, n = name, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    return name


def _convert(value, kind):
    if value is None or value.strip() == "":
        return None
    try:
        return int(value) if kind == "INTEGER" else float(value) if kind == "REAL" else value
    except ValueError:
        return value


def _infer(values):
    kind = "INTEGER"
    for value in values:
        if value is None or value.strip() == "":
            continue
        try:
            int(value)
        except ValueError:
            try:
                float(value)
                kind = "REAL"
            except ValueError:
                return "TEXT"
    return kind


def _connection():
    """The in-memory database for the current workspace, rebuilt when a CSV changes."""
    workspace = current_workspace()
    files = _csv_files(workspace)
    key = (workspace, tuple((f, os.path.getmtime(f), os.path.getsize(f)) for f in files))
    if key == _cache["key"]:
        return _cache["connection"], _cache["tables"]
    connection = sqlite3.connect(":memory:")
    tables = {}
    for path in files:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header = [re.sub(r"\W+", "_", h.strip()).strip("_").lower() or f"col{i}" for i, h in enumerate(rows[0])]
        data = [r + [""] * (len(header) - len(r)) for r in rows[1:]]
        types = [_infer([r[i] for r in data[:SAMPLE_FOR_TYPES]]) for i in range(len(header))]
        name = _table_name(path, tables)
        columns = ", ".join(f'"{h}" {t}' for h, t in zip(header, types))
        connection.execute(f'CREATE TABLE "{name}" ({columns})')
        connection.executemany(f'INSERT INTO "{name}" VALUES ({", ".join("?" * len(header))})',
                               [[_convert(v, t) for v, t in zip(r[:len(header)], types)] for r in data])
        tables[name] = {"file": os.path.relpath(path, workspace), "rows": len(data), "columns": list(zip(header, types))}
    connection.execute("PRAGMA query_only = ON")
    _cache.update(key=key, connection=connection, tables=tables)
    return connection, tables


def _format(cursor, rows, limit=MAX_ROWS):
    names = [d[0] for d in cursor.description]
    shown = rows[:limit]
    widths = [max([len(n)] + [len(_cell(r[i])) for r in shown]) for i, n in enumerate(names)]
    lines = [" | ".join(n.ljust(w) for n, w in zip(names, widths)), "-+-".join("-" * w for w in widths)]
    lines += [" | ".join(_cell(v).ljust(w) for v, w in zip(r, widths)) for r in shown]
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows (add LIMIT or aggregate)")
    return "\n".join(lines) + f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})"


def _cell(value):
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)[:80]


@tool("List the datasets (tables made from the CSV files) with their row counts and columns.", {}, required=[])
def list_tables():
    _, tables = _connection()
    if not tables:
        return "No CSV files found in data/ or the project folder."
    return "\n".join(f"- {name} ({t['rows']} rows, from {t['file']}): "
                     + ", ".join(f"{c} {k}" for c, k in t["columns"]) for name, t in tables.items())


@tool(
    "Profile one table: each column's type, missing values, distinct values and min/max, plus sample rows.",
    {"table": {"type": "string", "description": "Table name from list_tables"}},
)
def describe_table(table):
    connection, tables = _connection()
    if table not in tables:
        return f"Error: no table {table!r}. Tables: {', '.join(tables) or 'none'}."
    lines = [f"{table}: {tables[table]['rows']} rows"]
    for column, kind in tables[table]["columns"]:
        nulls, distinct, low, high = connection.execute(
            f'SELECT SUM("{column}" IS NULL), COUNT(DISTINCT "{column}"), MIN("{column}"), MAX("{column}") FROM "{table}"'
        ).fetchone()
        lines.append(f"- {column} {kind}: {nulls or 0} missing, {distinct} distinct, min {_cell(low)}, max {_cell(high)}")
    cursor = connection.execute(f'SELECT * FROM "{table}" LIMIT 5')
    return "\n".join(lines) + "\n\nSample rows:\n" + _format(cursor, cursor.fetchall())


@tool(
    "Run one read-only SQL query (SQLite dialect: SELECT or WITH) over the tables and return the result. "
    f"At most {MAX_ROWS} rows are shown: aggregate or use LIMIT for large results.",
    {"sql": {"type": "string", "description": "A single SELECT (or WITH ... SELECT) statement"}},
)
def query(sql):
    statement = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", statement):
        return "Error: only read-only SELECT or WITH queries are allowed."
    if ";" in statement:
        return "Error: send one statement at a time."
    connection, _ = _connection()
    try:
        cursor = connection.execute(statement)
        return _format(cursor, cursor.fetchall())
    except sqlite3.Error as e:
        return f"SQL error: {e}"


@tool(
    "Save a Markdown report to reports/<filename>. Use it for the final write-up when the user asks for a report.",
    {"filename": {"type": "string", "description": "e.g. sales_summary.md (letters, digits, - and _ only)"},
     "content": {"type": "string", "description": "The report in Markdown"}},
)
def write_report(filename, content):
    if not re.fullmatch(r"[\w-]+\.md", filename):
        return "Error: filename must look like name.md (letters, digits, - and _ only)."
    path = os.path.join("reports", filename)
    os.makedirs(os.path.join(current_workspace(), "reports"), exist_ok=True)
    with open(os.path.join(current_workspace(), path), "w") as f:
        f.write(content)
    return f"Saved {path} ({len(content.splitlines())} lines)."
