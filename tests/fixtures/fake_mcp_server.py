"""A deliberately awkward MCP server over stdio, for testing the client. Written from the spec too.

    python fixtures/fake_mcp_server.py [protocol-version] [cancel-log-file]
"""
import json
import os
import sys
import time

VERSION = sys.argv[1] if len(sys.argv) > 1 else "2025-06-18"
CANCEL_LOG = sys.argv[2] if len(sys.argv) > 2 else None
_next_id = [1000]
BACKLOG = []  # messages that arrived while we were waiting for the client's answer to our own request

TOOLS = [
    {"name": "echo", "description": "Echo text back", "annotations": {"readOnlyHint": True},
     "inputSchema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                     "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "fail", "description": "Always reports a tool error", "inputSchema": {"type": "object"}},
    {"name": "crash", "description": "Exits in the middle of the call", "inputSchema": {"type": "object"}},
    {"name": "slow", "description": "Takes 5 seconds", "inputSchema": {"type": "object"}},
    {"name": "ask roots", "description": "Asks the client for its roots", "inputSchema": {"type": "object"}},
    {"name": "change_tools", "description": "Announces that the tool list changed", "inputSchema": {"type": "object"}},
    {"name": "wants_sampling", "description": "Asks the client for an LLM completion", "inputSchema": {"type": "object"}},
    {"name": "env", "description": "Lists the environment variable names it can see", "inputSchema": {"type": "object"}},
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def ask_client(method, params=None):
    """Send a request to the client and wait for its response (skipping anything else)."""
    _next_id[0] += 1
    request_id = _next_id[0]
    send({"jsonrpc": "2.0", "id": request_id, "method": method, **({"params": params} if params else {})})
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        if message.get("id") == request_id and "method" not in message:
            return message
        BACKLOG.append(message)  # not ours: handle it after this


def text(value, is_error=False):
    return {"content": [{"type": "text", "text": value}], "isError": is_error}


def call_tool(name, args):
    if name == "echo":
        return text(f"echo: {args.get('text', '')}")
    if name == "fail":
        return text("the database is unreachable", is_error=True)
    if name == "crash":
        sys.stderr.write("fatal: crashing on purpose\n")
        sys.stderr.flush()
        os._exit(3)
    if name == "slow":
        time.sleep(5)
        return text("finally done")
    if name == "ask roots":
        reply = ask_client("roots/list")
        return text(json.dumps(reply.get("result", reply.get("error"))))
    if name == "change_tools":
        send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        return text("tools changed")
    if name == "wants_sampling":
        reply = ask_client("sampling/createMessage", {"messages": [], "maxTokens": 10})
        return text(f"client answered with error code {reply.get('error', {}).get('code')}")
    if name == "env":
        return text(",".join(sorted(os.environ)))
    raise KeyError(name)


def main():
    sys.stderr.write("fake server starting\n" + "noise " * 30_000 + "\n")  # > 64 KB: the client must drain stderr
    sys.stderr.flush()
    print("this line is not JSON-RPC and must be ignored", flush=True)
    while True:
        if BACKLOG:
            message = BACKLOG.pop(0)
        else:
            line = sys.stdin.readline()
            if not line:
                break
            if not line.strip():
                continue
            message = json.loads(line)
        method, request_id, params = message.get("method"), message.get("id"), message.get("params") or {}
        if method is None:
            continue  # a response to something we asked outside ask_client
        if request_id is None:  # notification
            if method == "notifications/initialized":
                ping = ask_client("ping")  # servers may ping the client too
                sys.stderr.write(f"ping answered: {ping}\n")
            elif method == "notifications/cancelled" and CANCEL_LOG:
                with open(CANCEL_LOG, "a") as f:
                    f.write(json.dumps(params) + "\n")
            continue
        if method == "initialize":
            result = {"protocolVersion": VERSION, "capabilities": {"tools": {"listChanged": True}},
                      "serverInfo": {"name": "fake-server", "version": "0.1"},
                      "instructions": "Use echo to repeat things."}
        elif method == "tools/list":
            page = 1 if params.get("cursor") == "page-2" else 0
            result = {"tools": TOOLS[:4] if page == 0 else TOOLS[4:]}
            if page == 0:
                result["nextCursor"] = "page-2"
        elif method == "tools/call":
            try:
                result = call_tool(params["name"], params.get("arguments") or {})
            except KeyError:
                send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": f"unknown tool {params['name']}"}})
                continue
        else:
            send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"no method {method}"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id, "result": result})


if __name__ == "__main__":
    main()
