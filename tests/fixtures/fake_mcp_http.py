"""A minimal Streamable HTTP MCP server for testing the client's HTTP transport (in-process, stdlib only).

Checks the client follows the transport rules: sends Accept with both content types, echoes the
Mcp-Session-Id it was given, sends MCP-Protocol-Version after initialization, and DELETEs the session
at the end. Answers tools/call with an SSE stream (a progress notification, then the response).
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SESSION = "session-abc123"


class Handler(BaseHTTPRequestHandler):
    seen = []  # (method, headers, body) of every request, for the test to inspect

    def log_message(self, *args):
        pass

    def _record(self, body=None):
        Handler.seen.append((self.command, dict(self.headers), body))

    def do_DELETE(self):
        self._record()
        self.send_response(200 if self.headers.get("Mcp-Session-Id") == SESSION else 404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._record(body)
        accept = self.headers.get("Accept", "")
        if "application/json" not in accept or "text/event-stream" not in accept:
            return self._reply(406, {"error": "Accept must list application/json and text/event-stream"})
        method = body.get("method")
        if method != "initialize" and self.headers.get("Mcp-Session-Id") != SESSION:
            return self._reply(404, {"error": "unknown session"})
        if "id" not in body:  # notification (or a response from the client)
            self.send_response(202)
            self.end_headers()
            return
        if method == "initialize":
            return self._reply(200, {"jsonrpc": "2.0", "id": body["id"], "result": {
                "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-http", "version": "0.1"}}}, session=True)
        if self.headers.get("MCP-Protocol-Version") != "2025-06-18":
            return self._reply(400, {"error": "missing MCP-Protocol-Version header"})
        if method == "tools/list":
            return self._reply(200, {"jsonrpc": "2.0", "id": body["id"], "result": {"tools": [
                {"name": "shout", "description": "Upper-cases text",
                 "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}})
        if method == "tools/call":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [{"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": 1, "progress": 0.5}},
                      {"jsonrpc": "2.0", "id": body["id"], "result": {"content": [
                          {"type": "text", "text": body["params"]["arguments"]["text"].upper()}]}}]
            for event in events:
                self.wfile.write(f"event: message\ndata: {json.dumps(event)}\n\n".encode())
            return
        self._reply(200, {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": "no such method"}})

    def _reply(self, status, payload, session=False):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if session:
            self.send_header("Mcp-Session-Id", SESSION)
        self.end_headers()
        self.wfile.write(data)


def serve():
    """Start on a free port; returns (url, server)."""
    Handler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/mcp", server
