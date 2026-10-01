"""A Model Context Protocol client written from the specification (no MCP library).

MCP is JSON-RPC 2.0 between a client (us) and a server that offers tools. Two transports:
- stdio: we start the server as a subprocess and exchange one JSON message per line on its stdin/stdout.
  Its stderr is for logs only.
- Streamable HTTP: every message we send is POSTed to one URL; the reply is either plain JSON or a
  Server-Sent Events stream that carries the response (and possibly requests/notifications for us).
Lifecycle: `initialize` (agree on a protocol version and capabilities) -> `notifications/initialized`
-> normal operation (`tools/list`, `tools/call`, ...) -> shutdown.
Messages are matched by id: a request has an id and expects a response with the same id; a
notification has no id and gets no response. The server may also send US requests (e.g. `ping`).
"""
import itertools
import json
import os
import subprocess
import threading

import requests

from models.sse import iter_events

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
CLIENT_INFO = {"name": "coding-agent", "title": "Coding agent built from scratch", "version": "0.12"}
REQUEST_TIMEOUT = 60
# A server gets a minimal environment, never our API keys (the official SDKs do the same).
INHERITED_ENV = ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER", "LANG", "LC_ALL", "TMPDIR"]

# JSON-RPC error codes
METHOD_NOT_FOUND = -32601
CONNECTION_CLOSED = -32000
REQUEST_TIMED_OUT = -32001

_CLOSED = object()


class MCPError(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(f"MCP error {code}: {message}")
        self.code, self.message, self.data = code, message, data


class StdioTransport:
    """The server as a subprocess; newline-delimited JSON on stdin/stdout."""

    def __init__(self, command, args=(), env=None, cwd=None):
        self.command = [command, *args]
        self.env = {k: os.environ[k] for k in INHERITED_ENV if k in os.environ} | (env or {})
        self.cwd = cwd
        self.proc = None
        self.stderr_tail = []
        self._write_lock = threading.Lock()

    def start(self, on_message):
        try:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, env=self.env, cwd=self.cwd)
        except FileNotFoundError as e:
            raise MCPError(CONNECTION_CLOSED, f"cannot start {self.command[0]!r}: {e}") from None
        threading.Thread(target=self._read_stdout, args=(on_message,), daemon=True).start()
        # stderr must be drained continuously, or a chatty server blocks once the pipe buffer fills
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self, on_message):
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.stderr_tail.append(f"[non-JSON on stdout] {line[:200]}")
                continue
            on_message(message)
        on_message(None)  # EOF: the server exited

    def _read_stderr(self):
        for raw in self.proc.stderr:
            self.stderr_tail = (self.stderr_tail + [raw.decode("utf-8", errors="replace").rstrip()])[-50:]

    def send(self, message):
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")  # json.dumps escapes newlines
        with self._write_lock:
            try:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                raise MCPError(CONNECTION_CLOSED, "the server's stdin is closed (it exited)") from None

    def close(self):
        """Spec shutdown for stdio: close stdin, wait, then SIGTERM, then SIGKILL."""
        if not self.proc:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        for stop in (None, self.proc.terminate, self.proc.kill):
            if stop:
                stop()
            try:
                self.proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                continue


class HttpTransport:
    """Streamable HTTP: POST each message; the reply is JSON or an SSE stream."""

    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers or {}
        self.session_id = None
        self.protocol_version = None  # sent on every request after initialization
        self.on_message = None

    def start(self, on_message):
        self.on_message = on_message

    def _headers(self):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    def send(self, message):
        try:
            response = requests.post(self.url, json=message, headers=self._headers(), stream=True,
                                     timeout=(10, REQUEST_TIMEOUT))
        except requests.RequestException as e:
            raise MCPError(CONNECTION_CLOSED, f"HTTP request failed: {e}") from None
        with response:
            if response.headers.get("Mcp-Session-Id"):
                self.session_id = response.headers["Mcp-Session-Id"]
            if response.status_code == 202:  # accepted: a notification or a response we sent
                return
            if response.status_code == 404 and self.session_id:
                raise MCPError(CONNECTION_CLOSED, "the server ended the session (HTTP 404)")
            if response.status_code >= 400:
                raise MCPError(CONNECTION_CLOSED, f"HTTP {response.status_code}: {response.text[:300]}")
            content_type = response.headers.get("Content-Type", "")
            if content_type.startswith("text/event-stream"):
                response.encoding = "utf-8"
                for event in iter_events(response.iter_lines(decode_unicode=True)):
                    self.on_message(event)
            elif content_type.startswith("application/json"):
                body = response.json()
                for item in body if isinstance(body, list) else [body]:
                    self.on_message(item)

    def close(self):
        if self.session_id:  # tell the server the session is over (it may not support this)
            try:
                requests.delete(self.url, headers=self._headers(), timeout=5)
            except requests.RequestException:
                pass


class MCPClient:
    def __init__(self, name, transport, roots=()):
        self.name = name
        self.transport = transport
        self.roots = [os.path.realpath(r) for r in roots]  # directories we tell the server it may work in
        self.server_info, self.capabilities, self.instructions = {}, {}, None
        self.protocol_version = None
        self.tools = []
        self.tools_changed = False  # set by notifications/tools/list_changed
        self.log = []  # notifications/message from the server
        self._pending = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # ---- JSON-RPC plumbing ----

    def request(self, method, params=None, timeout=REQUEST_TIMEOUT):
        request_id = next(self._ids)
        slot = {"done": threading.Event(), "response": None}
        with self._lock:
            self._pending[request_id] = slot
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            self.transport.send(message)
        except MCPError:
            self._pending.pop(request_id, None)
            raise
        if not slot["done"].wait(timeout):
            self._pending.pop(request_id, None)
            try:  # tell the server to stop working on it
                self.notify("notifications/cancelled", {"requestId": request_id, "reason": f"timed out after {timeout}s"})
            except MCPError:
                pass
            raise MCPError(REQUEST_TIMED_OUT, f"{method} timed out after {timeout}s")
        response = slot["response"]
        if response is _CLOSED:
            raise MCPError(CONNECTION_CLOSED, f"the server closed the connection during {method}")
        if "error" in response:
            error = response["error"] or {}
            raise MCPError(error.get("code"), error.get("message", "unknown error"), error.get("data"))
        return response.get("result") or {}

    def notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.transport.send(message)

    def _respond(self, request_id, result=None, error=None):
        message = {"jsonrpc": "2.0", "id": request_id}
        message.update({"error": error} if error else {"result": result if result is not None else {}})
        self.transport.send(message)

    def _on_message(self, message):
        if message is None:  # transport closed: fail everything still waiting
            with self._lock:
                pending, self._pending = self._pending, {}
            for slot in pending.values():
                slot["response"] = _CLOSED
                slot["done"].set()
            return
        if isinstance(message, list):  # JSON-RPC batch (allowed by the 2025-03-26 revision)
            for item in message:
                self._on_message(item)
            return
        if not isinstance(message, dict):
            return
        if "method" in message and "id" in message:
            self._handle_server_request(message)
        elif "method" in message:
            self._handle_notification(message)
        elif "id" in message:
            with self._lock:
                slot = self._pending.pop(message["id"], None)
            if slot:
                slot["response"] = message
                slot["done"].set()

    def _handle_server_request(self, message):
        method, request_id = message["method"], message["id"]
        try:
            if method == "ping":
                self._respond(request_id, {})
            elif method == "roots/list" and self.roots:
                self._respond(request_id, {"roots": [{"uri": f"file://{root}", "name": os.path.basename(root)}
                                                     for root in self.roots]})
            else:  # sampling, elicitation, ...: capabilities we did not declare
                self._respond(request_id, error={"code": METHOD_NOT_FOUND, "message": f"client does not support {method}"})
        except MCPError:
            pass  # the connection is gone; nothing to answer

    def _handle_notification(self, message):
        method, params = message["method"], message.get("params") or {}
        if method == "notifications/tools/list_changed":
            self.tools_changed = True
        elif method == "notifications/message":
            self.log = (self.log + [f"[{params.get('level', 'info')}] {params.get('data')}"])[-50:]
        # progress, cancelled, resources/*: nothing to do for a tools-only client

    # ---- MCP operations ----

    def connect(self):
        self.transport.start(self._on_message)
        capabilities = {"roots": {"listChanged": False}} if self.roots else {}
        result = self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": capabilities,
                                             "clientInfo": CLIENT_INFO})
        version = result.get("protocolVersion")
        if version not in SUPPORTED_VERSIONS:
            self.close()
            raise MCPError(CONNECTION_CLOSED, f"server speaks protocol {version!r}; this client supports "
                                              f"{', '.join(SUPPORTED_VERSIONS)}")
        self.protocol_version = version
        if isinstance(self.transport, HttpTransport):
            self.transport.protocol_version = version
        self.server_info = result.get("serverInfo") or {}
        self.capabilities = result.get("capabilities") or {}
        self.instructions = result.get("instructions")
        self.notify("notifications/initialized")
        if "tools" in self.capabilities:
            self.list_tools()
        return self

    def list_tools(self):
        tools, cursor = [], None
        for _ in range(100):  # follow pagination, with a safety limit
            result = self.request("tools/list", {"cursor": cursor} if cursor else None)
            tools += result.get("tools") or []
            cursor = result.get("nextCursor")
            if not cursor:
                break
        self.tools, self.tools_changed = tools, False
        return tools

    def call_tool(self, name, arguments, timeout=REQUEST_TIMEOUT):
        return self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)

    def close(self):
        self.transport.close()

    def stderr(self):
        return "\n".join(getattr(self.transport, "stderr_tail", []))


def format_result(result, max_chars=30_000):
    """A tools/call result as text for the model."""
    parts = []
    for block in result.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} {block.get('mimeType', '')}, {len(block.get('data', '')) * 3 // 4} bytes, not shown]")
        elif kind == "resource":
            resource = block.get("resource") or {}
            parts.append(resource.get("text") if "text" in resource else f"[binary resource {resource.get('uri')}]")
        elif kind == "resource_link":
            parts.append(f"[resource link: {block.get('uri')} {block.get('name', '')}]".strip())
        else:
            parts.append(f"[unsupported content type {kind!r}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False, indent=2))
    text = "\n".join(p for p in parts if p) or "(the tool returned no content)"
    if result.get("isError"):
        text = f"Error reported by the tool: {text}"
    return text[:max_chars]


def make_client(name, config, workspace):
    """Client for one entry of an mcpServers config ({"command", "args", "env", "cwd"} or {"url", "headers"})."""
    def expand(value):
        value = str(value).replace("${workspace}", workspace)
        while "${env:" in value:  # explicit opt-in for passing a secret from our environment
            start = value.index("${env:")
            end = value.index("}", start)
            value = value[:start] + os.environ.get(value[start + 6:end], "") + value[end + 1:]
        return value

    if "command" in config:
        transport = StdioTransport(expand(config["command"]), [expand(a) for a in config.get("args", [])],
                                   {k: expand(v) for k, v in (config.get("env") or {}).items()},
                                   cwd=expand(config["cwd"]) if config.get("cwd") else workspace)
    elif "url" in config:
        transport = HttpTransport(expand(config["url"]), {k: expand(v) for k, v in (config.get("headers") or {}).items()})
    else:
        raise ValueError(f"MCP server {name!r} needs either 'command' or 'url'")
    return MCPClient(name, transport, roots=[workspace])
