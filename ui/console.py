"""Terminal rendering of agent events (used by the interactive UI)."""
import json
import sys


class ConsoleUI:
    def __init__(self, out=sys.stdout):
        self.out = out
        self.streaming = False

    def _print(self, text):
        print(text, file=self.out, flush=True)

    def text(self, piece):
        if not self.streaming:
            print("\nAssistant: ", end="", file=self.out)
            self.streaming = True
        print(piece, end="", file=self.out, flush=True)

    def text_end(self):
        if self.streaming:
            print(file=self.out)
            self.streaming = False

    def tool_start(self, agent, name, args):
        prefix = "    ↳ " if agent == "sub" else "  → "
        self._print(f"{prefix}{name} {json.dumps(args)[:110]}")

    def tool_end(self, agent, name, result):
        first = result.splitlines()[0][:100] if result else "(empty)"
        self._print(f"{'      ' if agent == 'sub' else '    '}{first}")

    def blocked(self, agent, name, reason):
        self._print(f"  ⛔ {name}: {reason}")

    def status(self, message):
        self._print(f"  🗜  {message}")


class StderrUI(ConsoleUI):
    """Progress on stderr, so stdout carries only the result (headless mode)."""

    def __init__(self):
        super().__init__(out=sys.stderr)

    def text(self, piece):  # no streamed prose in headless progress output
        pass
