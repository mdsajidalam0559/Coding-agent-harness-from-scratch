"""Choosing a model: one place that knows providers, endpoints, keys and context windows.

A model spec is "<provider>:<model id>", optionally with "+text" to force the text protocol:
    groq:qwen/qwen3.8-27b    anthropic:claude-haiku-4-5    ollama:qwen3:8b+text
or the name of an entry in models.json (repo) / ~/.agent/models.json:
    {"fast": {"provider": "groq", "model": "openai/gpt-oss-20b", "context_window": 7000, "native_tools": true}}
Adding a model = adding a config entry; nothing else changes.
"""
import json
import os

from dotenv import load_dotenv

from context.budget import context_window as known_window
from models.anthropic_http import AnthropicAdapter
from models.openai_compat import OpenAICompatAdapter
from models.text_protocol import TextProtocolAdapter

PROVIDERS = {
    "openrouter": {"adapter": "openai", "url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_KEY"},
    "groq": {"adapter": "openai", "url": "https://api.groq.com/openai/v1", "key_env": "GROQAPI_KEY"},
    "gemini": {"adapter": "openai", "url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_env": "GEMINIAPI_KEY"},
    "ollama": {"adapter": "openai", "url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), "key_env": None},
    "anthropic": {"adapter": "anthropic", "key_env": "ANTHROPIC_API_KEY"},
}
DEFAULT_MODEL = "groq:openai/gpt-oss-120b"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_DIR, ".env"))  # API keys; explicit path because the agent may run in another directory
CONFIG_FILES = [os.path.join(REPO_DIR, "models.json"),
                os.path.expanduser("~/.agent/models.json")]


def load_aliases():
    aliases = {}
    for path in CONFIG_FILES:
        if os.path.exists(path):
            with open(path) as f:
                aliases.update(json.load(f))
    return aliases


def make_model(spec=None, log=None):
    """A ModelAdapter for a spec or alias (see module docstring)."""
    spec = spec or os.getenv("AGENT_MODEL_SPEC") or DEFAULT_MODEL
    entry = load_aliases().get(spec)
    if entry is None:
        text = spec.endswith("+text")
        provider, sep, model = spec.removesuffix("+text").partition(":")
        if not sep or provider not in PROVIDERS:
            raise ValueError(f"model spec {spec!r} should look like 'groq:qwen/qwen3.8-27b' "
                             f"(providers: {', '.join(PROVIDERS)}) or be an alias in models.json")
        entry = {"provider": provider, "model": model, "native_tools": not text}
    provider = PROVIDERS[entry["provider"]]
    window = entry.get("context_window") or known_window(entry["model"])
    name = f"{entry['provider']}:{entry['model']}"
    if provider["adapter"] == "anthropic":
        adapter = AnthropicAdapter(entry["model"], provider["key_env"], window, name=name, log=log)
    else:
        adapter = OpenAICompatAdapter(entry.get("url") or provider["url"], entry["model"], provider["key_env"],
                                      window, name=name, log=log)
    return adapter if entry.get("native_tools", True) else TextProtocolAdapter(adapter)
