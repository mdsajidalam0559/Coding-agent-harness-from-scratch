REGISTRY = {}
# Permission category (read | write | exec) of tools that are not built in: MCP tools, data tools, ...
# The permission engine (safety/permissions.py) reads this table.
TOOL_CATEGORIES = {}


def tool(description, parameters, required=None):
    """Register a function as a tool; its name becomes the tool name."""
    def decorator(fn):
        REGISTRY[fn.__name__] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": required if required is not None else list(parameters),
                    },
                },
            },
        }
        return fn
    return decorator


def tool_schemas(names):
    """Schemas for the given tool names, in that order."""
    return [REGISTRY[name]["schema"] for name in names]


def _argument_error(name, schema, args, strict=True):
    """A helpful message if args do not match the schema, else None (models do invent parameters).

    strict: reject parameters the schema does not list. True for our own Python tools (an unexpected
    keyword would crash them); external tools follow JSON Schema, where extras are allowed by default.
    """
    params, required = schema.get("properties") or {}, schema.get("required") or []
    unknown = sorted(set(args) - set(params)) if strict else []
    missing = [p for p in required if p not in args]
    if not unknown and not missing:
        return None
    problems = ([f"unknown parameter(s) {', '.join(unknown)}"] if unknown else []) + \
               ([f"missing required parameter(s) {', '.join(missing)}"] if missing else [])
    valid = ", ".join(f"{p} ({'required' if p in required else 'optional'})" for p in params)
    return f"Error: invalid arguments for {name}: {'; '.join(problems)}. Valid parameters: {valid}."


def execute_tool(name, args):
    entry = REGISTRY.get(name)
    if entry is None:
        return f"Unknown tool: {name}. Available tools: {', '.join(REGISTRY)}."
    if not isinstance(args, dict):
        return f"Error: arguments for {name} must be a JSON object."
    error = _argument_error(name, entry["schema"]["function"]["parameters"], args, entry.get("strict_args", True))
    if error:
        return error
    try:
        return str(entry["fn"](**args))
    except Exception as e:
        return f"Tool error: {e}"
