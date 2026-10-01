"""Agent Skills: folders of instructions (plus optional scripts and reference files) used on demand.

    my-skill/
      SKILL.md          ---
                        name: my-skill
                        description: What it does, and WHEN to use it.
                        ---
                        Instructions...
      scripts/, *.md    optional; SKILL.md tells the model when to read or run them

Progressive disclosure: only each skill's name and description go into the system prompt. When a task
matches, the model calls load_skill(name) to get the instructions and the list of the skill's files,
and opens those files only if it needs them.
Locations: <project>/.agent/skills/ (ships with the repo) and ~/.agent/skills/ (yours); project wins.
"""
import os
import re

from safety.permissions import TOOL_CATEGORIES
from tools import tool
from tools.registry import workspace as current_workspace

PROJECT_DIR = os.path.join(".agent", "skills")
USER_DIR = os.path.expanduser("~/.agent/skills")
SANDBOX_USER_DIR = "/skills"  # where user skills are mounted (read-only) inside the sandbox
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME, MAX_DESCRIPTION = 64, 1024
MAX_LISTED_FILES = 50

STATE = {"skills": {}, "user_dir_in_sandbox": None}  # set by discover() / the agent at startup
TOOL_CATEGORIES["load_skill"] = "read"


def parse_frontmatter(text):
    """(metadata, body) from a SKILL.md. A small YAML subset: `key: value`, quoted values, | and > blocks."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text
    meta, i = {}, 1
    while i < end:
        line = lines[i]
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        i += 1
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value in ("|", "|-", ">", ">-"):  # block scalar: the indented lines that follow
            block = []
            while i < end and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                block.append(lines[i].strip())
                i += 1
            value = ("\n" if value.startswith("|") else " ").join(block).strip()
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif value == "":  # a nested mapping (e.g. metadata:); skip its indented lines
            while i < end and lines[i].startswith((" ", "\t")):
                i += 1
            continue
        meta[key] = value
    return meta, "\n".join(lines[end + 1:]).strip()


def _files(skill_dir):
    found = []
    for root, dirs, names in os.walk(skill_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        for name in sorted(names):
            rel = os.path.relpath(os.path.join(root, name), skill_dir)
            if rel != "SKILL.md":
                found.append(rel)
    return found


def discover(workspace, user_dir=None):
    """Valid skills by name, and a list of problems with invalid ones."""
    user_dir = USER_DIR if user_dir is None else user_dir  # read at call time, so it can be reconfigured
    skills, problems = {}, []
    for source, base in (("user", user_dir), ("project", os.path.join(workspace, PROJECT_DIR))):
        if not base or not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            skill_dir = os.path.join(base, entry)
            skill_md = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            with open(skill_md, encoding="utf-8", errors="replace") as f:
                meta, body = parse_frontmatter(f.read())
            name, description = meta.get("name", ""), meta.get("description", "")
            if not NAME_RE.match(name) or len(name) > MAX_NAME:
                problems.append(f"{skill_md}: name must be lowercase letters, digits and hyphens (max {MAX_NAME})")
                continue
            if not description or len(description) > MAX_DESCRIPTION:
                problems.append(f"{skill_md}: description is required (max {MAX_DESCRIPTION} characters)")
                continue
            if name != entry:
                problems.append(f"{skill_md}: name {name!r} should match its folder {entry!r} (loaded anyway)")
            skills[name] = {"name": name, "description": description, "body": body, "source": source,
                            "dir": os.path.realpath(skill_dir), "files": _files(skill_dir)}  # project overrides user
    STATE["skills"] = skills
    return skills, problems


def prompt_section(skills):
    if not skills:
        return ""
    lines = [f"- {s['name']}: {s['description']}" for s in skills.values()]
    return ("\n\n# Skills\nSkills are packaged instructions for particular kinds of tasks. If the task matches a "
            "skill's description, call load_skill with its name BEFORE you start, then follow the instructions it "
            "returns.\n" + "\n".join(lines))


def _paths(skill, rel, workspace):
    """(path for read_file, path for bash) of a file inside a skill."""
    full = os.path.join(skill["dir"], rel)
    if skill["source"] == "project":
        local = os.path.relpath(full, workspace)
        return local, local
    in_sandbox = STATE["user_dir_in_sandbox"]
    bash_path = f"{in_sandbox}/{skill['name']}/{rel}" if in_sandbox else full
    return full, bash_path


@tool(
    "Load a skill's full instructions (see the Skills list in the system prompt). Call it before starting a "
    "task that matches a skill's description.",
    {"name": {"type": "string", "description": "The skill's name"}},
)
def load_skill(name):
    skill = STATE["skills"].get(name)
    if skill is None:
        available = ", ".join(STATE["skills"]) or "none"
        return f"Error: no skill named {name!r}. Available skills: {available}."
    out = f"# Skill: {skill['name']}\n\n{skill['body']}"
    if skill["files"]:
        workspace = current_workspace()
        listing = []
        for rel in skill["files"][:MAX_LISTED_FILES]:
            read_path, bash_path = _paths(skill, rel, workspace)
            listing.append(f"- {rel}: read_file path `{read_path}`" +
                           (f"; in bash `{bash_path}`" if bash_path != read_path else ""))
        more = len(skill["files"]) - MAX_LISTED_FILES
        out += ("\n\n## Files in this skill (open or run them only when the instructions call for it)\n"
                + "\n".join(listing) + (f"\n- ... and {more} more" if more > 0 else ""))
    return out
