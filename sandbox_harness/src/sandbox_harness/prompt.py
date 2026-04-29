from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PromptContextFile:
    path: str
    content: str


@dataclass(frozen=True)
class PiPromptPaths:
    readme_path: str
    docs_path: str
    examples_path: str


PI_DEFAULT_TOOL_NAMES = ("read", "bash", "edit", "write")
PI_TOOL_SNIPPETS = {
    "read": "Read file contents",
    "bash": "Execute bash commands (ls, grep, find, etc.)",
    "edit": "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
    "write": "Create or overwrite files",
}
PI_TOOL_GUIDELINES = {
    "read": ("Use read to examine files instead of cat or sed.",),
    "edit": (
        "Use edit for precise changes (edits[].oldText must match exactly)",
        "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
        "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
        "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
    ),
    "write": ("Use write only for new files or complete rewrites.",),
}


def build_pi_system_prompt(
    *,
    cwd: str,
    paths: PiPromptPaths,
    current_date: date | None = None,
    selected_tools: Sequence[str] = PI_DEFAULT_TOOL_NAMES,
    tool_snippets: Mapping[str, str] = PI_TOOL_SNIPPETS,
    prompt_guidelines: Sequence[str] = (),
    append_system_prompt: str | None = None,
    context_files: Sequence[PromptContextFile] = (),
    custom_prompt: str | None = None,
) -> str:
    today = date.today() if current_date is None else current_date
    date_text = today.strftime("%Y-%m-%d")
    prompt_cwd = cwd.replace("\\", "/")
    append_section = f"\n\n{append_system_prompt}" if append_system_prompt else ""

    if custom_prompt:
        prompt = custom_prompt
        if append_section:
            prompt += append_section
        prompt = _append_context_files(prompt, context_files)
        prompt += f"\nCurrent date: {date_text}"
        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    visible_tools = [name for name in selected_tools if name in tool_snippets]
    tools_list = "\n".join(f"- {name}: {tool_snippets[name]}" for name in visible_tools) if visible_tools else "(none)"
    guidelines = "\n".join(f"- {guideline}" for guideline in _build_guidelines(selected_tools, prompt_guidelines))

    prompt = f"""You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
{guidelines}

Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: {paths.readme_path}
- Additional docs: {paths.docs_path}
- Examples: {paths.examples_path} (extensions, custom tools, SDK)
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md)
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)"""

    if append_section:
        prompt += append_section
    prompt = _append_context_files(prompt, context_files)
    prompt += f"\nCurrent date: {date_text}"
    prompt += f"\nCurrent working directory: {prompt_cwd}"
    return prompt


def _build_guidelines(selected_tools: Sequence[str], extra_guidelines: Sequence[str]) -> list[str]:
    guidelines: list[str] = []
    seen: set[str] = set()

    def add(guideline: str) -> None:
        if guideline in seen:
            return
        seen.add(guideline)
        guidelines.append(guideline)

    has_bash = "bash" in selected_tools
    has_grep = "grep" in selected_tools
    has_find = "find" in selected_tools
    has_ls = "ls" in selected_tools
    if has_bash and not has_grep and not has_find and not has_ls:
        add("Use bash for file operations like ls, rg, find")
    elif has_bash and (has_grep or has_find or has_ls):
        add("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)")

    for name in selected_tools:
        for guideline in PI_TOOL_GUIDELINES.get(name, ()):
            add(guideline)
    for guideline in extra_guidelines:
        normalized = guideline.strip()
        if normalized:
            add(normalized)
    add("Be concise in your responses")
    add("Show file paths clearly when working with files")
    return guidelines


def _append_context_files(prompt: str, context_files: Sequence[PromptContextFile]) -> str:
    if not context_files:
        return prompt
    prompt += "\n\n# Project Context\n\n"
    prompt += "Project-specific instructions and guidelines:\n\n"
    for context_file in context_files:
        prompt += f"## {context_file.path}\n\n{context_file.content}\n\n"
    return prompt
