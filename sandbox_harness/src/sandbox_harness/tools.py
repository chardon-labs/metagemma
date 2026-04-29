from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sandbox_harness.backends import CommandResult, SandboxSession
from sandbox_harness.types import JsonObject, JsonValue, json_object, json_string, optional_json_int

ACTIVE_TOOL_NAMES = ("read", "bash", "edit", "write")
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
DEFAULT_READ_LIMIT = DEFAULT_MAX_LINES


@dataclass(frozen=True)
class OpenAIFunctionTool:
    type: str
    function: JsonObject


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    command_result: CommandResult | None = None

    def to_openai_message(self) -> JsonObject:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "content": self.content,
        }


def _schema(properties: JsonObject, required: list[str], *, additional_properties: bool | None = None) -> JsonObject:
    required_values: list[JsonValue] = [item for item in required]
    schema: JsonObject = {
        "type": "object",
        "properties": properties,
        "required": required_values,
    }
    if additional_properties is not None:
        schema["additionalProperties"] = additional_properties
    return schema


def pi_function_tools() -> list[OpenAIFunctionTool]:
    return [
        OpenAIFunctionTool(
            type="function",
            function={
                "name": "read",
                "description": (
                    "Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
                    f"Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines "
                    f"or {DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). Use offset/limit for large files. "
                    "When you need the full file, continue with offset until complete."
                ),
                "parameters": _schema(
                    {
                        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                        "offset": {
                            "type": "number",
                            "description": "Line number to start reading from (1-indexed)",
                        },
                        "limit": {"type": "number", "description": "Maximum number of lines to read"},
                    },
                    ["path"],
                ),
            },
        ),
        OpenAIFunctionTool(
            type="function",
            function={
                "name": "bash",
                "description": (
                    "Execute a bash command in the current working directory. Returns stdout and stderr. "
                    f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES / 1024}KB "
                    "(whichever is hit first). If truncated, full output is saved to a temp file. "
                    "Optionally provide a timeout in seconds."
                ),
                "parameters": _schema(
                    {
                        "command": {"type": "string", "description": "Bash command to execute."},
                        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
                    },
                    ["command"],
                ),
            },
        ),
        OpenAIFunctionTool(
            type="function",
            function={
                "name": "edit",
                "description": (
                    "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
                    "non-overlapping region of the original file. If two changes affect the same block or nearby "
                    "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
                    "unchanged regions just to connect distant changes."
                ),
                "parameters": _schema(
                    {
                        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                        "edits": {
                            "type": "array",
                            "description": (
                                "One or more targeted replacements. Each edit is matched against the original file, "
                                "not incrementally. Do not include overlapping or nested edits. If two changes touch "
                                "the same block or nearby lines, merge them into one edit instead."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "oldText": {
                                        "type": "string",
                                        "description": (
                                            "Exact text for one targeted replacement. It must be unique in the "
                                            "original file and must not overlap with any other edits[].oldText "
                                            "in the same call."
                                        ),
                                    },
                                    "newText": {
                                        "type": "string",
                                        "description": "Replacement text for this targeted edit.",
                                    },
                                },
                                "required": ["oldText", "newText"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    ["path", "edits"],
                    additional_properties=False,
                ),
            },
        ),
        OpenAIFunctionTool(
            type="function",
            function={
                "name": "write",
                "description": (
                    "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
                    "Automatically creates parent directories."
                ),
                "parameters": _schema(
                    {
                        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                        "content": {"type": "string", "description": "Content to write to the file"},
                    },
                    ["path", "content"],
                ),
            },
        ),
    ]


class ToolExecutor:
    def __init__(self, session: SandboxSession) -> None:
        self.session = session

    def execute(self, tool_call_id: str, name: str, arguments: Mapping[str, JsonValue]) -> ToolResult:
        try:
            if name == "read":
                return self._read(tool_call_id, arguments)
            if name == "bash":
                return self._bash(tool_call_id, arguments)
            if name == "edit":
                return self._edit(tool_call_id, arguments)
            if name == "write":
                return self._write(tool_call_id, arguments)
            return ToolResult(tool_call_id=tool_call_id, tool_name=name, content=f"Unknown tool: {name}", is_error=True)
        except Exception as exc:
            return ToolResult(tool_call_id=tool_call_id, tool_name=name, content=str(exc), is_error=True)

    def _workspace_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            candidate = Path(*candidate.parts[1:])
        if ".." in candidate.parts:
            raise ValueError(f"Path escapes workspace: {path}")
        resolved = (self.session.workspace / candidate).resolve()
        workspace = self.session.workspace.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def _read(self, tool_call_id: str, arguments: Mapping[str, JsonValue]) -> ToolResult:
        path = json_string(arguments.get("path"), name="path")
        offset = optional_json_int(arguments.get("offset"), name="offset") or 1
        limit = optional_json_int(arguments.get("limit"), name="limit") or DEFAULT_READ_LIMIT
        if offset < 1:
            raise ValueError("offset must be >= 1.")
        if limit < 1:
            raise ValueError("limit must be >= 1.")

        target = self._workspace_path(path)
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        selected = lines[offset - 1 : offset - 1 + limit]
        prefix = "" if offset == 1 else f"[Showing lines {offset}-{offset + len(selected) - 1}]\n"
        suffix = "" if offset - 1 + limit >= len(lines) else f"\n[Truncated at line {offset + limit - 1} of {len(lines)}]"
        return ToolResult(tool_call_id=tool_call_id, tool_name="read", content=f"{prefix}{''.join(selected)}{suffix}")

    def _bash(self, tool_call_id: str, arguments: Mapping[str, JsonValue]) -> ToolResult:
        command = json_string(arguments.get("command"), name="command")
        timeout = optional_json_int(arguments.get("timeout"), name="timeout")
        result = self.session.run(
            ["/bin/bash", "-lc", command],
            cwd=PurePosixPath("/workspace"),
            timeout_seconds=timeout,
        )
        pieces: list[str] = []
        if result.stdout:
            pieces.append(result.stdout)
        if result.stderr:
            pieces.append(result.stderr)
        if result.timed_out:
            pieces.append(f"\n[command timed out after {result.duration_seconds:.2f}s]")
        content = "".join(pieces)
        if not content:
            content = f"[exit code {result.exit_code}]"
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="bash",
            content=content,
            is_error=not result.ok,
            command_result=result,
        )

    def _edit(self, tool_call_id: str, arguments: Mapping[str, JsonValue]) -> ToolResult:
        path = json_string(arguments.get("path"), name="path")
        edits_value = arguments.get("edits")
        if not isinstance(edits_value, list):
            raise TypeError("edits must be an array.")

        target = self._workspace_path(path)
        text = target.read_text(encoding="utf-8")
        for edit_value in edits_value:
            edit = json_object(edit_value)
            old_text = json_string(edit.get("oldText"), name="oldText")
            new_text = json_string(edit.get("newText"), name="newText")
            occurrences = text.count(old_text)
            if occurrences != 1:
                raise ValueError(f"Expected exactly one occurrence of oldText, found {occurrences}.")
            text = text.replace(old_text, new_text, 1)
        target.write_text(text, encoding="utf-8")
        return ToolResult(tool_call_id=tool_call_id, tool_name="edit", content=f"Applied {len(edits_value)} edit(s) to {path}.")

    def _write(self, tool_call_id: str, arguments: Mapping[str, JsonValue]) -> ToolResult:
        path = json_string(arguments.get("path"), name="path")
        content = json_string(arguments.get("content"), name="content")
        target = self._workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(tool_call_id=tool_call_id, tool_name="write", content=f"Wrote {len(content)} bytes to {path}.")
