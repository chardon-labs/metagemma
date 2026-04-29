from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from sandbox_harness.tools import OpenAIFunctionTool
from sandbox_harness.types import JsonObject, JsonValue


class GemmaChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, JsonValue]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: Sequence[JsonObject] | None = None,
    ) -> str:
        pass


@dataclass(frozen=True)
class NativeToolCall:
    name: str
    arguments_text: str
    raw_text: str


@dataclass(frozen=True)
class NativeToolResponse:
    name: str
    response_text: str
    raw_text: str


_TOOL_CALL_RE = re.compile(r"<\|tool_call>call:([A-Za-z_][A-Za-z0-9_]*)\{(.*?)\}<tool_call\|>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(
    r"<\|tool_response>response:([A-Za-z_][A-Za-z0-9_]*)\{(.*?)\}<tool_response\|>",
    re.DOTALL,
)


def render_gemma_chat(
    tokenizer: GemmaChatTokenizer,
    messages: Sequence[Mapping[str, JsonValue]],
    *,
    tools: Sequence[OpenAIFunctionTool],
    enable_thinking: bool,
    add_generation_prompt: bool = True,
) -> str:
    tool_payloads = [tool.function for tool in tools]
    return tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        tools=tool_payloads,
    )


def scan_native_tool_calls(rendered_text: str) -> list[NativeToolCall]:
    return [
        NativeToolCall(name=match.group(1), arguments_text=match.group(2), raw_text=match.group(0))
        for match in _TOOL_CALL_RE.finditer(rendered_text)
    ]


def scan_native_tool_responses(rendered_text: str) -> list[NativeToolResponse]:
    return [
        NativeToolResponse(name=match.group(1), response_text=match.group(2), raw_text=match.group(0))
        for match in _TOOL_RESPONSE_RE.finditer(rendered_text)
    ]
