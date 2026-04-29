from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

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
    ) -> str: ...


def render_gemma_chat(
    tokenizer: GemmaChatTokenizer,
    messages: Sequence[Mapping[str, JsonValue]],
    *,
    tools: Sequence[OpenAIFunctionTool],
    enable_thinking: bool,
    add_generation_prompt: bool = True,
) -> str:
    tool_payloads: list[JsonObject] = [
        {"type": tool.type, "function": cast(JsonValue, tool.function)}
        for tool in tools
    ]
    return tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        tools=tool_payloads,
    )
