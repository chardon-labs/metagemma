from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import cast

from sandbox_harness.serialization import render_gemma_chat
from sandbox_harness.tools import pi_function_tools
from sandbox_harness.types import JsonObject, JsonValue


class FakeTokenizer:
    def __init__(self) -> None:
        self.tools: list[JsonObject] | None = None
        self.messages: list[dict[str, JsonValue]] | None = None
        self.tokenize = True
        self.add_generation_prompt = False
        self.enable_thinking = False

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, JsonValue]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: Sequence[JsonObject] | None = None,
    ) -> str:
        self.messages = [dict(message) for message in conversation]
        self.tools = None if tools is None else [dict(tool) for tool in tools]
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.enable_thinking = enable_thinking
        return "rendered"


class SerializationTest(unittest.TestCase):
    def test_render_passes_openai_style_tool_payloads(self) -> None:
        tokenizer = FakeTokenizer()
        rendered = render_gemma_chat(
            tokenizer,
            [{"role": "user", "content": "hello"}],
            tools=pi_function_tools(),
            enable_thinking=True,
        )
        self.assertEqual(rendered, "rendered")
        self.assertIsNotNone(tokenizer.tools)
        if tokenizer.tools is None:
            raise AssertionError("Expected tools to be recorded.")
        self.assertEqual(tokenizer.tools[0]["type"], "function")
        self.assertIn("function", tokenizer.tools[0])
        function = cast(JsonObject, tokenizer.tools[0]["function"])
        self.assertEqual(function["name"], "read")


if __name__ == "__main__":
    unittest.main()
