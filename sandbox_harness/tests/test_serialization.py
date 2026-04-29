from __future__ import annotations

import unittest

from sandbox_harness.serialization import scan_native_tool_calls, scan_native_tool_responses


class SerializationTest(unittest.TestCase):
    def test_scans_native_tool_call_tags(self) -> None:
        calls = scan_native_tool_calls("<|tool_call>call:bash{command:<|\"|>pytest<|\"|>}<tool_call|>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bash")
        self.assertIn("pytest", calls[0].arguments_text)

    def test_scans_native_tool_response_tags(self) -> None:
        responses = scan_native_tool_responses("<|tool_response>response:bash{value:<|\"|>ok<|\"|>}<tool_response|>")
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].name, "bash")
        self.assertIn("ok", responses[0].response_text)


if __name__ == "__main__":
    unittest.main()
