from __future__ import annotations

import unittest
from datetime import date

from sandbox_harness.prompt import PiPromptPaths, build_pi_system_prompt


class PromptTest(unittest.TestCase):
    def test_default_prompt_matches_pi_text_shape(self) -> None:
        prompt = build_pi_system_prompt(
            cwd="/tmp/work",
            paths=PiPromptPaths(
                readme_path="/pi/README.md",
                docs_path="/pi/docs",
                examples_path="/pi/examples",
            ),
            current_date=date(2026, 4, 28),
        )
        self.assertIn("You are an expert coding assistant operating inside pi, a coding agent harness.", prompt)
        self.assertIn("- read: Read file contents", prompt)
        self.assertIn("- bash: Execute bash commands (ls, grep, find, etc.)", prompt)
        self.assertIn("- edit: Make precise file edits with exact text replacement", prompt)
        self.assertIn("- write: Create or overwrite files", prompt)
        self.assertIn("- Use bash for file operations like ls, rg, find", prompt)
        self.assertIn("- Use read to examine files instead of cat or sed.", prompt)
        self.assertTrue(prompt.endswith("Current date: 2026-04-28\nCurrent working directory: /tmp/work"))


if __name__ == "__main__":
    unittest.main()
