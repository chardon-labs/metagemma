from __future__ import annotations

import unittest

from sandbox_harness.bootstrap import python_repo_initial_files, toy_addition_task


class BootstrapTest(unittest.TestCase):
    def test_toy_task_has_expected_repo_files(self) -> None:
        files = python_repo_initial_files(toy_addition_task())
        self.assertIn("README.md", files)
        self.assertIn("solution.py", files)
        self.assertIn("tests/test_visible.py", files)
        self.assertIn("hidden_tests/test_hidden.py", files)


if __name__ == "__main__":
    unittest.main()
