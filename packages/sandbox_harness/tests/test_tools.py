from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from sandbox_harness.backends import CommandLimits, CommandResult
from sandbox_harness.tools import ToolExecutor, pi_function_tools


class FakeSession:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.commands: list[Sequence[str]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: PurePosixPath | str = PurePosixPath("/workspace"),
        env: Mapping[str, str] | None = None,
        timeout_seconds: int | None = None,
        limits: CommandLimits | None = None,
    ) -> CommandResult:
        self.commands.append(tuple(argv))
        return CommandResult(
            argv=tuple(argv),
            exit_code=0,
            stdout="ok\n",
            stderr="",
            timed_out=False,
            duration_seconds=0.01,
        )

    def cleanup(self) -> None:
        pass

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        pass


class ToolExecutorTest(unittest.TestCase):
    def test_declares_only_required_pi_tools(self) -> None:
        names = [tool.function["name"] for tool in pi_function_tools()]
        self.assertEqual(names, ["read", "bash", "edit", "write"])

    def test_write_read_edit_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeSession(Path(tmp))
            executor = ToolExecutor(session)

            write = executor.execute("1", "write", {"path": "solution.py", "content": "def add(a, b):\n    return 0\n"})
            self.assertFalse(write.is_error)

            edit = executor.execute(
                "2",
                "edit",
                {
                    "path": "solution.py",
                    "edits": [{"oldText": "return 0", "newText": "return a + b"}],
                },
            )
            self.assertFalse(edit.is_error)

            read = executor.execute("3", "read", {"path": "solution.py"})
            self.assertIn("return a + b", read.content)

    def test_rejects_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = ToolExecutor(FakeSession(Path(tmp)))
            result = executor.execute("1", "write", {"path": "../escape.txt", "content": "bad"})
            self.assertTrue(result.is_error)

    def test_bash_runs_through_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeSession(Path(tmp))
            executor = ToolExecutor(session)
            result = executor.execute("1", "bash", {"command": "printf ok"})
            self.assertFalse(result.is_error)
            self.assertEqual(session.commands, [("bash", "-lc", "printf ok")])


if __name__ == "__main__":
    unittest.main()
