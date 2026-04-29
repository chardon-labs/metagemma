from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from sandbox_harness.backends import CommandLimits, CommandResult, SandboxSession
from sandbox_harness.runner import AgentTurn, EpisodeRunner, EpisodeTask, VerifierResult
from sandbox_harness.tools import OpenAIFunctionTool
from sandbox_harness.types import JsonObject


class FakeSession:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.cleaned = False

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: PurePosixPath | str = PurePosixPath("/workspace"),
        env: Mapping[str, str] | None = None,
        timeout_seconds: int | None = None,
        limits: CommandLimits | None = None,
    ) -> CommandResult:
        return CommandResult(
            argv=tuple(argv),
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            duration_seconds=0.01,
        )

    def cleanup(self) -> None:
        self.cleaned = True


class FakeBackend:
    name = "fake"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.session: FakeSession | None = None

    def is_supported(self) -> bool:
        return True

    def create_session(
        self,
        *,
        initial_files: Mapping[str, str] | None = None,
        metadata: JsonObject | None = None,
    ) -> SandboxSession:
        workspace = self.root / "workspace"
        workspace.mkdir()
        if initial_files is not None:
            for path, content in initial_files.items():
                target = workspace / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        self.session = FakeSession(workspace)
        return self.session


class RunnerTest(unittest.TestCase):
    def test_runs_tool_loop_and_records_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = FakeBackend(root)
            runner = EpisodeRunner(backend=backend, output_dir=str(root / "episodes"), max_turns=3)
            calls = 0

            def generate_turn(messages: Sequence[JsonObject], tools: Sequence[OpenAIFunctionTool]) -> AgentTurn:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return AgentTurn(
                        assistant_message={
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write",
                                        "arguments": json.dumps({"path": "solution.py", "content": "print('ok')\n"}),
                                    },
                                }
                            ],
                        }
                    )
                return AgentTurn(assistant_message={"content": "done"})

            def verifier(session: SandboxSession) -> VerifierResult:
                return VerifierResult(
                    status="passed",
                    output=(session.workspace / "solution.py").read_text(encoding="utf-8"),
                )

            result = runner.run(
                EpisodeTask(
                    episode_id="episode-1",
                    task_id="task-1",
                    prompt="write a solution",
                ),
                generate_turn=generate_turn,
                verifier=verifier,
            )

            self.assertEqual(result.record.status, "passed")
            self.assertEqual(calls, 2)
            self.assertTrue((root / "episodes" / "episodes.jsonl").exists())
            self.assertTrue((root / "episodes" / "artifacts" / "episode-1" / "final.diff").exists())
            session = backend.session
            self.assertIsNotNone(session)
            if session is None:
                raise AssertionError("Expected backend to keep the fake session.")
            self.assertTrue(session.cleaned)


if __name__ == "__main__":
    unittest.main()
