from __future__ import annotations

import difflib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sandbox_harness.traces import AssistantTurnTrace
from sandbox_harness.types import JsonObject, JsonValue

type EpisodeStatus = str


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    status: EpisodeStatus
    task_id: str
    workspace_path: str
    artifact_dir: str
    started_at: str
    finished_at: str | None = None
    metadata: JsonObject = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_relative_files(workspace: Path) -> list[Path]:
    paths: list[Path] = []
    for path in workspace.rglob("*"):
        if path.is_file():
            paths.append(path.relative_to(workspace))
    return sorted(paths)


def _snapshot(workspace: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for relative_path in _safe_relative_files(workspace):
        try:
            files[relative_path.as_posix()] = (workspace / relative_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def _unified_diff(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    chunks: list[str] = []
    for path in sorted(set(before) | set(after)):
        old = before.get(path, "")
        new = after.get(path, "")
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    return "".join(chunks)


class EpisodeRecorder:
    def __init__(
        self,
        *,
        output_dir: Path,
        episode_id: str,
        task_id: str,
        workspace: Path,
        metadata: JsonObject | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.episode_id = episode_id
        self.task_id = task_id
        self.workspace = workspace
        self.artifact_dir = output_dir / "artifacts" / episode_id
        self.metadata = {} if metadata is None else dict(metadata)
        self.started_at = _utc_now()
        self._initial_snapshot = _snapshot(workspace)
        self._messages: list[JsonObject] = []
        self._turn_traces: list[AssistantTurnTrace] = []

    def append_message(self, message: Mapping[str, JsonValue]) -> None:
        self._messages.append(dict(message))

    def append_turn_trace(self, trace: AssistantTurnTrace) -> None:
        self._turn_traces.append(trace)

    def finish(
        self,
        *,
        status: EpisodeStatus,
        verifier_output: str = "",
        extra_artifacts: Mapping[str, str] | None = None,
    ) -> EpisodeRecord:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        finished_at = _utc_now()
        final_snapshot = _snapshot(self.workspace)
        (self.artifact_dir / "final.diff").write_text(_unified_diff(self._initial_snapshot, final_snapshot), encoding="utf-8")
        (self.artifact_dir / "verifier_output.txt").write_text(verifier_output, encoding="utf-8")
        (self.artifact_dir / "transcript.json").write_text(
            json.dumps(self._messages, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.artifact_dir / "turn_traces.json").write_text(
            json.dumps([asdict(trace) for trace in self._turn_traces], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if extra_artifacts is not None:
            for name, content in extra_artifacts.items():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"Artifact name must stay inside artifact dir: {name}")
                target = self.artifact_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

        record = EpisodeRecord(
            episode_id=self.episode_id,
            status=status,
            task_id=self.task_id,
            workspace_path=str(self.workspace),
            artifact_dir=str(self.artifact_dir),
            started_at=self.started_at,
            finished_at=finished_at,
            metadata=self.metadata,
        )
        with (self.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True))
            handle.write("\n")
        summary = self._summary(record)
        with (self.output_dir / "transcripts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True))
            handle.write("\n")
        return record

    def _summary(self, record: EpisodeRecord) -> JsonObject:
        return {
            "episode_id": record.episode_id,
            "task_id": record.task_id,
            "status": record.status,
            "message_count": len(self._messages),
            "turn_trace_count": len(self._turn_traces),
            "artifact_dir": record.artifact_dir,
        }
