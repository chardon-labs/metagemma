from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from sandbox_harness.backends import SandboxBackend, SandboxSession
from sandbox_harness.episodes import EpisodeRecord, EpisodeRecorder, EpisodeStatus
from sandbox_harness.tools import OpenAIFunctionTool, ToolExecutor, pi_function_tools
from sandbox_harness.traces import AssistantTurnTrace
from sandbox_harness.types import JsonObject, JsonValue, json_object, json_string


@dataclass(frozen=True)
class EpisodeTask:
    episode_id: str
    task_id: str
    prompt: str
    initial_files: Mapping[str, str] = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTurn:
    assistant_message: JsonObject
    trace: AssistantTurnTrace | None = None


@dataclass(frozen=True)
class VerifierResult:
    status: EpisodeStatus
    output: str


@dataclass(frozen=True)
class EpisodeRunResult:
    record: EpisodeRecord
    messages: list[JsonObject]


class GenerateAgentTurn(Protocol):
    def __call__(self, messages: Sequence[JsonObject], tools: Sequence[OpenAIFunctionTool]) -> AgentTurn:
        pass


class VerifyEpisode(Protocol):
    def __call__(self, session: SandboxSession) -> VerifierResult:
        pass


class EpisodeRunner:
    def __init__(
        self,
        *,
        backend: SandboxBackend,
        output_dir: str,
        max_turns: int = 20,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1.")
        self.backend = backend
        self.output_dir = output_dir
        self.max_turns = max_turns

    def run(
        self,
        task: EpisodeTask,
        *,
        generate_turn: GenerateAgentTurn,
        verifier: VerifyEpisode | None = None,
    ) -> EpisodeRunResult:
        session = self.backend.create_session(initial_files=task.initial_files, metadata=task.metadata)
        try:
            return self._run_with_session(task, session, generate_turn=generate_turn, verifier=verifier)
        finally:
            session.cleanup()

    def _run_with_session(
        self,
        task: EpisodeTask,
        session: SandboxSession,
        *,
        generate_turn: GenerateAgentTurn,
        verifier: VerifyEpisode | None,
    ) -> EpisodeRunResult:
        messages: list[JsonObject] = [{"role": "user", "content": task.prompt}]
        recorder = EpisodeRecorder(
            output_dir=Path(self.output_dir),
            episode_id=task.episode_id,
            task_id=task.task_id,
            workspace=session.workspace,
            metadata=task.metadata,
        )
        recorder.append_message(messages[0])
        executor = ToolExecutor(session)
        tools = pi_function_tools()
        status: EpisodeStatus = "max_turns"

        for _turn_index in range(self.max_turns):
            turn = generate_turn(messages, tools)
            assistant_message = dict(turn.assistant_message)
            assistant_message["role"] = "assistant"
            messages.append(assistant_message)
            recorder.append_message(assistant_message)
            if turn.trace is not None:
                recorder.append_turn_trace(turn.trace)

            tool_calls = _tool_calls(assistant_message)
            if not tool_calls:
                status = "completed"
                break

            for tool_call in tool_calls:
                tool_call_id, name, arguments = _parse_tool_call(tool_call)
                result = executor.execute(tool_call_id, name, arguments)
                tool_message = result.to_openai_message()
                messages.append(tool_message)
                recorder.append_message(tool_message)

        verifier_output = ""
        if verifier is not None:
            verifier_result = verifier(session)
            status = verifier_result.status
            verifier_output = verifier_result.output

        record = recorder.finish(status=status, verifier_output=verifier_output)
        return EpisodeRunResult(record=record, messages=messages)


def _tool_calls(assistant_message: Mapping[str, JsonValue]) -> list[JsonObject]:
    value = assistant_message.get("tool_calls")
    if not isinstance(value, list):
        return []
    calls: list[JsonObject] = []
    for item in value:
        calls.append(json_object(item))
    return calls


def _parse_tool_call(tool_call: Mapping[str, JsonValue]) -> tuple[str, str, JsonObject]:
    tool_call_id = json_string(tool_call.get("id"), name="tool_call.id")
    function = json_object(tool_call.get("function"))
    name = json_string(function.get("name"), name="tool_call.function.name")
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, str):
        arguments = cast(JsonValue, json.loads(raw_arguments))
        return tool_call_id, name, json_object(arguments)
    return tool_call_id, name, json_object(raw_arguments)
