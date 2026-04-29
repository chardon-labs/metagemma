from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from confidence_trace import (
    CONFIDENCE_TOKEN_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TRACE_DIR,
    TraceManifest,
    configure_logging,
    fixed_top_logprobs,
    prompt_token_ids,
    verify_confidence_token,
    write_manifest,
    write_trace_shard,
)
from sandbox_harness.backends import (
    BubblewrapBackend,
    ProotBackend,
    SandboxBackend,
    SandboxConfig,
    SandboxSession,
)
from sandbox_harness.bootstrap import PythonRepoTaskSpec, python_repo_initial_files, toy_addition_task
from sandbox_harness.episodes import EpisodeRecorder, EpisodeStatus
from sandbox_harness.prompt import PiPromptPaths, build_pi_system_prompt
from sandbox_harness.serialization import GemmaChatTokenizer, render_gemma_chat
from sandbox_harness.tools import OpenAIFunctionTool, ToolExecutor, ToolResult, pi_function_tools
from sandbox_harness.traces import AssistantTurnTrace, TokenLogprob
from sandbox_harness.types import JsonObject, JsonValue


LOGGER = logging.getLogger(__name__)


MODEL_ID = DEFAULT_MODEL_ID
TOKENIZER_ID: str | None = DEFAULT_TOKENIZER_ID
OUTPUT_DIR = Path(DEFAULT_TRACE_DIR).parent / "coding-agent-traces"
SEED = 44
MAX_SEQUENCE_LENGTH = 32768
ENABLE_THINKING = True
LOGPROBS_K = 20
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = -1
REPETITION_PENALTY = 1.0
GPU_MEMORY_UTILIZATION = 0.85
TENSOR_PARALLEL_SIZE = 1
SHARD_SIZE = 256
MAX_TURNS = 20
COMMAND_TIMEOUT_SECONDS = 30
COMMAND_CPU_SECONDS = 30
COMMAND_MEMORY_BYTES: int | None = None
COMMAND_FILE_SIZE_BYTES = 256 * 1024 * 1024
COMMAND_PROCESS_COUNT = 256
TASK_SPLIT = "sft"

ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
DTYPE: ModelDType = "auto"


@dataclass(frozen=True)
class CodingTraceConfig:
    model_id: str = MODEL_ID
    tokenizer_id: str | None = TOKENIZER_ID
    output_dir: Path = OUTPUT_DIR
    seed: int = SEED
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    enable_thinking: bool = ENABLE_THINKING
    logprobs_k: int = LOGPROBS_K
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    repetition_penalty: float = REPETITION_PENALTY
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
    dtype: ModelDType = DTYPE
    shard_size: int = SHARD_SIZE
    max_turns: int = MAX_TURNS
    command_timeout_seconds: int = COMMAND_TIMEOUT_SECONDS
    command_cpu_seconds: int = COMMAND_CPU_SECONDS
    command_memory_bytes: int | None = COMMAND_MEMORY_BYTES
    command_file_size_bytes: int = COMMAND_FILE_SIZE_BYTES
    command_process_count: int = COMMAND_PROCESS_COUNT


@dataclass(frozen=True)
class RecordedCodingTurn:
    episode_id: str
    task_id: str
    problem_id: int
    turn_index: int
    message_index: int
    prompt_text: str
    prompt_token_ids: np.ndarray
    completion_text: str
    completion_token_ids: np.ndarray
    top_token_ids: np.ndarray
    top_logprobs: np.ndarray
    top_mask: np.ndarray
    finish_reason: str | None
    status: EpisodeStatus
    verifier_label: int
    verifier_output: str
    split: str


class CodingTraceShardBuilder:
    def __init__(
        self,
        *,
        output_dir: Path,
        shard_size: int,
    ) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_index = 0
        self.row_id = 0
        self.prompt_token_offset = 0
        self.completion_token_offset = 0
        self.rows: list[dict[str, JsonValue]] = []
        self.prompt_token_arrays: list[np.ndarray] = []
        self.completion_token_arrays: list[np.ndarray] = []
        self.top_token_id_arrays: list[np.ndarray] = []
        self.top_logprob_arrays: list[np.ndarray] = []
        self.top_mask_arrays: list[np.ndarray] = []
        self.shards: list[dict[str, str]] = []

    def add_turn(self, turn: RecordedCodingTurn, *, config: CodingTraceConfig) -> None:
        prompt_length = int(turn.prompt_token_ids.shape[0])
        token_length = int(turn.completion_token_ids.shape[0])
        if token_length == 0:
            LOGGER.warning("Skipping empty coding turn episode_id=%s turn_index=%s", turn.episode_id, turn.turn_index)
            return
        if prompt_length + token_length > config.max_sequence_length:
            LOGGER.warning(
                "Skipping over-length coding turn episode_id=%s turn_index=%s total_length=%s max_sequence_length=%s",
                turn.episode_id,
                turn.turn_index,
                prompt_length + token_length,
                config.max_sequence_length,
            )
            return

        row: dict[str, JsonValue] = {
            "row_id": self.row_id,
            "problem_id": turn.problem_id,
            "split": turn.split,
            "source_dataset": "sandbox_harness",
            "source_config": "coding_agent",
            "source_split": turn.split,
            "source_id": turn.task_id,
            "task_type": "coding_agent",
            "scorer": "hidden_tests",
            "question": turn.task_id,
            "gold_answer": "passed",
            "choices": [],
            "choice_labels": [],
            "prompt_text": turn.prompt_text,
            "sample_id": turn.turn_index,
            "completion_text": turn.completion_text,
            "finish_reason": turn.finish_reason,
            "stop_reason": None,
            "math_verify_label": turn.verifier_label,
            "verifier_label": turn.verifier_label,
            "extracted_prediction": turn.status,
            "normalized_prediction": turn.status,
            "normalized_gold": "passed",
            "score_error": None if turn.verifier_label == 1 else turn.verifier_output,
            "prompt_token_start": self.prompt_token_offset,
            "prompt_token_length": prompt_length,
            "token_start": self.completion_token_offset,
            "token_length": token_length,
            "total_token_length": prompt_length + token_length,
            "episode_id": turn.episode_id,
            "task_id": turn.task_id,
            "turn_index": turn.turn_index,
            "message_index": turn.message_index,
            "episode_status": turn.status,
            "max_sequence_length": config.max_sequence_length,
            "completion_token_budget": config.max_sequence_length - prompt_length,
            "enable_thinking": config.enable_thinking,
            "model_id": config.model_id,
            "tokenizer_id": config.tokenizer_id or config.model_id,
            "generation_seed": config.seed,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "logprobs_k": config.logprobs_k,
            "forbidden_token_id": CONFIDENCE_TOKEN_ID,
        }
        self.rows.append(row)
        self.prompt_token_arrays.append(turn.prompt_token_ids)
        self.completion_token_arrays.append(turn.completion_token_ids)
        self.top_token_id_arrays.append(turn.top_token_ids)
        self.top_logprob_arrays.append(turn.top_logprobs)
        self.top_mask_arrays.append(turn.top_mask)
        self.row_id += 1
        self.prompt_token_offset += prompt_length
        self.completion_token_offset += token_length
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        paths = write_trace_shard(
            output_dir=self.output_dir,
            shard_index=self.shard_index,
            rows=self.rows,
            prompt_token_arrays=self.prompt_token_arrays,
            completion_token_arrays=self.completion_token_arrays,
            top_token_id_arrays=self.top_token_id_arrays,
            top_logprob_arrays=self.top_logprob_arrays,
            top_mask_arrays=self.top_mask_arrays,
        )
        self.shards.append({"meta_path": paths.meta_path.name, "arrays_path": paths.arrays_path.name})
        self.shard_index += 1
        self.prompt_token_offset = 0
        self.completion_token_offset = 0
        self.rows = []
        self.prompt_token_arrays = []
        self.completion_token_arrays = []
        self.top_token_id_arrays = []
        self.top_logprob_arrays = []
        self.top_mask_arrays = []


def coding_task_specs() -> list[PythonRepoTaskSpec]:
    return [toy_addition_task()]


def sandbox_config(config: CodingTraceConfig) -> SandboxConfig:
    from sandbox_harness.backends import CommandLimits

    return SandboxConfig(
        timeout_seconds=config.command_timeout_seconds,
        limits=CommandLimits(
            cpu_seconds=config.command_cpu_seconds,
            memory_bytes=config.command_memory_bytes,
            file_size_bytes=config.command_file_size_bytes,
            process_count=config.command_process_count,
        ),
    )


def command_probe(command: Sequence[str]) -> str:
    executable = command[0]
    resolved = shutil.which(executable)
    if resolved is None:
        return f"{executable}: not found on PATH"
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    output = "".join(piece for piece in (completed.stdout, completed.stderr) if piece).strip()
    if not output:
        output = "(no output)"
    return f"{executable}: exit_code={completed.returncode} path={resolved}\n{output}"


def backend_probe_diagnostics() -> str:
    return "\n\n".join(
        [
            command_probe(["bwrap", "--die-with-parent", "--unshare-user", "--unshare-pid", "--ro-bind", "/", "/", "true"]),
            command_probe(["proot", "-r", "/", "true"]),
        ]
    )


def select_backend(config: CodingTraceConfig) -> SandboxBackend:
    selected_config = sandbox_config(config)
    for backend in (BubblewrapBackend(config=selected_config), ProotBackend(config=selected_config)):
        if backend.is_supported():
            LOGGER.info("Using sandbox backend: %s", backend.name)
            return backend
    raise RuntimeError(
        "No supported sandbox backend found. The sandbox requires bubblewrap or proot to be installed "
        "and permitted to create the required namespaces/mounts in this container.\n\n"
        f"{backend_probe_diagnostics()}"
    )


def system_prompt() -> str:
    return build_pi_system_prompt(
        cwd="/workspace",
        current_date=date.today(),
        paths=PiPromptPaths(
            readme_path="/workspace/README.md",
            docs_path="/workspace/docs",
            examples_path="/workspace/examples",
        ),
        selected_tools=("read", "bash", "edit", "write"),
    )


def task_prompt(spec: PythonRepoTaskSpec) -> str:
    return (
        f"{spec.prompt.strip()}\n\n"
        "Use the available tools to inspect the repository and make the required code changes. "
        "Run the visible tests before giving your final answer."
    )


def make_sampling_params(config: CodingTraceConfig, *, completion_token_budget: int) -> SamplingParams:
    return SamplingParams(
        n=1,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        max_tokens=completion_token_budget,
        logprobs=config.logprobs_k,
        seed=config.seed,
        logit_bias={CONFIDENCE_TOKEN_ID: -100.0},
        skip_special_tokens=False,
    )


def parsed_assistant_message(
    *,
    tokenizer: PreTrainedTokenizerBase,
    completion_text: str,
    turn_index: int,
) -> JsonObject:
    try:
        parsed = cast(JsonObject, tokenizer.parse_response(completion_text))
    except AttributeError as exc:
        raise RuntimeError(
            "Tokenizer does not provide response_schema/parse_response for native tool-call parsing."
        ) from exc

    message: JsonObject = dict(parsed)
    message["role"] = "assistant"
    if "thinking" in message and "reasoning" not in message:
        message["reasoning"] = message["thinking"]

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        normalized_tool_calls: list[JsonValue] = []
        for call_index, tool_call_value in enumerate(tool_calls):
            if not isinstance(tool_call_value, dict):
                raise TypeError("Parsed tool_calls must contain objects.")
            tool_call = dict(tool_call_value)
            tool_call.setdefault("id", f"call_{turn_index}_{call_index}")
            tool_call.setdefault("type", "function")
            normalized_tool_calls.append(tool_call)
        message["tool_calls"] = normalized_tool_calls
        message.setdefault("content", "")
    else:
        message.setdefault("content", completion_text)
    return message


def completion_logprob_arrays(
    *,
    completion_logprobs: Any,
    token_count: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[TokenLogprob]]]:
    if completion_logprobs is None:
        raise ValueError("vLLM did not return logprobs; SamplingParams.logprobs must be set.")
    if len(completion_logprobs) != token_count:
        raise ValueError(f"vLLM returned {len(completion_logprobs)} logprob entries for {token_count} tokens.")

    token_ids: list[np.ndarray] = []
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    trace_logprobs: list[list[TokenLogprob]] = []
    for step_logprobs in completion_logprobs:
        step_ids, step_values, step_mask = fixed_top_logprobs(
            cast(Mapping[int, Any], step_logprobs),
            k=k,
            forbidden_token_id=CONFIDENCE_TOKEN_ID,
        )
        token_ids.append(step_ids)
        values.append(step_values)
        masks.append(step_mask)
        trace_logprobs.append(
            [
                TokenLogprob(token_id=int(token_id), logprob=float(logprob))
                for token_id, logprob, mask in zip(step_ids, step_values, step_mask, strict=True)
                if bool(mask)
            ]
        )

    return np.stack(token_ids), np.stack(values), np.stack(masks), trace_logprobs


async def generate_next_turn(
    *,
    engine: AsyncLLMEngine,
    tokenizer: PreTrainedTokenizerBase,
    tools: Sequence[OpenAIFunctionTool],
    messages: Sequence[JsonObject],
    config: CodingTraceConfig,
    request_id: str,
) -> tuple[str, np.ndarray, Any]:
    prompt_text = render_gemma_chat(
        cast(GemmaChatTokenizer, tokenizer),
        messages,
        tools=tools,
        enable_thinking=config.enable_thinking,
        add_generation_prompt=True,
    )
    prompt_ids = prompt_token_ids(tokenizer, prompt_text)
    completion_token_budget = config.max_sequence_length - int(prompt_ids.shape[0])
    if completion_token_budget <= 0:
        raise RuntimeError(f"Prompt exceeds max_sequence_length for request_id={request_id}.")

    final_output: Any | None = None
    async for request_output in engine.generate(
        prompt_text,
        make_sampling_params(config, completion_token_budget=completion_token_budget),
        request_id,
    ):
        final_output = request_output
    if final_output is None:
        raise RuntimeError(f"vLLM returned no output for request_id={request_id}")
    return prompt_text, prompt_ids, final_output.outputs[0]


def verify_episode(session: SandboxSession, config: CodingTraceConfig) -> tuple[EpisodeStatus, str]:
    command = (
        "set -e\n"
        "cleanup_pytest_cache() {\n"
        "  rm -rf .pytest_cache\n"
        "}\n"
        "trap cleanup_pytest_cache EXIT\n"
        "if command -v python3 >/dev/null 2>&1; then\n"
        "  PYTHON=python3\n"
        "elif command -v python >/dev/null 2>&1; then\n"
        "  PYTHON=python\n"
        "else\n"
        "  echo 'Neither python3 nor python was found on PATH.'\n"
        "  echo \"PATH=$PATH\"\n"
        "  exit 127\n"
        "fi\n"
        '"$PYTHON" -m pytest tests hidden_tests -q'
    )
    result = session.run(
        ["bash", "-lc", command],
        cwd=PurePosixPath("/workspace"),
        timeout_seconds=config.command_timeout_seconds,
    )
    output = "".join(piece for piece in (result.stdout, result.stderr) if piece)
    if not output:
        output = f"[exit code {result.exit_code}]"
    return ("passed" if result.ok else "failed"), output


async def run_episode(
    *,
    engine: AsyncLLMEngine,
    tokenizer: PreTrainedTokenizerBase,
    backend: SandboxBackend,
    config: CodingTraceConfig,
    spec: PythonRepoTaskSpec,
    problem_id: int,
    builder: CodingTraceShardBuilder,
) -> None:
    episode_id = f"{TASK_SPLIT}-{problem_id:05d}-{spec.task_id}"
    session = backend.create_session(
        initial_files=python_repo_initial_files(spec),
        metadata={"task_id": spec.task_id, **spec.metadata},
    )
    recorder = EpisodeRecorder(
        output_dir=config.output_dir,
        episode_id=episode_id,
        task_id=spec.task_id,
        workspace=session.workspace,
        metadata={"task_id": spec.task_id, **spec.metadata},
    )
    messages: list[JsonObject] = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": task_prompt(spec)},
    ]
    for message in messages:
        recorder.append_message(message)

    executor = ToolExecutor(session)
    tools = pi_function_tools()
    pending_turns: list[RecordedCodingTurn] = []
    status: EpisodeStatus = "max_turns"
    verifier_output = ""
    try:
        for turn_index in range(config.max_turns):
            prompt_text, prompt_ids, completion = await generate_next_turn(
                engine=engine,
                tokenizer=tokenizer,
                tools=tools,
                messages=messages,
                config=config,
                request_id=f"{episode_id}-{turn_index}",
            )
            completion_ids = np.asarray(completion.token_ids, dtype=np.int32)
            top_token_ids, top_logprobs, top_mask, trace_logprobs = completion_logprob_arrays(
                completion_logprobs=completion.logprobs,
                token_count=int(completion_ids.shape[0]),
                k=config.logprobs_k,
            )
            completion_text = cast(
                str,
                tokenizer.decode([int(token_id) for token_id in completion_ids], skip_special_tokens=False),
            )
            message_index = len(messages)
            assistant = parsed_assistant_message(
                tokenizer=tokenizer,
                completion_text=completion_text,
                turn_index=turn_index,
            )
            messages.append(assistant)
            recorder.append_message(assistant)
            recorder.append_turn_trace(
                AssistantTurnTrace(
                    message_index=message_index,
                    prompt_text=prompt_text,
                    prompt_token_ids=[int(token_id) for token_id in prompt_ids],
                    completion_text=completion_text,
                    completion_token_ids=[int(token_id) for token_id in completion_ids],
                    top_logprobs=trace_logprobs,
                    finish_reason=completion.finish_reason,
                )
            )
            pending_turns.append(
                RecordedCodingTurn(
                    episode_id=episode_id,
                    task_id=spec.task_id,
                    problem_id=problem_id,
                    turn_index=turn_index,
                    message_index=message_index,
                    prompt_text=prompt_text,
                    prompt_token_ids=prompt_ids,
                    completion_text=completion_text,
                    completion_token_ids=completion_ids,
                    top_token_ids=top_token_ids,
                    top_logprobs=top_logprobs,
                    top_mask=top_mask,
                    finish_reason=completion.finish_reason,
                    status="max_turns",
                    verifier_label=0,
                    verifier_output="",
                    split=TASK_SPLIT,
                )
            )

            tool_calls = assistant.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                status = "completed"
                break

            for tool_call_value in tool_calls:
                if not isinstance(tool_call_value, dict):
                    raise TypeError("tool_calls must contain objects.")
                tool_call = cast(JsonObject, tool_call_value)
                tool_call_id = str(tool_call["id"])
                function = cast(JsonObject, tool_call["function"])
                tool_name = str(function["name"])
                tool_arguments = cast(JsonObject, function["arguments"])
                try:
                    result = await asyncio.to_thread(
                        executor.execute,
                        tool_call_id,
                        tool_name,
                        tool_arguments,
                    )
                    tool_message = result.to_openai_message()
                except Exception as exc:
                    result = ToolResult(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=str(exc),
                        is_error=True,
                    )
                    tool_message = result.to_openai_message()
                messages.append(tool_message)
                recorder.append_message(tool_message)

        status, verifier_output = await asyncio.to_thread(verify_episode, session, config)
        recorder.finish(status=status, verifier_output=verifier_output)
    finally:
        session.cleanup()

    verifier_label = 1 if status == "passed" else 0
    for turn in pending_turns:
        builder.add_turn(
            replace(
                turn,
                status=status,
                verifier_label=verifier_label,
                verifier_output=verifier_output,
            ),
            config=config,
        )


def make_async_engine(config: CodingTraceConfig, *, tokenizer_id: str) -> AsyncLLMEngine:
    return AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=config.model_id,
            tokenizer=tokenizer_id,
            trust_remote_code=True,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_sequence_length,
        )
    )


async def run_generation(config: CodingTraceConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    backend = select_backend(config)
    tokenizer_id = config.tokenizer_id or config.model_id
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True),
    )
    verify_confidence_token(tokenizer)

    engine = make_async_engine(config, tokenizer_id=tokenizer_id)
    builder = CodingTraceShardBuilder(output_dir=config.output_dir, shard_size=config.shard_size)
    tasks = coding_task_specs()
    try:
        episode_tasks = [
            asyncio.create_task(
                run_episode(
                    engine=engine,
                    tokenizer=tokenizer,
                    backend=backend,
                    config=config,
                    spec=spec,
                    problem_id=problem_id,
                    builder=builder,
                )
            )
            for problem_id, spec in enumerate(tasks)
        ]
        for task in asyncio.as_completed(episode_tasks):
            await task
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()

    builder.flush()
    write_manifest(
        config.output_dir,
        TraceManifest(
            model_id=config.model_id,
            tokenizer_id=tokenizer_id,
            dataset="sandbox_harness",
            dataset_config="coding_agent",
            seed=config.seed,
            sft_problem_count=len(tasks),
            eval_problem_count=0,
            num_generations=1,
            max_sequence_length=config.max_sequence_length,
            enable_thinking=config.enable_thinking,
            logprobs_k=config.logprobs_k,
            forbidden_token_id=CONFIDENCE_TOKEN_ID,
            shards=builder.shards,
            datasets=[
                {
                    "name": "sandbox_harness",
                    "config": "coding_agent",
                    "split": TASK_SPLIT,
                    "task_count": len(tasks),
                }
            ],
        ),
    )


def main() -> None:
    configure_logging()
    asyncio.run(run_generation(CodingTraceConfig()))


if __name__ == "__main__":
    main()
