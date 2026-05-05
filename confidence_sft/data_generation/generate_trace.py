from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
    apply_chat_template,
    configure_logging,
    fixed_top_logprobs,
    format_prompt,
    load_manifest,
    prompt_token_ids,
    read_trace_metadata,
    verify_confidence_token,
    write_manifest,
    write_trace_shard,
)
from dataset_specs import dataset_manifest_entries, prepare_problem_splits
from scorers import ScoreResult, score_completion


LOGGER = logging.getLogger(__name__)


MODEL_ID = DEFAULT_MODEL_ID
TOKENIZER_ID: str | None = DEFAULT_TOKENIZER_ID
OUTPUT_DIR = Path(DEFAULT_TRACE_DIR)
SEED = 44
NUM_GENERATIONS = 1
SFT_THINKING_FRACTION_NUMERATOR = 2
SFT_THINKING_FRACTION_DENOMINATOR = 3
MAX_SEQUENCE_LENGTH = 32768
ENABLE_THINKING = True
LOGPROBS_K = 20
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.0
GPU_MEMORY_UTILIZATION = 0.85
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
ProblemKey = tuple[str, str, str, str, str, str]
DTYPE: ModelDType = "auto"
TENSOR_PARALLEL_SIZE = 1
SHARD_SIZE = 512
SKIP_EVAL_GENERATION = False
APPEND = False


@dataclass(frozen=True)
class GenerateTraceConfig:
    model_id: str = MODEL_ID
    tokenizer_id: str | None = TOKENIZER_ID
    output_dir: Path = OUTPUT_DIR
    seed: int = SEED
    num_generations: int = NUM_GENERATIONS
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    enable_thinking: bool = ENABLE_THINKING
    logprobs_k: int = LOGPROBS_K
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    repetition_penalty: float = REPETITION_PENALTY
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    dtype: ModelDType = DTYPE
    tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
    shard_size: int = SHARD_SIZE
    skip_eval_generation: bool = SKIP_EVAL_GENERATION
    append: bool = APPEND


def hit_generation_token_limit(
    finish_reason: str | None,
    *,
    token_count: int,
    completion_token_budget: int,
) -> bool:
    if finish_reason == "length":
        return True
    return finish_reason is None and token_count >= completion_token_budget


def make_sampling_params(config: GenerateTraceConfig, *, completion_token_budget: int) -> SamplingParams:
    return SamplingParams(
        n=config.num_generations,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        max_tokens=completion_token_budget,
        logprobs=config.logprobs_k,
        seed=config.seed,
        logit_bias={CONFIDENCE_TOKEN_ID: -100.0},
    )


@dataclass(frozen=True)
class TracePromptRequest:
    request_index: int
    request_id: str
    split: str
    problem: dict[str, Any]
    prompt_text: str
    prompt_ids: np.ndarray
    completion_token_budget: int
    enable_thinking: bool


@dataclass(frozen=True)
class TraceRequestResult:
    request: TracePromptRequest
    output: Any


class TraceShardBuilder:
    def __init__(
        self,
        *,
        output_dir: Path,
        shard_size: int,
        max_sequence_length: int,
        initial_shard_index: int = 0,
        initial_row_id: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.max_sequence_length = max_sequence_length
        self.shard_index = initial_shard_index
        self.row_id = initial_row_id
        self.completion_token_offset = 0
        self.prompt_token_offset = 0
        self.rows: list[dict[str, Any]] = []
        self.prompt_token_arrays: list[np.ndarray] = []
        self.completion_token_arrays: list[np.ndarray] = []
        self.top_token_id_arrays: list[np.ndarray] = []
        self.top_logprob_arrays: list[np.ndarray] = []
        self.top_mask_arrays: list[np.ndarray] = []
        self.shards: list[dict[str, str]] = []

    def add_completion(
        self,
        *,
        split: str,
        problem: Mapping[str, Any],
        prompt_text: str,
        prompt_ids: np.ndarray,
        sample_id: int,
        completion_text: str,
        completion_token_ids: np.ndarray,
        top_token_ids: np.ndarray,
        top_logprobs: np.ndarray,
        top_mask: np.ndarray,
        finish_reason: str | None,
        stop_reason: int | str | None,
        generation_config: Mapping[str, Any],
        force_incorrect: bool = False,
    ) -> None:
        if completion_token_ids.size == 0:
            LOGGER.warning("Skipping empty completion for problem_id=%s sample_id=%s", problem["problem_id"], sample_id)
            return

        token_length = int(completion_token_ids.shape[0])
        prompt_length = int(prompt_ids.shape[0])
        total_length = prompt_length + token_length
        if total_length > self.max_sequence_length:
            LOGGER.warning(
                "Skipping over-length completion for problem_id=%s sample_id=%s total_length=%s max_sequence_length=%s",
                problem["problem_id"],
                sample_id,
                total_length,
                self.max_sequence_length,
            )
            return
        if top_token_ids.shape[0] != token_length:
            raise ValueError(f"top_token_ids length {top_token_ids.shape[0]} != token length {token_length}")

        gold_answer = str(problem["gold_answer"])
        if force_incorrect:
            score = ScoreResult(
                label=0,
                extracted_prediction="",
                normalized_prediction="",
                normalized_gold=gold_answer,
                scorer=str(problem["scorer"]),
                score_error="Generation hit token limit.",
            )
        else:
            score = score_completion(completion_text, problem)
        row = {
            "row_id": self.row_id,
            "problem_id": int(problem["problem_id"]),
            "split": split,
            "source_dataset": str(problem["source_dataset"]),
            "source_config": None if problem["source_config"] is None else str(problem["source_config"]),
            "source_split": str(problem["source_split"]),
            "source_id": str(problem["source_id"]),
            "task_type": str(problem["task_type"]),
            "scorer": score.scorer,
            "question": str(problem["question"]),
            "gold_answer": gold_answer,
            "choices": list(problem["choices"]),
            "choice_labels": list(problem["choice_labels"]),
            "prompt_text": prompt_text,
            "sample_id": sample_id,
            "completion_text": completion_text,
            "finish_reason": finish_reason,
            "stop_reason": None if stop_reason is None else str(stop_reason),
            "math_verify_label": score.label,
            "verifier_label": score.label,
            "extracted_prediction": score.extracted_prediction,
            "normalized_prediction": score.normalized_prediction,
            "normalized_gold": score.normalized_gold,
            "score_error": score.score_error,
            "prompt_token_start": self.prompt_token_offset,
            "prompt_token_length": prompt_length,
            "token_start": self.completion_token_offset,
            "token_length": token_length,
            "total_token_length": total_length,
            **dict(generation_config),
        }
        self.rows.append(row)
        self.prompt_token_arrays.append(prompt_ids)
        self.completion_token_arrays.append(completion_token_ids)
        self.top_token_id_arrays.append(top_token_ids)
        self.top_logprob_arrays.append(top_logprobs)
        self.top_mask_arrays.append(top_mask)

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
        self.shards.append(
            {
                "meta_path": paths.meta_path.name,
                "arrays_path": paths.arrays_path.name,
            }
        )
        self.shard_index += 1
        self.completion_token_offset = 0
        self.prompt_token_offset = 0
        self.rows = []
        self.prompt_token_arrays = []
        self.completion_token_arrays = []
        self.top_token_id_arrays = []
        self.top_logprob_arrays = []
        self.top_mask_arrays = []


def build_prompt_request(
    *,
    tokenizer: PreTrainedTokenizerBase,
    problem: Mapping[str, Any],
    enable_thinking: bool,
) -> tuple[dict[str, Any], str, np.ndarray]:
    messages = format_prompt(str(problem["user_prompt"]), system_prompt=str(problem["system_prompt"]))
    prompt_text = apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
    ids = prompt_token_ids(tokenizer, prompt_text)
    return dict(problem), prompt_text, ids


def completion_logprob_arrays(
    *,
    completion_logprobs: Any,
    token_count: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if completion_logprobs is None:
        raise ValueError("vLLM did not return logprobs; SamplingParams.logprobs must be set.")
    if len(completion_logprobs) != token_count:
        raise ValueError(f"vLLM returned {len(completion_logprobs)} logprob entries for {token_count} tokens.")

    token_ids: list[np.ndarray] = []
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for step_logprobs in completion_logprobs:
        step_ids, step_values, step_mask = fixed_top_logprobs(
            cast(Mapping[int, Any], step_logprobs),
            k=k,
            forbidden_token_id=CONFIDENCE_TOKEN_ID,
        )
        token_ids.append(step_ids)
        values.append(step_values)
        masks.append(step_mask)

    return np.stack(token_ids), np.stack(values), np.stack(masks)


def generation_config_payload(
    *,
    config: GenerateTraceConfig,
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id or config.model_id,
        "generation_seed": config.seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_sequence_length": config.max_sequence_length,
        "enable_thinking": enable_thinking,
        "logprobs_k": config.logprobs_k,
        "forbidden_token_id": CONFIDENCE_TOKEN_ID,
    }


def make_trace_prompt_request(
    *,
    tokenizer: PreTrainedTokenizerBase,
    config: GenerateTraceConfig,
    split: str,
    problem: Mapping[str, Any],
    enable_thinking: bool,
    request_index: int,
) -> TracePromptRequest | None:
    problem_record, prompt_text, ids = build_prompt_request(
        tokenizer=tokenizer,
        problem=problem,
        enable_thinking=enable_thinking,
    )
    prompt_length = int(ids.shape[0])
    completion_token_budget = config.max_sequence_length - prompt_length
    if completion_token_budget <= 0:
        LOGGER.warning(
            "Skipping over-length prompt for problem_id=%s prompt_length=%s max_sequence_length=%s",
            problem_record["problem_id"],
            prompt_length,
            config.max_sequence_length,
        )
        return None

    request_id = f"{split}-{problem_record['problem_id']}-{request_index}"
    return TracePromptRequest(
        request_index=request_index,
        request_id=request_id,
        split=split,
        problem=problem_record,
        prompt_text=prompt_text,
        prompt_ids=ids,
        completion_token_budget=completion_token_budget,
        enable_thinking=enable_thinking,
    )


async def generate_request(
    *,
    engine: AsyncLLMEngine,
    config: GenerateTraceConfig,
    request: TracePromptRequest,
) -> TraceRequestResult:
    final_output: Any | None = None
    async for request_output in engine.generate(
        request.prompt_text,
        make_sampling_params(config, completion_token_budget=request.completion_token_budget),
        request.request_id,
    ):
        final_output = request_output
    if final_output is None:
        raise RuntimeError(f"vLLM returned no output for request_id={request.request_id}")
    return TraceRequestResult(request=request, output=final_output)


def record_request_result(
    *,
    builder: TraceShardBuilder,
    config: GenerateTraceConfig,
    result: TraceRequestResult,
) -> None:
    request = result.request
    generation_config = generation_config_payload(config=config, enable_thinking=request.enable_thinking)
    outputs = sorted(result.output.outputs, key=lambda output: output.index)
    for completion in outputs:
        completion_ids = np.asarray(completion.token_ids, dtype=np.int32)
        if completion_ids.size == 0:
            LOGGER.warning(
                "Skipping empty completion for problem_id=%s sample_id=%s",
                request.problem["problem_id"],
                completion.index,
            )
            continue
        token_limited = hit_generation_token_limit(
            completion.finish_reason,
            token_count=int(completion_ids.shape[0]),
            completion_token_budget=request.completion_token_budget,
        )
        if token_limited:
            LOGGER.warning(
                "Marking token-limited completion incorrect for problem_id=%s sample_id=%s token_length=%s completion_token_budget=%s",
                request.problem["problem_id"],
                completion.index,
                int(completion_ids.shape[0]),
                request.completion_token_budget,
            )
        top_token_ids, top_logprobs, top_mask = completion_logprob_arrays(
            completion_logprobs=completion.logprobs,
            token_count=int(completion_ids.shape[0]),
            k=config.logprobs_k,
        )
        builder.add_completion(
            split=request.split,
            problem=request.problem,
            prompt_text=request.prompt_text,
            prompt_ids=request.prompt_ids,
            sample_id=int(completion.index),
            completion_text=completion.text,
            completion_token_ids=completion_ids,
            top_token_ids=top_token_ids,
            top_logprobs=top_logprobs,
            top_mask=top_mask,
            finish_reason=completion.finish_reason,
            stop_reason=completion.stop_reason,
            generation_config={
                **generation_config,
                "completion_token_budget": request.completion_token_budget,
            },
            force_incorrect=token_limited,
        )


async def record_completed_tasks(
    *,
    pending_tasks: set[asyncio.Task[TraceRequestResult]],
    builder: TraceShardBuilder,
    config: GenerateTraceConfig,
    wait_for_one: bool,
) -> int:
    if not pending_tasks:
        return 0

    done, pending = await asyncio.wait(
        pending_tasks,
        timeout=None if wait_for_one else 0,
        return_when=asyncio.FIRST_COMPLETED if wait_for_one else asyncio.ALL_COMPLETED,
    )
    pending_tasks.clear()
    pending_tasks.update(pending)
    for task in done:
        record_request_result(builder=builder, config=config, result=task.result())
    return len(done)


async def generate_all_requests(
    *,
    engine: AsyncLLMEngine,
    tokenizer: PreTrainedTokenizerBase,
    config: GenerateTraceConfig,
    builder: TraceShardBuilder,
    split_items: Sequence[tuple[str, Sequence[Mapping[str, Any]], bool]],
) -> None:
    pending_tasks: set[asyncio.Task[TraceRequestResult]] = set()
    next_request_index = 0
    queued = 0
    completed = 0
    for split, problems, enable_thinking in split_items:
        LOGGER.info(
            "Queueing %s traces for %s problems with enable_thinking=%s",
            split,
            len(problems),
            enable_thinking,
        )
        for problem in problems:
            request = make_trace_prompt_request(
                tokenizer=tokenizer,
                config=config,
                split=split,
                problem=problem,
                enable_thinking=enable_thinking,
                request_index=next_request_index,
            )
            if request is None:
                continue
            next_request_index += 1
            queued += 1
            pending_tasks.add(asyncio.create_task(generate_request(engine=engine, config=config, request=request)))
            if queued % 100 == 0:
                await asyncio.sleep(0)
                completed += await record_completed_tasks(
                    pending_tasks=pending_tasks,
                    builder=builder,
                    config=config,
                    wait_for_one=False,
                )
                if completed > 0 and completed % 100 == 0:
                    LOGGER.info("Completed %s/%s queued prompts.", completed, queued)

    LOGGER.info("Queued %s prompts to vLLM.", queued)
    while pending_tasks:
        completed += await record_completed_tasks(
            pending_tasks=pending_tasks,
            builder=builder,
            config=config,
            wait_for_one=True,
        )
        if completed % 100 == 0 or completed == queued:
            LOGGER.info("Completed %s/%s queued prompts.", completed, queued)


def make_async_engine(config: GenerateTraceConfig, *, tokenizer_id: str) -> AsyncLLMEngine:
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


async def run_generation_async(
    *,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_id: str,
    config: GenerateTraceConfig,
    builder: TraceShardBuilder,
    split_items: Sequence[tuple[str, Sequence[Mapping[str, Any]], bool]],
) -> None:
    engine = make_async_engine(config, tokenizer_id=tokenizer_id)
    try:
        await generate_all_requests(
            engine=engine,
            tokenizer=tokenizer,
            config=config,
            builder=builder,
            split_items=split_items,
        )
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        else:
            shutdown = getattr(engine, "shutdown_background_loop", None)
            if callable(shutdown):
                shutdown()


def split_sft_thinking_groups(
    problems: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for problem in problems:
        by_dataset.setdefault(str(problem["source_dataset"]), []).append(problem)

    thinking_problems: list[Mapping[str, Any]] = []
    no_thinking_problems: list[Mapping[str, Any]] = []
    for dataset_problems in by_dataset.values():
        thinking_count = len(dataset_problems) * SFT_THINKING_FRACTION_NUMERATOR // SFT_THINKING_FRACTION_DENOMINATOR
        thinking_problems.extend(dataset_problems[:thinking_count])
        no_thinking_problems.extend(dataset_problems[thinking_count:])
    return thinking_problems, no_thinking_problems


def parse_config() -> GenerateTraceConfig:
    parser = argparse.ArgumentParser(description="Generate confidence trace shards.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append only new dataset problems to the existing output snapshot.",
    )
    args = parser.parse_args()
    return GenerateTraceConfig(append=bool(args.append))


def problem_key(split: str, problem: Mapping[str, Any]) -> ProblemKey:
    source_config = "" if problem["source_config"] is None else str(problem["source_config"])
    return (
        split,
        str(problem["source_dataset"]),
        source_config,
        str(problem["source_split"]),
        str(problem["source_id"]),
        str(problem["task_type"]),
    )


def row_problem_key(row: Mapping[str, Any]) -> ProblemKey:
    source_config = "" if row["source_config"] is None else str(row["source_config"])
    return (
        str(row["split"]),
        str(row["source_dataset"]),
        source_config,
        str(row["source_split"]),
        str(row["source_id"]),
        str(row["task_type"]),
    )


def next_shard_index(shards: Sequence[Mapping[str, str]]) -> int:
    highest = -1
    for shard in shards:
        stem = Path(shard["meta_path"]).name.split(".", maxsplit=1)[0]
        prefix, _, suffix = stem.rpartition("-")
        if prefix == "trace" and suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


@dataclass(frozen=True)
class AppendState:
    manifest: TraceManifest
    existing_problem_keys: set[ProblemKey]
    next_row_id: int
    next_problem_id: int
    next_shard_index: int


def load_append_state(output_dir: Path) -> AppendState:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"--append requires an existing manifest at {manifest_path}")

    manifest = load_manifest(output_dir)
    existing_problem_keys: set[ProblemKey] = set()
    max_row_id = -1
    max_problem_id = -1
    for shard in manifest.shards:
        for row in read_trace_metadata(output_dir / shard["meta_path"]):
            existing_problem_keys.add(row_problem_key(row))
            max_row_id = max(max_row_id, int(row["row_id"]))
            max_problem_id = max(max_problem_id, int(row["problem_id"]))

    return AppendState(
        manifest=manifest,
        existing_problem_keys=existing_problem_keys,
        next_row_id=max_row_id + 1,
        next_problem_id=max_problem_id + 1,
        next_shard_index=next_shard_index(manifest.shards),
    )


def filter_append_problems(
    *,
    split: str,
    problems: Sequence[Mapping[str, Any]],
    existing_problem_keys: set[ProblemKey],
    first_problem_id: int,
) -> tuple[list[Mapping[str, Any]], int]:
    next_problem_id = first_problem_id
    new_problems: list[Mapping[str, Any]] = []
    for problem in problems:
        key = problem_key(split, problem)
        if key in existing_problem_keys:
            continue
        appended_problem = dict(problem)
        appended_problem["problem_id"] = next_problem_id
        next_problem_id += 1
        existing_problem_keys.add(key)
        new_problems.append(appended_problem)
    return new_problems, next_problem_id


def main() -> None:
    configure_logging()
    config = parse_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_id = config.tokenizer_id or config.model_id
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True),
    )
    verify_confidence_token(tokenizer)

    sft_problems, eval_problems = prepare_problem_splits(seed=config.seed)
    existing_shards: list[dict[str, str]] = []
    initial_row_id = 0
    initial_shard_index = 0
    previous_sft_problem_count = 0
    previous_eval_problem_count = 0
    if config.append:
        append_state = load_append_state(config.output_dir)
        previous_sft_problem_count = append_state.manifest.sft_problem_count
        previous_eval_problem_count = append_state.manifest.eval_problem_count
        existing_shards = append_state.manifest.shards
        initial_row_id = append_state.next_row_id
        initial_shard_index = append_state.next_shard_index
        sft_problems, next_problem_id = filter_append_problems(
            split="sft",
            problems=sft_problems,
            existing_problem_keys=append_state.existing_problem_keys,
            first_problem_id=append_state.next_problem_id,
        )
        eval_problems, _ = filter_append_problems(
            split="eval",
            problems=eval_problems,
            existing_problem_keys=append_state.existing_problem_keys,
            first_problem_id=next_problem_id,
        )
        LOGGER.info(
            "Append mode found %s new sft problems and %s new eval problems",
            len(sft_problems),
            0 if config.skip_eval_generation else len(eval_problems),
        )

    if not sft_problems and (config.skip_eval_generation or not eval_problems):
        LOGGER.info("No new problems to append.")
        return

    sft_thinking_problems, sft_no_thinking_problems = split_sft_thinking_groups(sft_problems)
    split_items: list[tuple[str, Sequence[Mapping[str, Any]], bool]] = [
        ("sft", sft_thinking_problems, True),
        ("sft", sft_no_thinking_problems, False),
    ]
    if not config.skip_eval_generation:
        split_items.append(("eval", eval_problems, config.enable_thinking))

    builder = TraceShardBuilder(
        output_dir=config.output_dir,
        shard_size=config.shard_size,
        max_sequence_length=config.max_sequence_length,
        initial_shard_index=initial_shard_index,
        initial_row_id=initial_row_id,
    )

    asyncio.run(
        run_generation_async(
            tokenizer=tokenizer,
            tokenizer_id=tokenizer_id,
            config=config,
            builder=builder,
            split_items=split_items,
        )
    )

    builder.flush()
    write_manifest(
        config.output_dir,
        TraceManifest(
            model_id=config.model_id,
            tokenizer_id=tokenizer_id,
            dataset="mixed",
            dataset_config="multi",
            seed=config.seed,
            sft_problem_count=previous_sft_problem_count + len(sft_problems),
            eval_problem_count=previous_eval_problem_count
            + (0 if config.skip_eval_generation else len(eval_problems)),
            num_generations=config.num_generations,
            max_sequence_length=config.max_sequence_length,
            enable_thinking=None,
            logprobs_k=config.logprobs_k,
            forbidden_token_id=CONFIDENCE_TOKEN_ID,
            shards=[*existing_shards, *builder.shards],
            datasets=dataset_manifest_entries(),
        ),
    )


if __name__ == "__main__":
    main()
