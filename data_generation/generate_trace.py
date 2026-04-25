from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from confidence_trace import (
    CONFIDENCE_TOKEN_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TRACE_DIR,
    TraceManifest,
    apply_chat_template,
    configure_logging,
    extract_prediction,
    fixed_top_logprobs,
    format_prompt,
    math_verify_label,
    prepare_gsm8k_problem_splits,
    prompt_token_ids,
    verify_confidence_token,
    write_manifest,
    write_trace_shard,
)


LOGGER = logging.getLogger(__name__)


MODEL_ID = DEFAULT_MODEL_ID
TOKENIZER_ID: str | None = DEFAULT_TOKENIZER_ID
OUTPUT_DIR = Path(DEFAULT_TRACE_DIR)
SEED = 42
SFT_PROBLEMS = 250
EVAL_PROBLEMS = 50
NUM_GENERATIONS = 16
MAX_TOKENS = 512
LOGPROBS_K = 20
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.0
GPU_MEMORY_UTILIZATION = 0.85
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
DTYPE: ModelDType = "auto"
TENSOR_PARALLEL_SIZE = 1
MAX_PROMPTS_PER_GENERATE: int | None = None
SHARD_SIZE = 512
SKIP_EVAL_GENERATION = False


@dataclass(frozen=True)
class GenerateTraceConfig:
    model_id: str = MODEL_ID
    tokenizer_id: str | None = TOKENIZER_ID
    output_dir: Path = OUTPUT_DIR
    seed: int = SEED
    sft_problems: int = SFT_PROBLEMS
    eval_problems: int = EVAL_PROBLEMS
    num_generations: int = NUM_GENERATIONS
    max_tokens: int = MAX_TOKENS
    logprobs_k: int = LOGPROBS_K
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    repetition_penalty: float = REPETITION_PENALTY
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    dtype: ModelDType = DTYPE
    tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
    max_prompts_per_generate: int | None = MAX_PROMPTS_PER_GENERATE
    shard_size: int = SHARD_SIZE
    skip_eval_generation: bool = SKIP_EVAL_GENERATION


def prompt_batches(
    items: Sequence[Mapping[str, Any]],
    max_prompts_per_generate: int | None,
) -> Iterable[Sequence[Mapping[str, Any]]]:
    if max_prompts_per_generate is None:
        yield items
        return
    if max_prompts_per_generate <= 0:
        raise ValueError("max_prompts_per_generate must be positive when set.")

    for start in range(0, len(items), max_prompts_per_generate):
        yield items[start : start + max_prompts_per_generate]


def make_sampling_params(config: GenerateTraceConfig) -> SamplingParams:
    return SamplingParams(
        n=config.num_generations,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        max_tokens=config.max_tokens,
        logprobs=config.logprobs_k,
        seed=config.seed,
        logit_bias={CONFIDENCE_TOKEN_ID: -100.0},
    )


class TraceShardBuilder:
    def __init__(self, *, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_index = 0
        self.row_id = 0
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
    ) -> None:
        if completion_token_ids.size == 0:
            LOGGER.warning("Skipping empty completion for problem_id=%s sample_id=%s", problem["problem_id"], sample_id)
            return

        token_length = int(completion_token_ids.shape[0])
        prompt_length = int(prompt_ids.shape[0])
        if top_token_ids.shape[0] != token_length:
            raise ValueError(f"top_token_ids length {top_token_ids.shape[0]} != token length {token_length}")

        gold_answer = str(problem["gold_answer"])
        row = {
            "row_id": self.row_id,
            "problem_id": int(problem["problem_id"]),
            "split": split,
            "question": str(problem["question"]),
            "gold_answer": gold_answer,
            "prompt_text": prompt_text,
            "sample_id": sample_id,
            "completion_text": completion_text,
            "finish_reason": finish_reason,
            "stop_reason": None if stop_reason is None else str(stop_reason),
            "math_verify_label": math_verify_label(completion_text, gold_answer),
            "extracted_prediction": extract_prediction(completion_text),
            "prompt_token_start": self.prompt_token_offset,
            "prompt_token_length": prompt_length,
            "token_start": self.completion_token_offset,
            "token_length": token_length,
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


def build_prompts(
    *,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    problems: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[np.ndarray], list[dict[str, Any]]]:
    prompt_texts: list[str] = []
    prompt_ids: list[np.ndarray] = []
    prompt_records: list[dict[str, Any]] = []
    for problem in problems:
        messages = format_prompt(str(problem["question"]))
        prompt_text = apply_chat_template(tokenizer, messages)
        ids = prompt_token_ids(tokenizer, prompt_text)
        prompt_texts.append(prompt_text)
        prompt_ids.append(ids)
        prompt_records.append({"split": split, "problem": dict(problem), "prompt_text": prompt_text})
    return prompt_texts, prompt_ids, prompt_records


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


def generate_split(
    *,
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
    config: GenerateTraceConfig,
    builder: TraceShardBuilder,
    split: str,
    problems: Sequence[Mapping[str, Any]],
    sampling_params: SamplingParams,
) -> None:
    generation_config = {
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id or config.model_id,
        "generation_seed": config.seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "logprobs_k": config.logprobs_k,
        "forbidden_token_id": CONFIDENCE_TOKEN_ID,
    }

    if config.max_prompts_per_generate is None:
        LOGGER.info("Submitting all %s %s prompts to vLLM in one generate call", len(problems), split)
    else:
        LOGGER.info(
            "Submitting %s %s prompts to vLLM in chunks of %s",
            len(problems),
            split,
            config.max_prompts_per_generate,
        )

    for problem_batch in prompt_batches(problems, config.max_prompts_per_generate):
        prompt_texts, prompt_ids, prompt_records = build_prompts(
            tokenizer=tokenizer,
            split=split,
            problems=problem_batch,
        )
        request_outputs = llm.generate(prompt_texts, sampling_params, use_tqdm=True)
        for request_output, ids, record in zip(request_outputs, prompt_ids, prompt_records, strict=True):
            outputs = sorted(request_output.outputs, key=lambda output: output.index)
            for completion in outputs:
                completion_ids = np.asarray(completion.token_ids, dtype=np.int32)
                if completion_ids.size == 0:
                    LOGGER.warning(
                        "Skipping empty completion for problem_id=%s sample_id=%s",
                        record["problem"]["problem_id"],
                        completion.index,
                    )
                    continue
                top_token_ids, top_logprobs, top_mask = completion_logprob_arrays(
                    completion_logprobs=completion.logprobs,
                    token_count=int(completion_ids.shape[0]),
                    k=config.logprobs_k,
                )
                builder.add_completion(
                    split=record["split"],
                    problem=record["problem"],
                    prompt_text=record["prompt_text"],
                    prompt_ids=ids,
                    sample_id=int(completion.index),
                    completion_text=completion.text,
                    completion_token_ids=completion_ids,
                    top_token_ids=top_token_ids,
                    top_logprobs=top_logprobs,
                    top_mask=top_mask,
                    finish_reason=completion.finish_reason,
                    stop_reason=completion.stop_reason,
                    generation_config=generation_config,
                )


def main() -> None:
    configure_logging()
    config = GenerateTraceConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_id = config.tokenizer_id or config.model_id
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True),
    )
    verify_confidence_token(tokenizer)

    sft_split, eval_split = prepare_gsm8k_problem_splits(
        seed=config.seed,
        sft_problem_count=config.sft_problems,
        eval_problem_count=config.eval_problems,
    )
    split_items: list[tuple[str, list[dict[str, Any]]]] = [("sft", [dict(row) for row in sft_split])]
    if not config.skip_eval_generation:
        split_items.append(("eval", [dict(row) for row in eval_split]))

    llm = LLM(
        model=config.model_id,
        tokenizer=tokenizer_id,
        trust_remote_code=True,
        dtype=config.dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    )
    sampling_params = make_sampling_params(config)
    builder = TraceShardBuilder(output_dir=config.output_dir, shard_size=config.shard_size)

    for split, problems in split_items:
        LOGGER.info("Generating %s traces for %s problems", split, len(problems))
        generate_split(
            llm=llm,
            tokenizer=tokenizer,
            config=config,
            builder=builder,
            split=split,
            problems=problems,
            sampling_params=sampling_params,
        )

    builder.flush()
    write_manifest(
        config.output_dir,
        TraceManifest(
            model_id=config.model_id,
            tokenizer_id=tokenizer_id,
            dataset="openai/gsm8k",
            dataset_config="main",
            seed=config.seed,
            sft_problem_count=config.sft_problems,
            eval_problem_count=0 if config.skip_eval_generation else config.eval_problems,
            num_generations=config.num_generations,
            max_tokens=config.max_tokens,
            logprobs_k=config.logprobs_k,
            forbidden_token_id=CONFIDENCE_TOKEN_ID,
            shards=builder.shards,
        ),
    )


if __name__ == "__main__":
    main()
