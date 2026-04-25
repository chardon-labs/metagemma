from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from math_verify import parse, verify
from transformers import PreTrainedTokenizerBase


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/gemma-3-1b-it"
DEFAULT_OUTPUT_DIR = "outputs/gemma-3-1b-it-confidence"
DEFAULT_TRACE_DIR = "traces/gemma-3-1b-it-gsm8k-confidence"
UNUSED0_TOKEN = "<unused0>"
UNUSED0_TOKEN_ID = 6
SYSTEM_PROMPT = (
    "Solve the following math problem. Give the final answer in the format: "
    "#### <answer>"
)

ChatMessage: TypeAlias = dict[str, str]

_GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?")


@dataclass(frozen=True)
class TraceShardPaths:
    meta_path: Path
    arrays_path: Path


@dataclass(frozen=True)
class TraceManifest:
    model_id: str
    tokenizer_id: str
    dataset: str
    dataset_config: str
    seed: int
    sft_problem_count: int
    eval_problem_count: int
    num_generations: int
    max_tokens: int
    logprobs_k: int
    forbidden_token_id: int
    shards: list[dict[str, str]]


def configure_logging() -> None:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def normalize_number(value: str) -> str:
    return value.replace(",", "").strip()


def extract_gsm8k_answer(answer: str) -> str:
    match = _GSM8K_FINAL_RE.search(answer)
    if match is not None:
        return normalize_number(match.group(1))

    numbers = _NUMBER_RE.findall(answer)
    return normalize_number(numbers[-1]) if numbers else answer.strip()


def extract_prediction(text: str) -> str:
    match = _GSM8K_FINAL_RE.search(text)
    if match is not None:
        return normalize_number(match.group(1))

    parsed = parse(text)
    if parsed:
        return str(parsed[0])

    numbers = _NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else text.strip()


def math_verify_label(completion: str, gold_answer: str) -> int:
    gold = parse(f"#### {gold_answer}")
    target = parse(completion)
    return int(bool(gold and target and verify(gold, target)))


def format_prompt(question: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def apply_chat_template(tokenizer: PreTrainedTokenizerBase, messages: Sequence[Mapping[str, str]]) -> str:
    conversation = [dict(message) for message in messages]
    rendered = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    return cast(str, rendered)


def prompt_token_ids(tokenizer: PreTrainedTokenizerBase, prompt_text: str) -> np.ndarray:
    encoded = tokenizer(prompt_text, add_special_tokens=False)
    return np.asarray(encoded["input_ids"], dtype=np.int32)


def verify_unused0_token(tokenizer: PreTrainedTokenizerBase) -> None:
    token_id = tokenizer.convert_tokens_to_ids(UNUSED0_TOKEN)
    if token_id != UNUSED0_TOKEN_ID:
        raise ValueError(
            f"{UNUSED0_TOKEN} resolved to token id {token_id}, expected {UNUSED0_TOKEN_ID}."
        )


def prepare_gsm8k_problem_splits(
    *,
    seed: int,
    sft_problem_count: int,
    eval_problem_count: int,
) -> tuple[Dataset, Dataset]:
    total = sft_problem_count + eval_problem_count
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    shuffled = dataset.shuffle(seed=seed).select(range(total))

    def map_example(example: Mapping[str, Any], index: int) -> dict[str, Any]:
        return {
            "problem_id": index,
            "question": str(example["question"]),
            "gold_answer": extract_gsm8k_answer(str(example["answer"])),
        }

    mapped = shuffled.map(map_example, with_indices=True, remove_columns=shuffled.column_names)
    sft_split = mapped.select(range(sft_problem_count))
    eval_split = mapped.select(range(sft_problem_count, total))
    return cast(Dataset, sft_split), cast(Dataset, eval_split)


def fixed_top_logprobs(
    logprobs: Mapping[int, Any] | None,
    *,
    k: int,
    forbidden_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_ids = np.full((k,), -1, dtype=np.int32)
    values = np.full((k,), -np.inf, dtype=np.float32)
    mask = np.zeros((k,), dtype=np.bool_)

    if logprobs is None:
        return token_ids, values, mask

    items: list[tuple[int, float]] = []
    for token_id, token_logprob in logprobs.items():
        token_id = int(token_id)
        if token_id == forbidden_token_id:
            continue
        value = float(getattr(token_logprob, "logprob", token_logprob))
        if np.isfinite(value):
            items.append((token_id, value))

    items.sort(key=lambda item: item[1], reverse=True)
    for index, (token_id, value) in enumerate(items[:k]):
        token_ids[index] = token_id
        values[index] = value
        mask[index] = True

    return token_ids, values, mask


def write_trace_shard(
    *,
    output_dir: Path,
    shard_index: int,
    rows: Sequence[Mapping[str, Any]],
    prompt_token_arrays: Sequence[np.ndarray],
    completion_token_arrays: Sequence[np.ndarray],
    top_token_id_arrays: Sequence[np.ndarray],
    top_logprob_arrays: Sequence[np.ndarray],
    top_mask_arrays: Sequence[np.ndarray],
) -> TraceShardPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trace-{shard_index:05d}"
    meta_path = output_dir / f"{stem}.meta.parquet"
    arrays_path = output_dir / f"{stem}.arrays.npz"

    prompt_token_ids_array = (
        np.concatenate(prompt_token_arrays).astype(np.int32, copy=False)
        if prompt_token_arrays
        else np.empty((0,), dtype=np.int32)
    )
    completion_token_ids_array = (
        np.concatenate(completion_token_arrays).astype(np.int32, copy=False)
        if completion_token_arrays
        else np.empty((0,), dtype=np.int32)
    )
    if top_token_id_arrays:
        top_logprob_token_ids = np.concatenate(top_token_id_arrays, axis=0).astype(np.int32, copy=False)
        top_logprobs = np.concatenate(top_logprob_arrays, axis=0).astype(np.float32, copy=False)
        top_logprob_mask = np.concatenate(top_mask_arrays, axis=0).astype(np.bool_, copy=False)
    else:
        top_logprob_token_ids = np.empty((0, 0), dtype=np.int32)
        top_logprobs = np.empty((0, 0), dtype=np.float32)
        top_logprob_mask = np.empty((0, 0), dtype=np.bool_)

    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), meta_path)
    np.savez_compressed(
        arrays_path,
        prompt_token_ids=prompt_token_ids_array,
        completion_token_ids=completion_token_ids_array,
        top_logprob_token_ids=top_logprob_token_ids,
        top_logprobs=top_logprobs,
        top_logprob_mask=top_logprob_mask,
    )
    LOGGER.info("Wrote %s rows to %s and %s", len(rows), meta_path, arrays_path)
    return TraceShardPaths(meta_path=meta_path, arrays_path=arrays_path)


def read_trace_metadata(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return [dict(row) for row in table.to_pylist()]


def write_manifest(output_dir: Path, manifest: TraceManifest) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote manifest to %s", manifest_path)


def load_manifest(trace_dir: Path) -> TraceManifest:
    payload = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    return TraceManifest(**payload)
