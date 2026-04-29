from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq


TRACE_DIR = Path(os.environ.get("TRACE_DIR", "../data/traces/gemma-4-E2B-it-mixed-confidence-4"))
MAX_TEXT_CHARS = 4000
CODING_TASK_TYPE = "coding_agent"
CODING_SOURCE_DATASET = "sandbox_harness"


@dataclass(frozen=True)
class ShardRows:
    meta_path: Path
    arrays_path: Path
    rows: list[dict[str, Any]]


def load_manifest(trace_dir: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8")))


def load_shard_rows(trace_dir: Path, manifest: Mapping[str, Any]) -> list[ShardRows]:
    shards: list[ShardRows] = []
    for shard in cast(Sequence[Mapping[str, str]], manifest["shards"]):
        meta_path = trace_dir / shard["meta_path"]
        arrays_path = trace_dir / shard["arrays_path"]
        table = pq.read_table(meta_path)
        shards.append(
            ShardRows(
                meta_path=meta_path,
                arrays_path=arrays_path,
                rows=[dict(row) for row in table.to_pylist()],
            )
        )
    return shards


def is_coding_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("task_type") == CODING_TASK_TYPE
        or row.get("source_dataset") == CODING_SOURCE_DATASET
        or "episode_id" in row
    )


def row_label(row: Mapping[str, Any]) -> int:
    value = row.get("verifier_label", row.get("math_verify_label", 0))
    return int(value)


def row_episode_id(row: Mapping[str, Any]) -> str:
    value = row.get("episode_id")
    if value is not None:
        return str(value)
    return f"problem-{int(row['problem_id'])}"


def validate_shard_shapes(shard: ShardRows, *, logprobs_k: int, max_sequence_length: int | None) -> list[str]:
    errors: list[str] = []
    arrays = np.load(shard.arrays_path)
    required_arrays = {
        "prompt_token_ids",
        "completion_token_ids",
        "top_logprob_token_ids",
        "top_logprobs",
        "top_logprob_mask",
    }
    missing = sorted(required_arrays - set(arrays.files))
    if missing:
        return [f"{shard.arrays_path}: missing arrays {missing}"]

    prompt_token_ids = arrays["prompt_token_ids"]
    completion_token_ids = arrays["completion_token_ids"]
    top_logprob_token_ids = arrays["top_logprob_token_ids"]
    top_logprobs = arrays["top_logprobs"]
    top_logprob_mask = arrays["top_logprob_mask"]

    if prompt_token_ids.ndim != 1:
        errors.append(f"{shard.arrays_path}: prompt_token_ids shape should be 1D, got {prompt_token_ids.shape}")
    if completion_token_ids.ndim != 1:
        errors.append(
            f"{shard.arrays_path}: completion_token_ids shape should be 1D, got {completion_token_ids.shape}"
        )
    expected_top_shape = (completion_token_ids.shape[0], logprobs_k)
    for name, array in (
        ("top_logprob_token_ids", top_logprob_token_ids),
        ("top_logprobs", top_logprobs),
        ("top_logprob_mask", top_logprob_mask),
    ):
        if array.shape != expected_top_shape:
            errors.append(f"{shard.arrays_path}: {name} shape {array.shape} != {expected_top_shape}")

    for row in shard.rows:
        prompt_start = int(row["prompt_token_start"])
        prompt_length = int(row["prompt_token_length"])
        token_start = int(row["token_start"])
        token_length = int(row["token_length"])
        total_token_length = int(row["total_token_length"])
        if prompt_start < 0 or prompt_start + prompt_length > prompt_token_ids.shape[0]:
            errors.append(f"{shard.meta_path}: row_id={row['row_id']} prompt span out of bounds")
        if token_start < 0 or token_start + token_length > completion_token_ids.shape[0]:
            errors.append(f"{shard.meta_path}: row_id={row['row_id']} completion span out of bounds")
        if total_token_length != prompt_length + token_length:
            errors.append(f"{shard.meta_path}: row_id={row['row_id']} total_token_length mismatch")
        if max_sequence_length is not None and total_token_length > max_sequence_length:
            errors.append(f"{shard.meta_path}: row_id={row['row_id']} exceeds max_sequence_length")
    return errors


def truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    hidden = len(text) - MAX_TEXT_CHARS
    return f"{text[:MAX_TEXT_CHARS]}\n... [truncated {hidden} chars]"


def artifact_transcript(trace_dir: Path, episode_id: str) -> list[dict[str, Any]] | None:
    path = trace_dir / "artifacts" / episode_id / "transcript.json"
    if not path.exists():
        return None
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def print_message(message: Mapping[str, Any]) -> None:
    role = str(message.get("role", "unknown")).upper()
    print(f"\n[{role}]")
    content = message.get("content")
    if isinstance(content, str) and content:
        print(truncate(content))
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            function = dict(tool_call.get("function", {}))
            print(f"tool_call {tool_call.get('id')}: {function.get('name')} {function.get('arguments')}")
    if role == "TOOL":
        print(f"name={message.get('name')} tool_call_id={message.get('tool_call_id')}")


def print_episode(trace_dir: Path, rows: Sequence[Mapping[str, Any]], *, title: str) -> None:
    if not rows:
        return
    first = rows[0]
    episode_id = row_episode_id(first)
    label = row_label(first)
    status = first.get("episode_status", first.get("normalized_prediction", "unknown"))
    print("\n" + "=" * 100)
    print(f"{title}: episode_id={episode_id} label={label} status={status} turns={len(rows)}")
    print("=" * 100)

    transcript = artifact_transcript(trace_dir, episode_id)
    if transcript is not None:
        for message in transcript:
            print_message(message)
        return

    for row in sorted(rows, key=lambda item: int(item.get("turn_index", item.get("sample_id", 0)))):
        turn_index = int(row.get("turn_index", row.get("sample_id", 0)))
        print(f"\n--- turn {turn_index} prompt ---")
        print(truncate(str(row["prompt_text"])))
        print(f"\n--- turn {turn_index} completion ---")
        print(truncate(str(row["completion_text"])))


def main() -> None:
    manifest = load_manifest(TRACE_DIR)
    shards = load_shard_rows(TRACE_DIR, manifest)
    logprobs_k = int(manifest["logprobs_k"])
    max_sequence_length = manifest.get("max_sequence_length")
    if max_sequence_length is not None:
        max_sequence_length = int(max_sequence_length)

    shape_errors: list[str] = []
    all_rows: list[dict[str, Any]] = []
    for shard in shards:
        all_rows.extend(shard.rows)
        shape_errors.extend(
            validate_shard_shapes(shard, logprobs_k=logprobs_k, max_sequence_length=max_sequence_length)
        )

    coding_rows = [row for row in all_rows if is_coding_row(row)]
    rows_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in coding_rows:
        rows_by_episode[row_episode_id(row)].append(row)

    episode_labels = {
        episode_id: max(row_label(row) for row in episode_rows)
        for episode_id, episode_rows in rows_by_episode.items()
    }
    task_type_counts = Counter(str(row.get("task_type")) for row in all_rows)
    coding_row_labels = Counter(row_label(row) for row in coding_rows)
    coding_episode_labels = Counter(episode_labels.values())

    print(f"trace_dir={TRACE_DIR}")
    print(f"manifest_dataset={manifest.get('dataset')}:{manifest.get('dataset_config')}")
    print(f"shards={len(shards)} rows={len(all_rows)}")
    print(f"task_type_counts={dict(task_type_counts)}")
    print(f"shape_errors={len(shape_errors)}")
    for error in shape_errors[:20]:
        print(f"SHAPE_ERROR {error}")
    if len(shape_errors) > 20:
        print(f"... {len(shape_errors) - 20} more shape errors")

    print(f"coding_rows={len(coding_rows)} coding_episodes={len(rows_by_episode)}")
    print(f"coding_row_labels={dict(coding_row_labels)}")
    print(f"coding_episode_labels={dict(coding_episode_labels)}")
    print(f"correct_coding_rows={coding_row_labels.get(1, 0)}")
    print(f"correct_coding_episodes={coding_episode_labels.get(1, 0)}")

    correct_episode = next((episode_id for episode_id, label in episode_labels.items() if label == 1), None)
    incorrect_episode = next((episode_id for episode_id, label in episode_labels.items() if label == 0), None)
    if correct_episode is None:
        print("\nNo correct coding episode found.")
    else:
        print_episode(TRACE_DIR, rows_by_episode[correct_episode], title="CORRECT CODING TRACE")

    if incorrect_episode is None:
        print("\nNo incorrect coding episode found.")
    else:
        print_episode(TRACE_DIR, rows_by_episode[incorrect_episode], title="INCORRECT CODING TRACE")


if __name__ == "__main__":
    main()
