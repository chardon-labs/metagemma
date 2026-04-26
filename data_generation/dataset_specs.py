from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypedDict, cast

from datasets import Dataset, load_dataset

from confidence_trace import extract_gsm8k_answer


LETTERS = tuple("ABCDEFGHIJKLMNOP")
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

MATH_SYSTEM_PROMPT = "Solve the following math problem. Give the final answer in the format: #### <answer>"
MCQ_SYSTEM_PROMPT = (
    "Answer the multiple-choice question. Think through the problem, then give the final answer "
    "in the format: #### <letter>"
)
FACT_CHECK_SYSTEM_PROMPT = (
    "Classify the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO. Think through the claim, then "
    "give the final answer in the format: #### <label>"
)


class Problem(TypedDict):
    problem_id: int
    source_dataset: str
    source_config: str | None
    source_split: str
    source_id: str
    task_type: str
    scorer: str
    system_prompt: str
    user_prompt: str
    question: str
    gold_answer: str
    choices: list[str]
    choice_labels: list[str]


Loader = Callable[["DatasetSpec", int], tuple[list[Problem], list[Problem]]]
TaskType = Literal["math", "multiple_choice", "fact_check"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    dataset_config: str | None
    task_type: TaskType
    scorer: str
    sft_count: int
    eval_count: int
    loader: Loader
    sft_split: str
    eval_split: str

    def manifest_entry(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("loader")
        return payload


@dataclass(frozen=True)
class DatasetSampleCounts:
    sft_count: int
    eval_count: int


DEFAULT_DATASET_SAMPLE_COUNTS = DatasetSampleCounts(sft_count=200, eval_count=30)
DATASET_SAMPLE_COUNTS = {
    "gsm8k": DEFAULT_DATASET_SAMPLE_COUNTS,
    "math500": DEFAULT_DATASET_SAMPLE_COUNTS,
    "mmlu_pro": DEFAULT_DATASET_SAMPLE_COUNTS,
    "arc_challenge": DEFAULT_DATASET_SAMPLE_COUNTS,
    "truthfulqa_mc1": DEFAULT_DATASET_SAMPLE_COUNTS,
    "fever": DEFAULT_DATASET_SAMPLE_COUNTS,
}


def _sample_counts(name: str) -> DatasetSampleCounts:
    return DATASET_SAMPLE_COUNTS.get(name, DEFAULT_DATASET_SAMPLE_COUNTS)


def _select_shuffled(dataset: Dataset, *, seed: int, count: int) -> Dataset:
    limit = min(count, len(dataset))
    return cast(Dataset, dataset.shuffle(seed=seed).select(range(limit)))


def _split_single_dataset(dataset: Dataset, *, seed: int, sft_count: int, eval_count: int) -> tuple[Dataset, Dataset]:
    total = min(sft_count + eval_count, len(dataset))
    shuffled = cast(Dataset, dataset.shuffle(seed=seed).select(range(total)))
    sft_end = min(sft_count, len(shuffled))
    sft_split = shuffled.select(range(sft_end))
    eval_split = shuffled.select(range(sft_end, len(shuffled)))
    return cast(Dataset, sft_split), cast(Dataset, eval_split)


def _choice_prompt(question: str, labels: Sequence[str], choices: Sequence[str]) -> str:
    rendered_choices = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True))
    return f"Question:\n{question}\n\nOptions:\n{rendered_choices}"


def _problem(
    *,
    source_dataset: str,
    source_config: str | None,
    source_split: str,
    source_id: str,
    task_type: str,
    scorer: str,
    system_prompt: str,
    user_prompt: str,
    question: str,
    gold_answer: str,
    choices: Sequence[str] = (),
    choice_labels: Sequence[str] = (),
) -> Problem:
    return {
        "problem_id": -1,
        "source_dataset": source_dataset,
        "source_config": source_config,
        "source_split": source_split,
        "source_id": source_id,
        "task_type": task_type,
        "scorer": scorer,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "question": question,
        "gold_answer": gold_answer,
        "choices": list(choices),
        "choice_labels": list(choice_labels),
    }


def _load_split(dataset_id: str, dataset_config: str | None, split: str) -> Dataset:
    if dataset_config is None:
        return load_dataset(dataset_id, split=split)
    return load_dataset(dataset_id, dataset_config, split=split)


def load_gsm8k(spec: DatasetSpec, seed: int) -> tuple[list[Problem], list[Problem]]:
    dataset = _load_split(spec.dataset_id, spec.dataset_config, spec.sft_split)
    sft_split, eval_split = _split_single_dataset(
        dataset,
        seed=seed,
        sft_count=spec.sft_count,
        eval_count=spec.eval_count,
    )

    def convert(row: Mapping[str, Any], split: str, index: int) -> Problem:
        question = str(row["question"])
        return _problem(
            source_dataset=spec.dataset_id,
            source_config=spec.dataset_config,
            source_split=split,
            source_id=f"{split}:{index}",
            task_type=spec.task_type,
            scorer=spec.scorer,
            system_prompt=MATH_SYSTEM_PROMPT,
            user_prompt=question,
            question=question,
            gold_answer=extract_gsm8k_answer(str(row["answer"])),
        )

    return (
        [convert(cast(Mapping[str, Any], row), spec.sft_split, index) for index, row in enumerate(sft_split)],
        [
            convert(cast(Mapping[str, Any], row), f"{spec.sft_split}:eval", index)
            for index, row in enumerate(eval_split)
        ],
    )


def load_math500(spec: DatasetSpec, seed: int) -> tuple[list[Problem], list[Problem]]:
    dataset = _load_split(spec.dataset_id, spec.dataset_config, spec.sft_split)
    sft_split, eval_split = _split_single_dataset(
        dataset,
        seed=seed,
        sft_count=spec.sft_count,
        eval_count=spec.eval_count,
    )

    def convert(row: Mapping[str, Any], split: str) -> Problem:
        question = str(row["problem"])
        return _problem(
            source_dataset=spec.dataset_id,
            source_config=spec.dataset_config,
            source_split=split,
            source_id=str(row["unique_id"]),
            task_type=spec.task_type,
            scorer=spec.scorer,
            system_prompt=MATH_SYSTEM_PROMPT,
            user_prompt=question,
            question=question,
            gold_answer=str(row["answer"]),
        )

    return (
        [convert(cast(Mapping[str, Any], row), spec.sft_split) for row in sft_split],
        [convert(cast(Mapping[str, Any], row), spec.sft_split) for row in eval_split],
    )


def load_mmlu_pro(spec: DatasetSpec, seed: int) -> tuple[list[Problem], list[Problem]]:
    dataset = _load_split(spec.dataset_id, spec.dataset_config, spec.sft_split)
    sft_split, eval_split = _split_single_dataset(
        dataset,
        seed=seed,
        sft_count=spec.sft_count,
        eval_count=spec.eval_count,
    )

    def convert(row: Mapping[str, Any], split: str) -> Problem:
        choices = [str(choice).strip() for choice in row["options"] if str(choice).strip() != "N/A"]
        labels = list(LETTERS[: len(choices)])
        question = str(row["question"])
        return _problem(
            source_dataset=spec.dataset_id,
            source_config=spec.dataset_config,
            source_split=split,
            source_id=str(row["question_id"]),
            task_type=spec.task_type,
            scorer=spec.scorer,
            system_prompt=MCQ_SYSTEM_PROMPT,
            user_prompt=_choice_prompt(question, labels, choices),
            question=question,
            gold_answer=str(row["answer"]).strip(),
            choices=choices,
            choice_labels=labels,
        )

    return (
        [convert(cast(Mapping[str, Any], row), spec.sft_split) for row in sft_split],
        [convert(cast(Mapping[str, Any], row), spec.sft_split) for row in eval_split],
    )


def load_arc_challenge(spec: DatasetSpec, seed: int) -> tuple[list[Problem], list[Problem]]:
    sft_dataset = _select_shuffled(
        _load_split(spec.dataset_id, spec.dataset_config, spec.sft_split),
        seed=seed,
        count=spec.sft_count,
    )
    eval_dataset = _select_shuffled(
        _load_split(spec.dataset_id, spec.dataset_config, spec.eval_split),
        seed=seed,
        count=spec.eval_count,
    )

    def convert(row: Mapping[str, Any], split: str) -> Problem:
        choice_payload = cast(Mapping[str, Sequence[str]], row["choices"])
        labels = [str(label) for label in choice_payload["label"]]
        choices = [str(choice) for choice in choice_payload["text"]]
        question = str(row["question"])
        return _problem(
            source_dataset=spec.dataset_id,
            source_config=spec.dataset_config,
            source_split=split,
            source_id=str(row["id"]),
            task_type=spec.task_type,
            scorer=spec.scorer,
            system_prompt=MCQ_SYSTEM_PROMPT,
            user_prompt=_choice_prompt(question, labels, choices),
            question=question,
            gold_answer=str(row["answerKey"]).strip(),
            choices=choices,
            choice_labels=labels,
        )

    return (
        [convert(cast(Mapping[str, Any], row), spec.sft_split) for row in sft_dataset],
        [convert(cast(Mapping[str, Any], row), spec.eval_split) for row in eval_dataset],
    )


def load_truthfulqa_mc1(spec: DatasetSpec, seed: int) -> tuple[list[Problem], list[Problem]]:
    dataset = _load_split(spec.dataset_id, spec.dataset_config, spec.sft_split)
    sft_split, eval_split = _split_single_dataset(
        dataset,
        seed=seed,
        sft_count=spec.sft_count,
        eval_count=spec.eval_count,
    )

    def convert(row: Mapping[str, Any], split: str, index: int) -> Problem:
        targets = cast(Mapping[str, Sequence[Any]], row["mc1_targets"])
        unshuffled_choices = [str(choice) for choice in targets["choices"]]
        unshuffled_target_labels = [int(label) for label in targets["labels"]]
        permutation = list(range(len(unshuffled_choices)))
        random.Random(f"{seed}:{split}:{index}").shuffle(permutation)
        choices = [unshuffled_choices[choice_index] for choice_index in permutation]
        target_labels = [unshuffled_target_labels[choice_index] for choice_index in permutation]
        labels = list(LETTERS[: len(choices)])
        correct_index = target_labels.index(1)
        question = str(row["question"])
        return _problem(
            source_dataset=spec.dataset_id,
            source_config=spec.dataset_config,
            source_split=split,
            source_id=f"{split}:{index}",
            task_type=spec.task_type,
            scorer=spec.scorer,
            system_prompt=MCQ_SYSTEM_PROMPT,
            user_prompt=_choice_prompt(question, labels, choices),
            question=question,
            gold_answer=labels[correct_index],
            choices=choices,
            choice_labels=labels,
        )

    return (
        [convert(cast(Mapping[str, Any], row), spec.sft_split, index) for index, row in enumerate(sft_split)],
        [
            convert(cast(Mapping[str, Any], row), f"{spec.sft_split}:eval", index)
            for index, row in enumerate(eval_split)
        ],
    )


def load_fever(spec: DatasetSpec, seed: int) -> tuple[list[Problem], list[Problem]]:
    sft_dataset = _select_shuffled(
        _load_split(spec.dataset_id, spec.dataset_config, spec.sft_split),
        seed=seed,
        count=spec.sft_count,
    )
    eval_dataset = _select_shuffled(
        _load_split(spec.dataset_id, spec.dataset_config, spec.eval_split),
        seed=seed,
        count=spec.eval_count,
    )

    def convert(row: Mapping[str, Any], split: str) -> Problem:
        claim = str(row["claim"])
        return _problem(
            source_dataset=spec.dataset_id,
            source_config=spec.dataset_config,
            source_split=split,
            source_id=str(row["id"]),
            task_type=spec.task_type,
            scorer=spec.scorer,
            system_prompt=FACT_CHECK_SYSTEM_PROMPT,
            user_prompt=f"Claim:\n{claim}",
            question=claim,
            gold_answer=str(row["label"]),
            choices=FEVER_LABELS,
            choice_labels=FEVER_LABELS,
        )

    return (
        [convert(cast(Mapping[str, Any], row), spec.sft_split) for row in sft_dataset],
        [convert(cast(Mapping[str, Any], row), spec.eval_split) for row in eval_dataset],
    )


DATASET_SPECS = [
    DatasetSpec(
        name="gsm8k",
        dataset_id="openai/gsm8k",
        dataset_config="main",
        task_type="math",
        scorer="math_verify",
        sft_count=_sample_counts("gsm8k").sft_count,
        eval_count=_sample_counts("gsm8k").eval_count,
        loader=load_gsm8k,
        sft_split="train",
        eval_split="train",
    ),
    DatasetSpec(
        name="math500",
        dataset_id="HuggingFaceH4/MATH-500",
        dataset_config=None,
        task_type="math",
        scorer="math_verify",
        sft_count=_sample_counts("math500").sft_count,
        eval_count=_sample_counts("math500").eval_count,
        loader=load_math500,
        sft_split="test",
        eval_split="test",
    ),
    DatasetSpec(
        name="mmlu_pro",
        dataset_id="TIGER-Lab/MMLU-Pro",
        dataset_config=None,
        task_type="multiple_choice",
        scorer="multiple_choice_exact",
        sft_count=_sample_counts("mmlu_pro").sft_count,
        eval_count=_sample_counts("mmlu_pro").eval_count,
        loader=load_mmlu_pro,
        sft_split="test",
        eval_split="test",
    ),
    DatasetSpec(
        name="arc_challenge",
        dataset_id="allenai/ai2_arc",
        dataset_config="ARC-Challenge",
        task_type="multiple_choice",
        scorer="multiple_choice_exact",
        sft_count=_sample_counts("arc_challenge").sft_count,
        eval_count=_sample_counts("arc_challenge").eval_count,
        loader=load_arc_challenge,
        sft_split="train",
        eval_split="validation",
    ),
    DatasetSpec(
        name="truthfulqa_mc1",
        dataset_id="truthfulqa/truthful_qa",
        dataset_config="multiple_choice",
        task_type="multiple_choice",
        scorer="multiple_choice_exact",
        sft_count=_sample_counts("truthfulqa_mc1").sft_count,
        eval_count=_sample_counts("truthfulqa_mc1").eval_count,
        loader=load_truthfulqa_mc1,
        sft_split="validation",
        eval_split="validation",
    ),
    DatasetSpec(
        name="fever",
        dataset_id="maxzoech/fever",
        dataset_config=None,
        task_type="fact_check",
        scorer="label_exact",
        sft_count=_sample_counts("fever").sft_count,
        eval_count=_sample_counts("fever").eval_count,
        loader=load_fever,
        sft_split="train",
        eval_split="test",
    ),
]


def prepare_problem_splits(seed: int) -> tuple[list[Problem], list[Problem]]:
    sft_problems: list[Problem] = []
    eval_problems: list[Problem] = []
    for index, spec in enumerate(DATASET_SPECS):
        spec_sft, spec_eval = spec.loader(spec, seed + index)
        sft_problems.extend(spec_sft)
        eval_problems.extend(spec_eval)

    for problem_id, problem in enumerate([*sft_problems, *eval_problems]):
        problem["problem_id"] = problem_id

    return sft_problems, eval_problems


def dataset_manifest_entries() -> list[dict[str, Any]]:
    return [spec.manifest_entry() for spec in DATASET_SPECS]
