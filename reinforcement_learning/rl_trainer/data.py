import random
from collections.abc import Iterator
from typing import cast

from rl_trainer.types import DatasetLike, PromptBatch, TrainingExample


def build_example(raw: dict[str, object]) -> TrainingExample:
    prompt = raw.get("prompt")
    if not isinstance(prompt, list):
        raise TypeError("Each training example must contain a list-valued `prompt` field.")

    fields = {key: value for key, value in raw.items() if key != "prompt"}
    return TrainingExample(prompt=cast(list[dict[str, object]], prompt), fields=fields)


def make_prompt_batch(examples: list[TrainingExample]) -> PromptBatch:
    return PromptBatch(examples=examples, prompts=[example.prompt for example in examples])


def iter_batches(dataset: DatasetLike, batch_size: int, *, shuffle: bool, seed: int) -> Iterator[list[TrainingExample]]:
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    position = 0

    while True:
        if position == 0 and shuffle:
            rng.shuffle(indices)

        batch_indices = indices[position : position + batch_size]
        if len(batch_indices) < batch_size:
            position = 0
            continue

        position += batch_size
        if position >= len(indices):
            position = 0

        yield [build_example(dataset[index]) for index in batch_indices]
