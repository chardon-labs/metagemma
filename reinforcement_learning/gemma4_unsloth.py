from pathlib import Path
from typing import TypeAlias

import unsloth
from safetensors import safe_open
from transformers import TextStreamer
from unsloth import FastVisionModel

from rl_trainer import PrintCallback, RLTrainer, RLTrainerConfig, TrainerCallback
from rl_trainer.sudoku import (
    CurriculumCallback,
    Grid,
    SudokuCurriculum,
    SudokuDataset,
    correct_cell_fraction,
    exact_match,
    grid_from_sequence,
    has_correct_shape,
    has_numbers_in_range,
    int_from_sequence,
    is_valid_solution,
    parse_solution_grid,
    preserves_given_cells,
)
from rl_trainer.types import Completion

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 8192
LORA_RANK = 32
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
DATASET_SIZE = 1000
MAX_STEPS = 60
OUTPUT_DIR = Path("outputs")
ADAPTER_DIR = Path("gemma_4_lora")
MAX_COMPLETION_LENGTH = 2048

RewardValue: TypeAlias = int | float | str | Grid
RewardField: TypeAlias = list[RewardValue]


async def solution_parses(
    completions: list[Completion],
    size: RewardField,
    **_kwargs: RewardField,
) -> list[float | None]:
    scores: list[float | None] = []
    for index, completion in enumerate(completions):
        parsed = parse_solution_grid(completion[0]["content"], int_from_sequence(size, index))
        scores.append(1.0 if parsed is not None else 0.0)
    return scores


async def correct_shape(
    completions: list[Completion],
    size: RewardField,
    **_kwargs: RewardField,
) -> list[float | None]:
    scores: list[float | None] = []
    for index, completion in enumerate(completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        scores.append(1.0 if has_correct_shape(parsed, expected_size) else 0.0)
    return scores


async def numbers_in_range(
    completions: list[Completion],
    size: RewardField,
    **_kwargs: RewardField,
) -> list[float | None]:
    scores: list[float | None] = []
    for index, completion in enumerate(completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        scores.append(1.0 if has_numbers_in_range(parsed, expected_size) else 0.0)
    return scores


async def respects_given_cells(
    completions: list[Completion],
    puzzle: RewardField,
    size: RewardField,
    **_kwargs: RewardField,
) -> list[float | None]:
    scores: list[float | None] = []
    for index, completion in enumerate(completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        prompt_grid = grid_from_sequence(puzzle, index)
        scores.append(2.0 if preserves_given_cells(parsed, prompt_grid, expected_size) else 0.0)
    return scores


async def valid_sudoku(
    completions: list[Completion],
    size: RewardField,
    box_rows: RewardField,
    box_cols: RewardField,
    **_kwargs: RewardField,
) -> list[float | None]:
    scores: list[float | None] = []
    for index, completion in enumerate(completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        scores.append(
            4.0
            if is_valid_solution(
                parsed,
                expected_size,
                int_from_sequence(box_rows, index),
                int_from_sequence(box_cols, index),
            )
            else 0.0
        )
    return scores


async def correct_cells(
    completions: list[Completion],
    solution: RewardField,
    size: RewardField,
    **_kwargs: RewardField,
) -> list[float | None]:
    scores: list[float | None] = []
    for index, completion in enumerate(completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        solution_grid = grid_from_sequence(solution, index)
        scores.append(4.0 * correct_cell_fraction(parsed, solution_grid, expected_size))
    return scores


async def exact_solution(
    completions: list[Completion],
    solution: RewardField,
    size: RewardField,
    **_kwargs: RewardField,
) -> list[float | None]:
    scores: list[float | None] = []
    for index, completion in enumerate(completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        solution_grid = grid_from_sequence(solution, index)
        scores.append(10.0 if exact_match(parsed, solution_grid, expected_size) else 0.0)
    return scores


def load_model_and_tokenizer():
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_STATE,
    )
    return model, tokenizer


def build_training_config() -> RLTrainerConfig:
    return RLTrainerConfig(
        temperature=1.0,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        logging_steps=1,
        batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=100,
        output_dir=OUTPUT_DIR,
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="bnpo",
        mask_truncated_completions=True,
    )


def run_base_generation(model, tokenizer, dataset: SudokuDataset) -> None:
    sample = dataset[0]
    prompt = sample["prompt"]
    if not isinstance(prompt, list):
        raise TypeError("Expected prompt list.")

    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text=text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    print("=" * 50)
    print("BASE MODEL OUTPUT")
    print("=" * 50)
    model.generate(
        **inputs,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
        max_new_tokens=512,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )


def verify_saved_lora(adapter_dir: Path) -> None:
    adapter_path = adapter_dir / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt") as tensors:
        for key in tensors.keys():
            tensor = tensors.get_tensor(key)
            zero_fraction = (tensor == 0).sum().item() / tensor.numel()
            if zero_fraction == 1.0:
                raise ValueError(f"{key} is all zeros")


def run_trained_generation(model, tokenizer, dataset: SudokuDataset) -> None:
    sample = dataset[0]
    prompt = sample["prompt"]
    if not isinstance(prompt, list):
        raise TypeError("Expected prompt list.")

    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    model.generate(
        **tokenizer(images=None, text=text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=MAX_COMPLETION_LENGTH,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )


def main() -> None:
    model, tokenizer = load_model_and_tokenizer()
    curriculum = SudokuCurriculum()
    dataset = SudokuDataset(
        size=DATASET_SIZE,
        curriculum=curriculum,
        seed=RANDOM_STATE,
    )

    print("Dataset sample:")
    print(dataset[0])

    run_base_generation(model, tokenizer, dataset)
    callbacks: list[TrainerCallback] = [PrintCallback(), CurriculumCallback(curriculum)]

    trainer = RLTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_functions=[
            solution_parses,
            correct_shape,
            numbers_in_range,
            respects_given_cells,
            valid_sudoku,
            correct_cells,
            exact_solution,
        ],
        config=build_training_config(),
        callbacks=callbacks,
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    verify_saved_lora(ADAPTER_DIR)
    run_trained_generation(model, tokenizer, dataset)


if __name__ == "__main__":
    main()
