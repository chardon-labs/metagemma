import random
from pathlib import Path
from typing import Any

from unsloth import FastVisionModel
import torch

from rl_trainer import PrintCallback, RLTrainer, RLTrainerConfig
from rl_trainer.generation import VLLMRolloutEngine
from rl_trainer.types import CompletionRecord, StepMetrics
from tasks.sudoku import SUDOKU_REWARD_FUNCTIONS, SinglePuzzleDataset, build_sudoku_prompt, generate_puzzle
from tasks.sudoku.parsing import parse_solution_grid
from tasks.sudoku.types import SudokuPuzzle
from tasks.sudoku.validation import exact_match

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 8192
MAX_COMPLETION_LENGTH = 2048
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
FULL_FINETUNING = True

PUZZLE_DIFFICULTY = 0.35
CANDIDATE_COMPLETIONS = 8
MIN_ACCEPTED_SOLVES = 2
MAX_ACCEPTED_SOLVES = 6
MAX_PUZZLE_CANDIDATES = 32
EVAL_COMPLETIONS = 64
PERIODIC_EVAL_STEPS = 10
PERIODIC_EVAL_COMPLETIONS = 128

DATASET_SIZE = 1000
MAX_STEPS = 300
OUTPUT_DIR = Path("outputs/sudoku_single_puzzle_smoke")
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"

VLLM_GPU_MEMORY_UTILIZATION = 0.20
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_ENFORCE_EAGER = True
VLLM_SYNC_STEPS = 1
VLLM_SYNC_BACKEND = "inprocess"
VLLM_SYNC_CHUNK_BYTES = 8 * 1024 * 1024 * 1024


def load_model_and_tokenizer() -> tuple[Any, Any]:
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
        full_finetuning=FULL_FINETUNING,
    )
    if FULL_FINETUNING:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return model, tokenizer


def build_training_config() -> RLTrainerConfig:
    return RLTrainerConfig(
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.0,
        warmup_ratio=0.03,
        logging_steps=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=0,
        output_dir=OUTPUT_DIR,
        mask_truncated_completions=False,
        max_grad_norm=1.0,
        seed=RANDOM_STATE,
        shuffle=True,
        optimizer="adamw",
        empty_cache_steps=1,
    )


def build_vllm_engine(
    model_name_or_path: str,
    tokenizer: Any,
    config: RLTrainerConfig,
    *,
    sync_steps: int = 0,
) -> VLLMRolloutEngine:
    return VLLMRolloutEngine(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        config=config,
        device=torch.device("cuda"),
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        enforce_eager=VLLM_ENFORCE_EAGER,
        sync_steps=sync_steps,
        sync_chunk_bytes=VLLM_SYNC_CHUNK_BYTES,
        sync_backend=VLLM_SYNC_BACKEND,
    )


def select_smoke_puzzle(
    *,
    rollout_engine: VLLMRolloutEngine,
    rng: random.Random,
) -> SudokuPuzzle:
    closest_puzzle: SudokuPuzzle | None = None
    closest_distance = CANDIDATE_COMPLETIONS + 1
    closest_solves = 0

    for candidate_index in range(1, MAX_PUZZLE_CANDIDATES + 1):
        puzzle = generate_puzzle(PUZZLE_DIFFICULTY, rng)
        solve_count = evaluate_puzzle(
            rollout_engine=rollout_engine,
            puzzle=puzzle,
            completion_count=CANDIDATE_COMPLETIONS,
        )
        print(f"candidate={candidate_index} exact_solves={solve_count}/{CANDIDATE_COMPLETIONS}")
        if MIN_ACCEPTED_SOLVES <= solve_count <= MAX_ACCEPTED_SOLVES:
            print("Selected accepted candidate.")
            return puzzle

        distance = abs(solve_count - (CANDIDATE_COMPLETIONS // 2))
        if closest_puzzle is None or distance < closest_distance:
            closest_puzzle = puzzle
            closest_distance = distance
            closest_solves = solve_count

    if closest_puzzle is None:
        raise RuntimeError("No Sudoku puzzle candidates were generated.")

    print(
        "No candidate landed in acceptance band; "
        f"using closest observed solve count {closest_solves}/{CANDIDATE_COMPLETIONS}."
    )
    return closest_puzzle


def evaluate_puzzle(
    *,
    rollout_engine: VLLMRolloutEngine,
    puzzle: SudokuPuzzle,
    completion_count: int,
) -> int:
    completions = rollout_engine.generate_completions([build_sudoku_prompt(puzzle)], count=completion_count)
    return exact_solve_count(completions, puzzle)


def print_training_config(config: RLTrainerConfig) -> None:
    print(
        "smoke_training_config "
        f"generations={config.num_generations} lr={config.learning_rate:.2e} "
        f"weight_decay={config.weight_decay:.3g} temperature={config.temperature:.2f} "
        f"max_completion={config.max_completion_length} "
        f"mask_truncated={config.mask_truncated_completions} "
        f"vllm_sync_steps={VLLM_SYNC_STEPS}",
        flush=True,
    )


def exact_solve_count(completions: list[str], puzzle: SudokuPuzzle) -> int:
    solves = 0
    for completion in completions:
        parsed = parse_solution_grid(completion, puzzle.size)
        if exact_match(parsed, puzzle.solution, puzzle.size):
            solves += 1
    return solves


def print_puzzle(puzzle: SudokuPuzzle) -> None:
    print("Selected puzzle:")
    for row in puzzle.puzzle:
        print(" ".join(str(cell) for cell in row))
    print("Solution:")
    for row in puzzle.solution:
        print(" ".join(str(cell) for cell in row))


class SudokuEvalCallback:
    def __init__(self, *, rollout_engine: VLLMRolloutEngine, puzzle: SudokuPuzzle) -> None:
        self.rollout_engine = rollout_engine
        self.puzzle = puzzle

    def on_step_end(self, metrics: StepMetrics) -> None:
        if metrics.step % PERIODIC_EVAL_STEPS != 0:
            return
        solves = evaluate_puzzle(
            rollout_engine=self.rollout_engine,
            puzzle=self.puzzle,
            completion_count=PERIODIC_EVAL_COMPLETIONS,
        )
        print(f"eval_step={metrics.step} exact_solve_rate={solves}/{PERIODIC_EVAL_COMPLETIONS}", flush=True)

    def on_completions(self, records: list[CompletionRecord]) -> None:
        del records


def main() -> None:
    config = build_training_config()
    print_training_config(config)
    rng = random.Random(RANDOM_STATE)
    model, tokenizer = load_model_and_tokenizer()
    rollout_engine = build_vllm_engine(MODEL_NAME, tokenizer, config, sync_steps=VLLM_SYNC_STEPS)

    puzzle = select_smoke_puzzle(rollout_engine=rollout_engine, rng=rng)
    print_puzzle(puzzle)

    base_solves = evaluate_puzzle(
        rollout_engine=rollout_engine,
        puzzle=puzzle,
        completion_count=EVAL_COMPLETIONS,
    )
    print(f"base_exact_solve_rate={base_solves}/{EVAL_COMPLETIONS}")

    dataset = SinglePuzzleDataset(puzzle=puzzle, size=DATASET_SIZE)
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_functions=SUDOKU_REWARD_FUNCTIONS,
        config=config,
        rollout_engine=rollout_engine,
        callbacks=[PrintCallback(), SudokuEvalCallback(rollout_engine=rollout_engine, puzzle=puzzle)],
    )
    trainer.train()

    trained_solves = evaluate_puzzle(
        rollout_engine=rollout_engine,
        puzzle=puzzle,
        completion_count=EVAL_COMPLETIONS,
    )
    print(f"trained_exact_solve_rate={trained_solves}/{EVAL_COMPLETIONS}")

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    del trainer
    del rollout_engine
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
