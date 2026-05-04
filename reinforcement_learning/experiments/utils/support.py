import contextlib
import io
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from rl_trainer import RLTrainerConfig
from rl_trainer.generation import VLLMRolloutEngine
from rl_trainer.types import CompletionRecord, RewardBatch, StepMetrics
from tasks.sudoku import build_sudoku_prompt, generate_puzzle
from tasks.sudoku.parsing import parse_solution_grid
from tasks.sudoku.types import SudokuPuzzle
from tasks.sudoku.validation import exact_match

VLLMPrompt = list[dict[str, object]] | list[dict[str, str | list[dict[str, str]]]]


def load_model_and_tokenizer(
    *,
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    full_finetuning: bool,
    quiet: bool = False,
) -> tuple[Any, Any]:
    from unsloth import FastVisionModel

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                fast_inference=fast_inference,
                full_finetuning=full_finetuning,
            )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=fast_inference,
            full_finetuning=full_finetuning,
        )

    if full_finetuning:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return model, tokenizer


def build_vllm_engine(
    *,
    model_name_or_path: str,
    tokenizer: Any,
    config: RLTrainerConfig,
    sync_steps: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    enforce_eager: bool,
    sync_chunk_bytes: int,
    sync_backend: str,
) -> VLLMRolloutEngine:
    return VLLMRolloutEngine(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        config=config,
        device=torch.device("cuda"),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        sync_steps=sync_steps,
        sync_chunk_bytes=sync_chunk_bytes,
        sync_backend=sync_backend,
    )


def build_hey_prompt(prompt_text: str) -> list[dict[str, object]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]


class HeyDataset:
    def __init__(self, *, size: int, prompt_text: str) -> None:
        self.size = size
        self.prompt_text = prompt_text

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _index: int) -> dict[str, object]:
        return {"prompt": build_hey_prompt(self.prompt_text)}


async def short_completion_reward(batch: RewardBatch) -> list[float | None]:
    return [-sum(mask) for mask in batch.completion_mask]


def select_smoke_puzzle(
    *,
    rollout_engine: VLLMRolloutEngine,
    rng: random.Random,
    puzzle_difficulty: float,
    candidate_completions: int,
    min_accepted_solves: int,
    max_accepted_solves: int,
    max_puzzle_candidates: int,
) -> SudokuPuzzle:
    closest_puzzle: SudokuPuzzle | None = None
    closest_distance = candidate_completions + 1
    closest_solves = 0

    for candidate_index in range(1, max_puzzle_candidates + 1):
        puzzle = generate_puzzle(puzzle_difficulty, rng)
        solve_count = evaluate_puzzle(
            rollout_engine=rollout_engine,
            puzzle=puzzle,
            completion_count=candidate_completions,
        )
        print(f"candidate={candidate_index} exact_solves={solve_count}/{candidate_completions}")
        if min_accepted_solves <= solve_count <= max_accepted_solves:
            print("Selected accepted candidate.")
            return puzzle

        distance = abs(solve_count - (candidate_completions // 2))
        if closest_puzzle is None or distance < closest_distance:
            closest_puzzle = puzzle
            closest_distance = distance
            closest_solves = solve_count

    if closest_puzzle is None:
        raise RuntimeError("No Sudoku puzzle candidates were generated.")

    print(
        "No candidate landed in acceptance band; "
        f"using closest observed solve count {closest_solves}/{candidate_completions}."
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


def load_sudoku_validation_puzzles(path: Path) -> list[SudokuPuzzle]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        SudokuPuzzle(
            size=int(puzzle["size"]),
            box_rows=int(puzzle["box_rows"]),
            box_cols=int(puzzle["box_cols"]),
            puzzle=[[int(cell) for cell in row] for row in puzzle["puzzle"]],
            solution=[[int(cell) for cell in row] for row in puzzle["solution"]],
            difficulty=float(puzzle["difficulty"]),
            blanks=int(puzzle["blanks"]),
        )
        for puzzle in raw["puzzles"]
    ]


def evaluate_puzzle_set(
    *,
    rollout_engine: VLLMRolloutEngine,
    puzzles: list[SudokuPuzzle],
    completions_per_puzzle: int,
) -> float:
    prompts: list[VLLMPrompt] = []
    for puzzle in puzzles:
        prompt: list[dict[str, object]] = []
        for message in build_sudoku_prompt(puzzle):
            prompt.append({"role": message["role"], "content": message["content"]})
        prompts.append(prompt)
    completions = rollout_engine.generate_completions(prompts, count=completions_per_puzzle)
    solves = 0
    total = 0
    completion_index = 0
    for puzzle in puzzles:
        for _ in range(completions_per_puzzle):
            completion = completions[completion_index]
            completion_index += 1
            total += 1
            parsed = parse_solution_grid(completion, puzzle.size)
            if exact_match(parsed, puzzle.solution, puzzle.size):
                solves += 1
    return solves / max(1, total)


class SudokuEvalCallback:
    def __init__(
        self,
        *,
        rollout_engine: VLLMRolloutEngine,
        puzzle: SudokuPuzzle,
        periodic_eval_steps: int,
        periodic_eval_completions: int,
    ) -> None:
        self.rollout_engine = rollout_engine
        self.puzzle = puzzle
        self.periodic_eval_steps = periodic_eval_steps
        self.periodic_eval_completions = periodic_eval_completions

    def on_step_end(self, metrics: StepMetrics) -> None:
        if metrics.step % self.periodic_eval_steps != 0:
            return
        solves = evaluate_puzzle(
            rollout_engine=self.rollout_engine,
            puzzle=self.puzzle,
            completion_count=self.periodic_eval_completions,
        )
        print(f"eval_step={metrics.step} exact_solve_rate={solves}/{self.periodic_eval_completions}", flush=True)

    def on_completions(self, records: list[CompletionRecord]) -> None:
        del records


class SudokuValidationCallback:
    def __init__(
        self,
        *,
        rollout_engine: VLLMRolloutEngine,
        puzzles: list[SudokuPuzzle],
        eval_steps: int,
        completions_per_puzzle: int,
    ) -> None:
        self.rollout_engine = rollout_engine
        self.puzzles = puzzles
        self.eval_steps = eval_steps
        self.completions_per_puzzle = completions_per_puzzle

    def on_step_end(self, metrics: StepMetrics) -> StepMetrics | None:
        if metrics.step % self.eval_steps != 0:
            return None

        validation_reward = evaluate_puzzle_set(
            rollout_engine=self.rollout_engine,
            puzzles=self.puzzles,
            completions_per_puzzle=self.completions_per_puzzle,
        )
        print(f"validation_step={metrics.step} exact_solution={validation_reward:.3f}", flush=True)
        return replace(metrics, validation_reward_mean=validation_reward)

    def on_completions(self, records: list[CompletionRecord]) -> None:
        del records
