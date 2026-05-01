from collections.abc import Sequence

from rl_trainer.types import RewardBatch, RewardFunction
from tasks.sudoku.parsing import grid_from_sequence, int_from_sequence, parse_solution_grid
from tasks.sudoku.validation import (
    correct_cell_fraction,
    exact_match,
    has_correct_shape,
    has_numbers_in_range,
    is_valid_solution,
    preserves_given_cells,
)


def _field(batch: RewardBatch, name: str) -> Sequence[object]:
    return batch.extra_fields[name]


async def solution_parses(batch: RewardBatch) -> list[float | None]:
    scores: list[float | None] = []
    size = _field(batch, "size")
    for index, completion in enumerate(batch.completions):
        parsed = parse_solution_grid(completion[0]["content"], int_from_sequence(size, index))
        scores.append(1.0 if parsed is not None else 0.0)
    return scores


async def correct_shape(batch: RewardBatch) -> list[float | None]:
    scores: list[float | None] = []
    size = _field(batch, "size")
    for index, completion in enumerate(batch.completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        scores.append(1.0 if has_correct_shape(parsed, expected_size) else 0.0)
    return scores


async def numbers_in_range(batch: RewardBatch) -> list[float | None]:
    scores: list[float | None] = []
    size = _field(batch, "size")
    for index, completion in enumerate(batch.completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        scores.append(1.0 if has_numbers_in_range(parsed, expected_size) else 0.0)
    return scores


async def respects_given_cells(batch: RewardBatch) -> list[float | None]:
    scores: list[float | None] = []
    puzzle = _field(batch, "puzzle")
    size = _field(batch, "size")
    for index, completion in enumerate(batch.completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        prompt_grid = grid_from_sequence(puzzle, index)
        scores.append(2.0 if preserves_given_cells(parsed, prompt_grid, expected_size) else 0.0)
    return scores


async def valid_sudoku(batch: RewardBatch) -> list[float | None]:
    scores: list[float | None] = []
    size = _field(batch, "size")
    box_rows = _field(batch, "box_rows")
    box_cols = _field(batch, "box_cols")
    for index, completion in enumerate(batch.completions):
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


async def correct_cells(batch: RewardBatch) -> list[float | None]:
    scores: list[float | None] = []
    solution = _field(batch, "solution")
    size = _field(batch, "size")
    for index, completion in enumerate(batch.completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        solution_grid = grid_from_sequence(solution, index)
        scores.append(4.0 * correct_cell_fraction(parsed, solution_grid, expected_size))
    return scores


async def exact_solution(batch: RewardBatch) -> list[float | None]:
    scores: list[float | None] = []
    solution = _field(batch, "solution")
    size = _field(batch, "size")
    for index, completion in enumerate(batch.completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        solution_grid = grid_from_sequence(solution, index)
        scores.append(10.0 if exact_match(parsed, solution_grid, expected_size) else 0.0)
    return scores


SUDOKU_REWARD_FUNCTIONS: list[RewardFunction] = [
    solution_parses,
    correct_shape,
    numbers_in_range,
    respects_given_cells,
    valid_sudoku,
    correct_cells,
    exact_solution,
]
