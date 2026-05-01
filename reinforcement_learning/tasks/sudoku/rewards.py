from rl_trainer.types import Completion, RewardFunction
from tasks.sudoku.parsing import grid_from_sequence, int_from_sequence, parse_solution_grid
from tasks.sudoku.types import RewardField
from tasks.sudoku.validation import (
    correct_cell_fraction,
    exact_match,
    has_correct_shape,
    has_numbers_in_range,
    is_valid_solution,
    preserves_given_cells,
)


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


SUDOKU_REWARD_FUNCTIONS: list[RewardFunction] = [
    solution_parses,
    correct_shape,
    numbers_in_range,
    respects_given_cells,
    valid_sudoku,
    correct_cells,
    exact_solution,
]
