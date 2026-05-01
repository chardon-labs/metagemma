from collections.abc import Sequence

from rl_trainer.types import RewardBatch, RewardFunction
from tasks.sudoku.parsing import grid_from_sequence, int_from_sequence, parse_solution_grid
from tasks.sudoku.validation import exact_match


def _field(batch: RewardBatch, name: str) -> Sequence[object]:
    return batch.extra_fields[name]


async def exact_solution(batch: RewardBatch) -> list[float | None]:
    scores: list[float | None] = []
    solution = _field(batch, "solution")
    size = _field(batch, "size")
    for index, completion in enumerate(batch.completions):
        expected_size = int_from_sequence(size, index)
        parsed = parse_solution_grid(completion[0]["content"], expected_size)
        solution_grid = grid_from_sequence(solution, index)
        scores.append(1.0 if exact_match(parsed, solution_grid, expected_size) else 0.0)
    return scores


SUDOKU_REWARD_FUNCTIONS: list[RewardFunction] = [
    exact_solution,
]
