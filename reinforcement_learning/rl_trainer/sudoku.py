import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from rl_trainer.callbacks import TrainerCallback
from rl_trainer.types import CompletionRecord, StepMetrics

Grid: TypeAlias = list[list[int]]
PromptMessage: TypeAlias = dict[str, str | list[dict[str, str]]]
DatasetValue: TypeAlias = int | float | str | Grid | list[PromptMessage]

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class SudokuPuzzle:
    size: int
    box_rows: int
    box_cols: int
    puzzle: Grid
    solution: Grid
    difficulty: float
    blanks: int


@dataclass(frozen=True)
class SudokuSpec:
    size: int
    box_rows: int
    box_cols: int
    min_blanks: int
    max_blanks: int


@dataclass
class SudokuCurriculum:
    difficulty: float = 0.0
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    window: int = 20
    history: list[dict[str, float]] = field(default_factory=list)

    def update(self, metrics: dict[str, float]) -> None:
        self.history.append(metrics)
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

        averaged = self._averaged_metrics()
        parse_ready = (
            averaged.get("solution_parses", 0.0) >= 0.95
            and averaged.get("correct_shape", 0.0) >= 0.95
            and averaged.get("numbers_in_range", 0.0) >= 0.95
        )
        if not parse_ready:
            self.difficulty = max(self.min_difficulty, self.difficulty - 0.05)
            return

        exact = averaged.get("exact_solution", 0.0)
        valid = averaged.get("valid_sudoku", 0.0)
        total = averaged.get("reward_mean", 0.0)

        if exact >= 9.0 and valid >= 3.5:
            self.difficulty = min(self.max_difficulty, self.difficulty + 0.03)
        elif total <= 1.0:
            self.difficulty = max(self.min_difficulty, self.difficulty - 0.05)

    def _averaged_metrics(self) -> dict[str, float]:
        if not self.history:
            return {}

        keys = {key for metrics in self.history for key in metrics}
        return {
            key: sum(metrics.get(key, 0.0) for metrics in self.history) / len(self.history)
            for key in keys
        }


class CurriculumCallback(TrainerCallback):
    def __init__(self, curriculum: SudokuCurriculum) -> None:
        self.curriculum = curriculum

    def on_step_end(self, metrics: StepMetrics) -> None:
        reward_metrics = dict(metrics.reward_function_means)
        reward_metrics["reward_mean"] = metrics.reward_mean
        self.curriculum.update(reward_metrics)
        print(f"curriculum_difficulty={self.curriculum.difficulty:.3f}")

    def on_completions(self, records: list[CompletionRecord]) -> None:
        _ = records
        return


class SudokuDataset:
    def __init__(self, *, size: int, curriculum: SudokuCurriculum, seed: int) -> None:
        self.size = size
        self.curriculum = curriculum
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _index: int) -> dict[str, DatasetValue]:
        puzzle = generate_puzzle(self.curriculum.difficulty, self.rng)
        return {
            "prompt": build_sudoku_prompt(puzzle),
            "solution": puzzle.solution,
            "puzzle": puzzle.puzzle,
            "size": puzzle.size,
            "box_rows": puzzle.box_rows,
            "box_cols": puzzle.box_cols,
            "difficulty": puzzle.difficulty,
            "blanks": puzzle.blanks,
        }


def generate_puzzle(difficulty: float, rng: random.Random) -> SudokuPuzzle:
    difficulty = min(1.0, max(0.0, difficulty))
    spec = _sample_spec(difficulty, rng)
    solution = _generate_solution(spec, rng)
    blanks = _target_blanks(spec, difficulty, rng)
    puzzle = _remove_cells(solution, spec, blanks, rng)
    actual_blanks = sum(cell == 0 for row in puzzle for cell in row)
    return SudokuPuzzle(
        size=spec.size,
        box_rows=spec.box_rows,
        box_cols=spec.box_cols,
        puzzle=puzzle,
        solution=solution,
        difficulty=difficulty,
        blanks=actual_blanks,
    )


def build_sudoku_prompt(puzzle: SudokuPuzzle) -> list[PromptMessage]:
    rows = "\n".join(" ".join(str(cell) for cell in row) for row in puzzle.puzzle)
    text = f"""
Solve this {puzzle.size}x{puzzle.size} Sudoku puzzle.

Rules:
- Replace every 0 with a number from 1 to {puzzle.size}.
- Each row must contain each number from 1 to {puzzle.size} exactly once.
- Each column must contain each number from 1 to {puzzle.size} exactly once.
- Each {puzzle.box_rows}x{puzzle.box_cols} box must contain each number from 1 to {puzzle.size} exactly once.
- Keep the given nonzero cells unchanged.

Think through the puzzle, then put only the completed grid inside <answer> tags.

Puzzle:
{rows}

Final answer format:
<answer>
row 1 numbers separated by spaces
row 2 numbers separated by spaces
...
</answer>
""".strip()
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def parse_solution_grid(text: str, size: int) -> Grid | None:
    match = ANSWER_RE.search(text)
    answer_text = match.group(1) if match is not None else text
    rows = [line.strip() for line in answer_text.strip().splitlines() if line.strip()]
    parsed: Grid = []

    for row in rows:
        values = [int(value) for value in NUMBER_RE.findall(row)]
        if values:
            parsed.append(values)

    if len(parsed) != size:
        return None

    return parsed


def has_correct_shape(grid: Grid | None, size: int) -> bool:
    return grid is not None and len(grid) == size and all(len(row) == size for row in grid)


def has_numbers_in_range(grid: Grid | None, size: int) -> bool:
    if grid is None or len(grid) != size or any(len(row) != size for row in grid):
        return False

    return all(1 <= cell <= size for row in grid for cell in row)


def preserves_given_cells(grid: Grid | None, puzzle: Grid, size: int) -> bool:
    if grid is None or len(grid) != size or any(len(row) != size for row in grid):
        return False

    return all(
        puzzle[row][col] == 0 or grid[row][col] == puzzle[row][col]
        for row in range(size)
        for col in range(size)
    )


def is_valid_solution(grid: Grid | None, size: int, box_rows: int, box_cols: int) -> bool:
    if grid is None or not has_numbers_in_range(grid, size):
        return False

    required = set(range(1, size + 1))
    if any(set(row) != required for row in grid):
        return False

    for col in range(size):
        if {grid[row][col] for row in range(size)} != required:
            return False

    for box_row in range(0, size, box_rows):
        for box_col in range(0, size, box_cols):
            values = {
                grid[row][col]
                for row in range(box_row, box_row + box_rows)
                for col in range(box_col, box_col + box_cols)
            }
            if values != required:
                return False

    return True


def correct_cell_fraction(grid: Grid | None, solution: Grid, size: int) -> float:
    if grid is None or len(grid) != size or any(len(row) != size for row in grid):
        return 0.0

    correct = sum(
        grid[row][col] == solution[row][col]
        for row in range(size)
        for col in range(size)
    )
    return correct / (size * size)


def exact_match(grid: Grid | None, solution: Grid, size: int) -> bool:
    return has_correct_shape(grid, size) and grid == solution


def _sample_spec(difficulty: float, rng: random.Random) -> SudokuSpec:
    specs = {
        "4": SudokuSpec(size=4, box_rows=2, box_cols=2, min_blanks=1, max_blanks=10),
        "6": SudokuSpec(size=6, box_rows=2, box_cols=3, min_blanks=6, max_blanks=24),
        "9": SudokuSpec(size=9, box_rows=3, box_cols=3, min_blanks=20, max_blanks=56),
    }
    if difficulty < 0.30:
        return specs["4"]
    if difficulty < 0.45:
        return rng.choice([specs["4"], specs["6"]])
    if difficulty < 0.70:
        return specs["6"]
    if difficulty < 0.85:
        return rng.choice([specs["6"], specs["9"]])
    return specs["9"]


def _target_blanks(spec: SudokuSpec, difficulty: float, rng: random.Random) -> int:
    if difficulty <= 0.0 and spec.size == 4:
        return 1

    local_difficulty = _local_difficulty(spec.size, difficulty)
    span = spec.max_blanks - spec.min_blanks
    target = spec.min_blanks + round(span * local_difficulty)
    jitter = rng.randint(-1, 1)
    return min(spec.max_blanks, max(spec.min_blanks, target + jitter))


def _local_difficulty(size: int, difficulty: float) -> float:
    if size == 4:
        return min(1.0, difficulty / 0.30)
    if size == 6:
        return min(1.0, max(0.0, (difficulty - 0.30) / 0.40))
    return min(1.0, max(0.0, (difficulty - 0.70) / 0.30))


def _generate_solution(spec: SudokuSpec, rng: random.Random) -> Grid:
    rows = _shuffled_groups(spec.box_rows, spec.box_cols, rng)
    cols = _shuffled_groups(spec.box_cols, spec.box_rows, rng)
    numbers = list(range(1, spec.size + 1))
    rng.shuffle(numbers)

    return [
        [numbers[_pattern(row, col, spec.box_rows, spec.box_cols)] for col in cols]
        for row in rows
    ]


def _shuffled_groups(group_size: int, groups: int, rng: random.Random) -> list[int]:
    group_indices = list(range(groups))
    rng.shuffle(group_indices)
    values: list[int] = []
    for group in group_indices:
        local = list(range(group_size))
        rng.shuffle(local)
        values.extend(group * group_size + item for item in local)
    return values


def _pattern(row: int, col: int, box_rows: int, box_cols: int) -> int:
    return (box_cols * (row % box_rows) + row // box_rows + col) % (box_rows * box_cols)


def _remove_cells(solution: Grid, spec: SudokuSpec, target_blanks: int, rng: random.Random) -> Grid:
    puzzle = [row[:] for row in solution]
    cells = [(row, col) for row in range(spec.size) for col in range(spec.size)]
    rng.shuffle(cells)
    blanks = 0

    for row, col in cells:
        if blanks >= target_blanks:
            break

        previous = puzzle[row][col]
        puzzle[row][col] = 0
        if _count_solutions(puzzle, spec, limit=2) == 1:
            blanks += 1
        else:
            puzzle[row][col] = previous

    return puzzle


def _count_solutions(grid: Grid, spec: SudokuSpec, *, limit: int) -> int:
    board = [row[:] for row in grid]
    return _search_count(board, spec, limit)


def _search_count(board: Grid, spec: SudokuSpec, limit: int) -> int:
    cell = _least_candidate_cell(board, spec)
    if cell is None:
        return 1

    row, col, candidates = cell
    count = 0
    for value in candidates:
        board[row][col] = value
        count += _search_count(board, spec, limit - count)
        if count >= limit:
            break
    board[row][col] = 0
    return count


def _least_candidate_cell(board: Grid, spec: SudokuSpec) -> tuple[int, int, list[int]] | None:
    best: tuple[int, int, list[int]] | None = None
    for row in range(spec.size):
        for col in range(spec.size):
            if board[row][col] != 0:
                continue

            candidates = _candidates(board, spec, row, col)
            if not candidates:
                return row, col, []
            if best is None or len(candidates) < len(best[2]):
                best = row, col, candidates

    return best


def _candidates(board: Grid, spec: SudokuSpec, row: int, col: int) -> list[int]:
    used = set(board[row])
    used.update(board[r][col] for r in range(spec.size))

    box_row = spec.box_rows * (row // spec.box_rows)
    box_col = spec.box_cols * (col // spec.box_cols)
    used.update(
        board[r][c]
        for r in range(box_row, box_row + spec.box_rows)
        for c in range(box_col, box_col + spec.box_cols)
    )
    return [value for value in range(1, spec.size + 1) if value not in used]


def grid_from_sequence(values: Sequence[object], index: int) -> Grid:
    grid = values[index]
    if _is_grid(grid):
        return cast(Grid, grid)
    raise TypeError("Expected Sudoku grid reward field.")


def int_from_sequence(values: Sequence[object], index: int) -> int:
    value = values[index]
    if isinstance(value, int):
        return value
    raise TypeError("Expected integer reward field.")


def _is_grid(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, list) and all(isinstance(cell, int) for cell in row)
        for row in value
    )


__all__ = [
    "CurriculumCallback",
    "Grid",
    "SudokuCurriculum",
    "SudokuDataset",
    "build_sudoku_prompt",
    "correct_cell_fraction",
    "exact_match",
    "generate_puzzle",
    "grid_from_sequence",
    "has_correct_shape",
    "has_numbers_in_range",
    "int_from_sequence",
    "is_valid_solution",
    "parse_solution_grid",
    "preserves_given_cells",
]
