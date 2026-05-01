import random

from tasks.sudoku.types import Grid, SudokuPuzzle, SudokuSpec


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
