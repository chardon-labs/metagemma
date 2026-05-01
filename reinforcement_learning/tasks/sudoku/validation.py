from tasks.sudoku.types import Grid


def has_correct_shape(grid: Grid | None, size: int) -> bool:
    return grid is not None and len(grid) == size and all(len(row) == size for row in grid)


def exact_match(grid: Grid | None, solution: Grid, size: int) -> bool:
    return has_correct_shape(grid, size) and grid == solution
