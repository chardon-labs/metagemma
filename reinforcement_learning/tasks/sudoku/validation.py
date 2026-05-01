from tasks.sudoku.types import Grid


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
