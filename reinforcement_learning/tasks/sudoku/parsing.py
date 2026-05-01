import re
from collections.abc import Sequence
from typing import cast

from tasks.sudoku.types import Grid

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"\d+")


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
