from dataclasses import dataclass
from typing import TypeAlias

Grid: TypeAlias = list[list[int]]
PromptMessage: TypeAlias = dict[str, str | list[dict[str, str]]]
DatasetValue: TypeAlias = int | float | str | Grid | list[PromptMessage]
RewardValue: TypeAlias = int | float | str | Grid
RewardField: TypeAlias = list[RewardValue]


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
