import random

from tasks.sudoku.curriculum import SudokuCurriculum
from tasks.sudoku.generator import generate_puzzle
from tasks.sudoku.prompts import build_sudoku_prompt
from tasks.sudoku.types import DatasetValue, SudokuPuzzle


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


class SinglePuzzleDataset:
    def __init__(self, *, puzzle: SudokuPuzzle, size: int) -> None:
        self.puzzle = puzzle
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _index: int) -> dict[str, DatasetValue]:
        return {
            "prompt": build_sudoku_prompt(self.puzzle),
            "solution": self.puzzle.solution,
            "puzzle": self.puzzle.puzzle,
            "size": self.puzzle.size,
            "box_rows": self.puzzle.box_rows,
            "box_cols": self.puzzle.box_cols,
            "difficulty": self.puzzle.difficulty,
            "blanks": self.puzzle.blanks,
        }
