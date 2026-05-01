from tasks.sudoku.curriculum import CurriculumCallback, SudokuCurriculum
from tasks.sudoku.dataset import SinglePuzzleDataset, SudokuDataset
from tasks.sudoku.generator import generate_puzzle
from tasks.sudoku.prompts import build_sudoku_prompt
from tasks.sudoku.rewards import (
    SUDOKU_REWARD_FUNCTIONS,
    correct_cells,
    correct_shape,
    exact_solution,
    numbers_in_range,
    respects_given_cells,
    solution_parses,
    valid_sudoku,
)
from tasks.sudoku.types import Grid, SudokuPuzzle, SudokuSpec

__all__ = [
    "CurriculumCallback",
    "Grid",
    "SUDOKU_REWARD_FUNCTIONS",
    "SinglePuzzleDataset",
    "SudokuCurriculum",
    "SudokuDataset",
    "SudokuPuzzle",
    "SudokuSpec",
    "build_sudoku_prompt",
    "correct_cells",
    "correct_shape",
    "exact_solution",
    "generate_puzzle",
    "numbers_in_range",
    "respects_given_cells",
    "solution_parses",
    "valid_sudoku",
]
