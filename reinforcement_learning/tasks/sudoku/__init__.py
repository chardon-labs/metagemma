from tasks.sudoku.curriculum import CurriculumCallback, SudokuCurriculum
from tasks.sudoku.dataset import SinglePuzzleDataset, SudokuDataset
from tasks.sudoku.generator import generate_puzzle
from tasks.sudoku.prompts import build_sudoku_prompt
from tasks.sudoku.rewards import (
    SUDOKU_REWARD_FUNCTIONS,
    exact_solution,
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
    "exact_solution",
    "generate_puzzle",
]
