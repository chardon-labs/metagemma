from tasks.sudoku.types import PromptMessage, SudokuPuzzle


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
