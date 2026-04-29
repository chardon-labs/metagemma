import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
from datasets import Dataset
from safetensors import safe_open
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer
from unsloth import (
    FastVisionModel,
    check_python_modules,
    create_locked_down_function,
    execute_with_time_limit,
)

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 32
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
SUDOKU_SIZE = 9
BOX_SIZE = 3
DEFAULT_DIFFICULTY = 40
DATASET_SIZE = 1000
MAX_STRATEGY_MOVES = 100
STRATEGY_TIMEOUT_SECONDS = 10
MAX_STEPS = 60
OUTPUT_DIR = Path("outputs")
ADAPTER_DIR = Path("gemma_4_lora")
PROMPT = """
Create a Sudoku solving strategy using only native Python built-in functions without any import statements.
You are given two lists of lists (9x9 grids):
- board: current state (0 means empty)
- initial: starting puzzle (0 means was empty, numbers are fixed)

Return a tuple (row, col, number) for the next move.
- row: 0-8 (row index)
- col: 0-8 (column index)
- number: 1-9 (digit to place)

Only place numbers in cells that are both empty in initial and empty in board.
Use Sudoku rules: no duplicates in rows, columns, or 3x3 boxes.
Output your function in backticks:
```python
def strategy(board, initial):
    return (row, col, number)
```
All helper functions must be inside def strategy. Output only the function.
""".strip()

Board: TypeAlias = list[list[int]]
Move: TypeAlias = tuple[int, int, int]
StrategyResult: TypeAlias = Move | list[int]
Strategy: TypeAlias = Callable[[Board, Board], StrategyResult]
CompletionMessage: TypeAlias = dict[str, str]
Completion: TypeAlias = list[CompletionMessage]
RewardKwarg: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | Board
    | list[str]
    | list[int]
    | list[float]
    | list[Completion]
)

PRINT_COUNTER = 0


def _is_valid_placement(board: Board, row: int, col: int, num: int) -> bool:
    if num in board[row]:
        return False

    if num in [board[r][col] for r in range(SUDOKU_SIZE)]:
        return False

    box_row = BOX_SIZE * (row // BOX_SIZE)
    box_col = BOX_SIZE * (col // BOX_SIZE)
    for r in range(box_row, box_row + BOX_SIZE):
        for c in range(box_col, box_col + BOX_SIZE):
            if board[r][c] == num:
                return False

    return True


def _solve_sudoku(board: Board) -> bool:
    for row in range(SUDOKU_SIZE):
        for col in range(SUDOKU_SIZE):
            if board[row][col] == 0:
                for num in range(1, SUDOKU_SIZE + 1):
                    if _is_valid_placement(board, row, col, num):
                        board[row][col] = num
                        if _solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True


def _generate_complete_board(rng: random.Random) -> Board:
    board = [[0 for _ in range(SUDOKU_SIZE)] for _ in range(SUDOKU_SIZE)]

    for box in range(BOX_SIZE):
        nums = list(range(1, SUDOKU_SIZE + 1))
        rng.shuffle(nums)
        for i in range(BOX_SIZE):
            for j in range(BOX_SIZE):
                board[box * BOX_SIZE + i][box * BOX_SIZE + j] = nums[
                    i * BOX_SIZE + j
                ]

    _solve_sudoku(board)
    return board


@dataclass
class SudokuGame:
    difficulty: int = DEFAULT_DIFFICULTY
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _board: Board = field(init=False, repr=False)
    _solution: Board = field(init=False, repr=False)
    _initial_board: Board = field(init=False, repr=False)
    _moves: int = field(default=0, init=False, repr=False)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

        complete_board = _generate_complete_board(self._rng)
        self._solution = copy.deepcopy(complete_board)
        self._board = copy.deepcopy(complete_board)
        cells = [(r, c) for r in range(SUDOKU_SIZE) for c in range(SUDOKU_SIZE)]
        self._rng.shuffle(cells)

        for r, c in cells[: self.difficulty]:
            self._board[r][c] = 0

        self._initial_board = copy.deepcopy(self._board)
        self._update_state()

    def board(self) -> Board:
        return [row[:] for row in self._board]

    def initial_board(self) -> Board:
        return [row[:] for row in self._initial_board]

    def state(self) -> str:
        return self._state

    def moves(self) -> int:
        return self._moves

    def place_number(self, row: int, col: int, num: int) -> bool:
        if not (0 <= row < SUDOKU_SIZE and 0 <= col < SUDOKU_SIZE):
            self._state = "failed"
            return False

        if not (1 <= num <= SUDOKU_SIZE):
            self._state = "failed"
            return False

        if self._initial_board[row][col] != 0:
            self._state = "failed"
            return False

        if self._board[row][col] != 0:
            self._state = "failed"
            return False

        if not _is_valid_placement(self._board, row, col, num):
            self._state = "failed"
            return False

        self._board[row][col] = num
        self._moves += 1
        self._update_state()
        return True

    def pretty(self, colors: bool = True) -> str:
        reset = "\x1b[0m"
        initial = "\x1b[38;5;45m"
        placed = "\x1b[38;5;226m"
        empty = "\x1b[38;5;239m"
        lines = ["+-------+-------+-------+"]

        for row in range(SUDOKU_SIZE):
            row_values = ["|"]
            for col in range(SUDOKU_SIZE):
                num = self._board[row][col]
                if colors and num == 0:
                    row_values.append(f"{empty}.{reset}")
                elif colors and self._initial_board[row][col] != 0:
                    row_values.append(f"{initial}{num}{reset}")
                elif colors:
                    row_values.append(f"{placed}{num}{reset}")
                else:
                    row_values.append(str(num) if num != 0 else ".")

                if col % BOX_SIZE == BOX_SIZE - 1:
                    row_values.append("|")

            lines.append(" ".join(row_values))
            if row % BOX_SIZE == BOX_SIZE - 1:
                lines.append("+-------+-------+-------+")

        return "\n".join(lines)

    def _update_state(self) -> None:
        if all(
            self._board[r][c] != 0
            for r in range(SUDOKU_SIZE)
            for c in range(SUDOKU_SIZE)
        ):
            self._state = "success" if self._board == self._solution else "failed"
        else:
            self._state = "ongoing"


def _execute_strategy(strategy: Strategy, game: SudokuGame) -> tuple[int, str]:
    valid_moves = 0

    while game.state() == "ongoing" and valid_moves < MAX_STRATEGY_MOVES:
        try:
            result = strategy(game.board(), game.initial_board())
            if not isinstance(result, tuple | list) or len(result) != 3:
                return valid_moves, "failed"

            row, col, num = result
            if not all(isinstance(value, int) for value in (row, col, num)):
                return valid_moves, "failed"

            if game.place_number(row, col, num):
                valid_moves += 1
            else:
                return valid_moves, "failed"
        except Exception:
            return valid_moves, "failed"

    if valid_moves >= MAX_STRATEGY_MOVES and game.state() == "ongoing":
        return valid_moves, "failed"

    return valid_moves, game.state()


@execute_with_time_limit(STRATEGY_TIMEOUT_SECONDS)
def execute_strategy(strategy: Strategy, game: SudokuGame) -> tuple[int, str]:
    return _execute_strategy(strategy, game)


def extract_function(text: str) -> str | None:
    if text.count("```") < 2:
        return None

    first = text.find("```") + 3
    second = text.find("```", first)
    function = text[first:second].strip().removeprefix("python\n")
    def_index = function.find("def")
    if def_index < 0:
        return None

    function = function[def_index:]
    if function.startswith("def strategy(board, initial):"):
        return function

    return None


def function_works(
    completions: list[Completion], **_kwargs: RewardKwarg
) -> list[float]:
    scores: list[float] = []

    for completion in completions:
        function = extract_function(completion[0]["content"])
        if function is None:
            scores.append(-2.0)
            continue

        ok, info = check_python_modules(function)
        if not ok or "error" in info:
            scores.append(-2.0)
            continue

        try:
            create_locked_down_function(function)
            scores.append(1.0)
        except Exception:
            scores.append(-1.0)

    return scores


def no_cheating(completions: list[Completion], **_kwargs: RewardKwarg) -> list[float]:
    scores: list[float] = []

    for completion in completions:
        function = extract_function(completion[0]["content"])
        if function is None:
            scores.append(-1.0)
            continue

        ok, _info = check_python_modules(function)
        scores.append(1.0 if ok else -20.0)

    return scores


def strategy_succeeds(
    completions: list[Completion], **_kwargs: RewardKwarg
) -> list[float]:
    global PRINT_COUNTER

    scores: list[float] = []
    seed = int(np.random.randint(10_000))

    for completion in completions:
        printed = False
        function = extract_function(completion[0]["content"])

        if PRINT_COUNTER % 5 == 0:
            printed = True
            print("\n" + "=" * 60)
            print(function)
            print("=" * 60)
        PRINT_COUNTER += 1

        if function is None:
            scores.append(0.0)
            continue

        ok, info = check_python_modules(function)
        if not ok or "error" in info:
            scores.append(0.0)
            continue

        try:
            new_strategy = create_locked_down_function(function)
        except Exception:
            scores.append(0.0)
            continue

        try:
            game = SudokuGame(difficulty=DEFAULT_DIFFICULTY, seed=seed)
            valid_moves, game_state = execute_strategy(new_strategy, game)
            if valid_moves == DEFAULT_DIFFICULTY:
                game_state = "success"

            print(f"\nValid moves: {valid_moves}, Final state: {game_state}")

            if not printed:
                print("Strategy:")
                print(function[:200] + "..." if len(function) > 200 else function)

            print("\nFinal board:")
            print(game.pretty())

            if game_state == "success":
                scores.append(30.0)
            elif valid_moves > 0:
                scores.append(valid_moves * 0.2)
            else:
                scores.append(-2.0)
        except TimeoutError:
            print("Timeout")
            scores.append(-1.0)
        except Exception as exc:
            print(f"Exception: {str(exc)[:100]}")
            scores.append(-3.0)

    return scores


def load_model_and_tokenizer():
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_STATE,
    )
    return model, tokenizer


def build_dataset() -> Dataset:
    return Dataset.from_list(
        [{"prompt": [{"role": "user", "content": PROMPT}], "answer": 0}]
        * DATASET_SIZE
    )


def get_prompt_length(tokenizer) -> int:
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            add_generation_prompt=True,
        )
    )


def build_training_args(max_completion_length: int) -> GRPOConfig:
    return GRPOConfig(
        temperature=1.0,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_completion_length=max_completion_length,
        max_steps=MAX_STEPS,
        save_steps=100,
        report_to="none",
        output_dir=str(OUTPUT_DIR),
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="bnpo",
        mask_truncated_completions=True,
    )


def run_base_generation(model, tokenizer) -> None:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text=text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    print("=" * 50)
    print("BASE MODEL OUTPUT")
    print("=" * 50)
    model.generate(
        **inputs,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
        max_new_tokens=128,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )


def verify_saved_lora(adapter_dir: Path) -> None:
    adapter_path = adapter_dir / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt") as tensors:
        for key in tensors.keys():
            tensor = tensors.get_tensor(key)
            zero_fraction = (tensor == 0).sum().item() / tensor.numel()
            if zero_fraction == 1.0:
                raise ValueError(f"{key} is all zeros")


def run_trained_generation(model, tokenizer) -> None:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    model.generate(
        **tokenizer(images=None, text=text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=512,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )


def main() -> None:
    model, tokenizer = load_model_and_tokenizer()
    dataset = build_dataset()
    prompt_length = get_prompt_length(tokenizer)
    max_completion_length = MAX_SEQ_LENGTH - (prompt_length + 1)

    print(f"Maximum prompt length: {prompt_length}")
    print("Dataset sample:")
    print(dataset[0])

    run_base_generation(model, tokenizer)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[function_works, no_cheating, strategy_succeeds],
        args=build_training_args(max_completion_length),
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    verify_saved_lora(ADAPTER_DIR)
    run_trained_generation(model, tokenizer)


if __name__ == "__main__":
    main()
