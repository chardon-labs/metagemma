import json
import random
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, cast

from rl_trainer.sudoku import (
    Grid,
    SudokuPuzzle,
    exact_match,
    generate_puzzle,
    parse_solution_grid,
)

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "google/gemma-4-e2b"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "lm_studio_sudoku_probe.json"

RANDOM_SEED = 3407
MAX_CANDIDATE_PUZZLES = 80
SAMPLES_PER_PUZZLE = 8
COARSE_SAMPLES_PER_PUZZLE = 1
MAX_PARALLEL_REQUESTS = 2
ACCEPT_MIN_SOLVE_RATE = 0.25
ACCEPT_MAX_SOLVE_RATE = 0.75
TARGET_SOLVE_RATE = 0.50
REQUEST_TIMEOUT_SECONDS = 120
SLEEP_BETWEEN_REQUESTS_SECONDS = 0.05

TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 2048

DIFFICULTY_CYCLE = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]
REFINE_DIFFICULTY_RADIUS = 0.04
REFINE_DIFFICULTY_STEP = 0.005

JSONValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class SampleResult:
    solved: bool
    parsed: Grid | None
    response: str


@dataclass(frozen=True)
class CandidateResult:
    puzzle: SudokuPuzzle
    solve_rate: float
    samples: list[SampleResult]


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    refinement_difficulties = find_refinement_difficulties(rng)
    best: CandidateResult | None = None

    for candidate_index in range(MAX_CANDIDATE_PUZZLES):
        difficulty = refinement_difficulties[candidate_index % len(refinement_difficulties)]
        puzzle = generate_puzzle(difficulty, rng)
        result = evaluate_candidate(puzzle)
        best = closer_to_target(result, best)
        print_candidate(candidate_index, result, best)

        if ACCEPT_MIN_SOLVE_RATE <= result.solve_rate <= ACCEPT_MAX_SOLVE_RATE:
            write_result(result, accepted=True)
            print(f"Accepted puzzle written to {OUTPUT_PATH}")
            return

    if best is None:
        raise RuntimeError("No candidate puzzles were evaluated.")

    write_result(best, accepted=False)
    print(f"No puzzle landed in target band. Best candidate written to {OUTPUT_PATH}")


def find_refinement_difficulties(rng: random.Random) -> list[float]:
    previous_solved_difficulty: float | None = None

    print("Coarse scan: one sample per difficulty until solved -> unsolved transition.")
    for difficulty in DIFFICULTY_CYCLE:
        puzzle = generate_puzzle(difficulty, rng)
        result = evaluate_candidate(puzzle, sample_count=COARSE_SAMPLES_PER_PUZZLE)
        solved = result.solve_rate > 0.0
        print(
            f"coarse difficulty={difficulty:.2f} size={puzzle.size} blanks={puzzle.blanks} "
            f"solved={int(solved)}"
        )

        if previous_solved_difficulty is not None and not solved:
            difficulties = midpoint_out_difficulties(
                previous_solved_difficulty,
                difficulty,
                radius=REFINE_DIFFICULTY_RADIUS,
            )
            print(
                f"Refining around transition {previous_solved_difficulty:.2f} -> {difficulty:.2f}: "
                f"{[round(value, 3) for value in difficulties]}"
            )
            return difficulties

        if solved:
            previous_solved_difficulty = difficulty

    if previous_solved_difficulty is None:
        print("No coarse difficulty solved. Refining around easiest difficulties.")
        return difficulty_range(0.0, min(0.12, DIFFICULTY_CYCLE[-1]))

    print("All coarse difficulties solved. Refining around hardest scanned difficulties.")
    return midpoint_out_difficulties(
        max(0.0, DIFFICULTY_CYCLE[-2]),
        DIFFICULTY_CYCLE[-1],
        radius=REFINE_DIFFICULTY_RADIUS,
    )


def midpoint_out_difficulties(lower_transition: float, upper_transition: float, *, radius: float) -> list[float]:
    lower = max(0.0, lower_transition - radius)
    upper = min(1.0, upper_transition + radius)
    midpoint = round((lower_transition + upper_transition) / 2, 3)
    values = difficulty_range(lower, upper)
    return sorted(values, key=lambda value: (abs(value - midpoint), value))


def difficulty_range(lower: float, upper: float) -> list[float]:
    values: list[float] = []
    value = lower
    while value <= upper + 1e-9:
        values.append(round(value, 3))
        value += REFINE_DIFFICULTY_STEP
    return values


def evaluate_candidate(puzzle: SudokuPuzzle, *, sample_count: int = SAMPLES_PER_PUZZLE) -> CandidateResult:
    if sample_count <= 1:
        sample = evaluate_one_sample(puzzle, sample_index=0, sample_count=sample_count)
        return CandidateResult(
            puzzle=puzzle,
            solve_rate=1.0 if sample.solved else 0.0,
            samples=[sample],
        )

    samples = run_parallel_samples(puzzle, sample_count)
    solve_rate = sum(sample.solved for sample in samples) / len(samples)
    return CandidateResult(puzzle=puzzle, solve_rate=solve_rate, samples=samples)


def run_parallel_samples(puzzle: SudokuPuzzle, sample_count: int) -> list[SampleResult]:
    samples: list[SampleResult] = []
    lock = threading.Lock()
    semaphore = threading.Semaphore(MAX_PARALLEL_REQUESTS)
    threads: list[threading.Thread] = []

    def worker(sample_index: int) -> None:
        with semaphore:
            try:
                sample = evaluate_one_sample(puzzle, sample_index=sample_index, sample_count=sample_count)
            except RuntimeError as exc:
                print(f"  sample={sample_index + 1:02d}/{sample_count} error={exc}")
                sample = SampleResult(solved=False, parsed=None, response=str(exc))
        with lock:
            samples.append(sample)

    for sample_index in range(sample_count):
        thread = threading.Thread(target=worker, args=(sample_index,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return samples


def evaluate_one_sample(puzzle: SudokuPuzzle, *, sample_index: int, sample_count: int) -> SampleResult:
    response = call_lm_studio(build_prompt(puzzle))
    parsed = parse_solution_grid(response, puzzle.size)
    solved = exact_match(parsed, puzzle.solution, puzzle.size)
    print(f"  sample={sample_index + 1:02d}/{sample_count} solved={int(solved)}")
    return SampleResult(solved=solved, parsed=parsed, response=response)


def build_prompt(puzzle: SudokuPuzzle) -> str:
    rows = "\n".join(" ".join(str(cell) for cell in row) for row in puzzle.puzzle)
    return f"""
Solve this {puzzle.size}x{puzzle.size} Sudoku puzzle.

Rules:
- Replace every 0 with a number from 1 to {puzzle.size}.
- Each row must contain each number from 1 to {puzzle.size} exactly once.
- Each column must contain each number from 1 to {puzzle.size} exactly once.
- Each {puzzle.box_rows}x{puzzle.box_cols} box must contain each number from 1 to {puzzle.size} exactly once.
- Keep the given nonzero cells unchanged.

Do not write reasoning. Put only the completed grid inside <answer> tags.

Puzzle:
{rows}

Final answer format:
<answer>
row 1 numbers separated by spaces
row 2 numbers separated by spaces
...
</answer>
""".strip()


def call_lm_studio(prompt: str) -> str:
    payload: JSONObject = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        LM_STUDIO_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to call LM Studio at {LM_STUDIO_URL}: {exc}") from exc

    data = cast(JSONObject, json.loads(raw))
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"LM Studio response did not contain choices: {raw}")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"LM Studio response choice was malformed: {raw}")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"LM Studio response did not contain a message: {raw}")

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"LM Studio response message did not contain text content: {raw}")

    return content


def closer_to_target(candidate: CandidateResult, current: CandidateResult | None) -> CandidateResult:
    if current is None:
        return candidate

    candidate_distance = abs(candidate.solve_rate - TARGET_SOLVE_RATE)
    current_distance = abs(current.solve_rate - TARGET_SOLVE_RATE)
    return candidate if candidate_distance < current_distance else current


def print_candidate(index: int, result: CandidateResult, best: CandidateResult | None) -> None:
    best_rate = -1.0 if best is None else best.solve_rate
    print(
        f"candidate={index + 1:02d} size={result.puzzle.size} blanks={result.puzzle.blanks} "
        f"difficulty={result.puzzle.difficulty:.2f} solve_rate={result.solve_rate:.3f} "
        f"best={best_rate:.3f}"
    )


def write_result(result: CandidateResult, *, accepted: bool) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accepted": accepted,
        "model": MODEL,
        "lm_studio_url": LM_STUDIO_URL,
        "samples_per_puzzle": SAMPLES_PER_PUZZLE,
        "target_solve_rate": TARGET_SOLVE_RATE,
        "accept_min_solve_rate": ACCEPT_MIN_SOLVE_RATE,
        "accept_max_solve_rate": ACCEPT_MAX_SOLVE_RATE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "solve_rate": result.solve_rate,
        "puzzle": asdict(result.puzzle),
        "prompt": build_prompt(result.puzzle),
        "samples": [
            {
                "solved": sample.solved,
                "parsed": sample.parsed,
                "response": sample.response,
            }
            for sample in result.samples
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
