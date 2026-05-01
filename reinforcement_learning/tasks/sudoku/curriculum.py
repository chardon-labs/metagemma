from dataclasses import dataclass, field

from rl_trainer.callbacks import TrainerCallback
from rl_trainer.types import CompletionRecord, StepMetrics


@dataclass
class SudokuCurriculum:
    difficulty: float = 0.35
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0
    window: int = 20
    history: list[dict[str, float]] = field(default_factory=list)

    def update(self, metrics: dict[str, float]) -> None:
        self.history.append(metrics)
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

        averaged = self._averaged_metrics()
        exact = averaged.get("exact_solution", 0.0)

        if exact >= 0.8:
            self.difficulty = min(self.max_difficulty, self.difficulty + 0.03)
        elif exact <= 0.2:
            self.difficulty = max(self.min_difficulty, self.difficulty - 0.05)

    def _averaged_metrics(self) -> dict[str, float]:
        if not self.history:
            return {}

        keys = {key for metrics in self.history for key in metrics}
        return {
            key: sum(metrics.get(key, 0.0) for metrics in self.history) / len(self.history)
            for key in keys
        }


class CurriculumCallback(TrainerCallback):
    def __init__(self, curriculum: SudokuCurriculum) -> None:
        self.curriculum = curriculum

    def on_step_end(self, metrics: StepMetrics) -> None:
        reward_metrics = dict(metrics.reward_function_means)
        reward_metrics["reward_mean"] = metrics.reward_mean
        self.curriculum.update(reward_metrics)
        print(f"curriculum_difficulty={self.curriculum.difficulty:.3f}")

    def on_completions(self, records: list[CompletionRecord]) -> None:
        _ = records
        return
