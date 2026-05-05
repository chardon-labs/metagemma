from dataclasses import dataclass, field

from rl_trainer.callbacks import TrainerCallback
from rl_trainer.types import CompletionRecord, RLStepMetrics


@dataclass
class SudokuCurriculum:
    difficulty: float = 0.35
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0
    window: int = 10
    score_ema_decay: float = 0.8
    target_score: float = 0.5
    max_increase: float = 0.03
    max_decrease: float = 0.03
    history: list[dict[str, float]] = field(default_factory=list)
    score_ema: float | None = None
    last_delta: float = 0.0

    def update(self, metrics: dict[str, float]) -> None:
        self.history.append(metrics)
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

        exact = metrics.get("exact_solution", metrics.get("reward_mean", 0.0))
        self.score_ema = self._update_score_ema(exact)
        self.last_delta = self._difficulty_delta(self.score_ema)
        self.difficulty = min(self.max_difficulty, max(self.min_difficulty, self.difficulty + self.last_delta))

    def _update_score_ema(self, exact_score: float) -> float:
        if self.score_ema is None:
            return exact_score
        return self.score_ema_decay * self.score_ema + (1.0 - self.score_ema_decay) * exact_score

    def _difficulty_delta(self, score: float) -> float:
        if score >= self.target_score:
            span = max(1e-6, 1.0 - self.target_score)
            scale = min(1.0, (score - self.target_score) / span)
            return self.max_increase * scale

        span = max(1e-6, self.target_score)
        scale = min(1.0, (self.target_score - score) / span)
        return -self.max_decrease * scale

    def _averaged_metrics(self) -> dict[str, float]:
        if not self.history:
            return {}

        keys = {key for metrics in self.history for key in metrics}
        return {
            key: sum(metrics.get(key, 0.0) for metrics in self.history) / len(self.history)
            for key in keys
        }


@dataclass(frozen=True)
class SudokuStepMetrics:
    rl: RLStepMetrics

    @classmethod
    def from_rl_metrics(cls, metrics: RLStepMetrics) -> "SudokuStepMetrics":
        return cls(rl=metrics)

    def curriculum_metrics(self) -> dict[str, float]:
        reward = self.rl.unfiltered_reward or self.rl.reward
        metrics = dict(reward.function_means)
        metrics["reward_mean"] = reward.mean
        return metrics


class CurriculumCallback(TrainerCallback):
    def __init__(self, curriculum: SudokuCurriculum) -> None:
        self.curriculum = curriculum

    def on_step_end(self, metrics: RLStepMetrics) -> None:
        sudoku_metrics = SudokuStepMetrics.from_rl_metrics(metrics)
        self.curriculum.update(sudoku_metrics.curriculum_metrics())
        score_ema = self.curriculum.score_ema if self.curriculum.score_ema is not None else 0.0
        print(
            f"curriculum_difficulty={self.curriculum.difficulty:.3f} "
            f"score_ema={score_ema:.3f} delta={self.curriculum.last_delta:+.4f}"
        )

    def on_completions(self, records: list[CompletionRecord]) -> None:
        _ = records
        return
