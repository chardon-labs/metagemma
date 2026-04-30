from typing import Protocol

from rich.console import Console
from rich.table import Table

from rl_trainer.types import CompletionRecord, StepMetrics


class TrainerCallback(Protocol):
    def on_step_end(self, metrics: StepMetrics) -> None: ...

    def on_completions(self, records: list[CompletionRecord]) -> None: ...


class PrintCallback:
    def __init__(self) -> None:
        self.console = Console()

    def on_step_end(self, metrics: StepMetrics) -> None:
        reward_parts = " ".join(
            f"{name}={value:.3f}" for name, value in sorted(metrics.reward_function_means.items())
        )
        self.console.print(
            f"step={metrics.step} loss={metrics.loss:.4f} reward={metrics.reward_mean:.3f} "
            f"reward_std={metrics.reward_std:.3f} len={metrics.completion_length_mean:.1f} "
            f"lr={metrics.learning_rate:.2e} ratio={metrics.mean_ratio:.3f} "
            f"clip={metrics.clip_ratio:.3f} {reward_parts}"
        )

    def on_completions(self, records: list[CompletionRecord]) -> None:
        table = Table(title="Completions", show_lines=True)
        table.add_column("Reward", justify="right")
        table.add_column("Adv", justify="right")
        table.add_column("Completion")
        for record in records[:3]:
            table.add_row(
                f"{record.reward:.3f}",
                f"{record.advantages:.3f}",
                record.completion[:600],
            )
        self.console.print(table)
