from typing import Protocol

import plotille
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from rl_trainer.types import CompletionRecord, StepMetrics

MAX_HISTORY = 30
RECENT_STEPS = 5
MAX_COMPLETIONS = 2
MAX_COMPLETION_CHARS = 140
PLOT_HEIGHT = 8
PLOT_MARGIN = 22


class TrainerCallback(Protocol):
    def on_step_end(self, metrics: StepMetrics) -> None: ...

    def on_completions(self, records: list[CompletionRecord]) -> None: ...


class PrintCallback:
    def __init__(self) -> None:
        self.history: list[StepMetrics] = []
        self.latest_completions: list[CompletionRecord] = []
        self.console = Console()
        self.live = Live(self._render(), console=self.console, refresh_per_second=4, transient=False)
        self.started = False
        self.closed = False

    def on_step_end(self, metrics: StepMetrics) -> None:
        self._start()
        self.history.append(metrics)
        self.history = self.history[-MAX_HISTORY:]
        self.live.update(self._render(), refresh=True)

    def on_completions(self, records: list[CompletionRecord]) -> None:
        self._start()
        self.latest_completions = records[:MAX_COMPLETIONS]
        self.live.update(self._render(), refresh=True)

    def close(self) -> None:
        if self.closed:
            return
        if self.started:
            self.live.stop()
        self.closed = True

    def _start(self) -> None:
        if self.started:
            return
        self.live.start()
        self.started = True

    def _render(self) -> Group:
        return Group(
            Panel(self._summary_line(), title="Training", padding=(0, 1)),
            Panel(self._reward_plot(), title="Reward over steps", padding=(0, 1)),
            Columns(
                [
                    Panel(
                        self._history_table(),
                        title=f"Recent ({min(len(self.history), RECENT_STEPS)})",
                        padding=(0, 1),
                    ),
                    Panel(self._reward_function_table(), title="Rewards", padding=(0, 1)),
                ],
                expand=True,
                equal=False,
            ),
            Panel(self._completion_table(), title="Latest completions", padding=(0, 1)),
        )

    def _summary_line(self) -> str:
        latest = self.history[-1] if self.history else None
        if latest is None:
            return "step=-  reward=-  len=-  loss=-"

        return (
            f"step={latest.step}  reward={latest.reward_mean:.3f}±{latest.reward_std:.3f}  "
            f"len={latest.completion_length_mean:.1f} active={latest.active_completion_length_mean:.1f}  "
            f"loss_seq={latest.loss_sequence_fraction:.2f}  loss={latest.loss:.4f}  "
            f"lr={latest.learning_rate:.2e} raw_grad={latest.grad_norm:.3f} "
            f"clip={latest.grad_clip_scale:.2f}{self._timing_summary(latest)}{self._sync_summary(latest)}"
        )

    def _timing_summary(self, metrics: StepMetrics) -> str:
        timings = metrics.timings
        if timings is None:
            return ""

        return (
            f"  t=roll:{timings.rollout_seconds:.2f}s "
            f"back:{timings.backward_seconds:.2f}s opt:{timings.optimizer_seconds:.2f}s"
        )

    def _sync_summary(self, metrics: StepMetrics) -> str:
        stats = metrics.rollout_sync_stats
        if stats is None:
            return ""

        gib = stats.synced_bytes / (1024**3)
        return (
            f"  sync=step:{stats.step} tensors:{stats.synced_tensors}/{stats.loaded_tensors} "
            f"bytes:{gib:.2f}GiB"
        )

    def _history_table(self) -> Table:
        table = Table(box=box.SIMPLE, padding=(0, 1), expand=False)
        table.add_column("Step", justify="right")
        table.add_column("Reward", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Len", justify="right")
        table.add_column("Active", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("RawGrad", justify="right")
        table.add_column("Clip", justify="right")
        table.add_column("Roll", justify="right")
        table.add_column("Back", justify="right")
        for metrics in self.history[-RECENT_STEPS:]:
            timings = metrics.timings
            table.add_row(
                str(metrics.step),
                f"{metrics.reward_mean:.3f}",
                f"{metrics.reward_std:.3f}",
                f"{metrics.completion_length_mean:.1f}",
                f"{metrics.active_completion_length_mean:.1f}",
                f"{metrics.loss:.4f}",
                f"{metrics.grad_norm:.3f}",
                f"{metrics.grad_clip_scale:.2f}",
                "-" if timings is None else f"{timings.rollout_seconds:.2f}",
                "-" if timings is None else f"{timings.backward_seconds:.2f}",
            )
        return table

    def _reward_plot(self) -> str:
        if not self.history:
            return "No reward history yet."

        steps = [metrics.step for metrics in self.history]
        rewards = [metrics.reward_mean for metrics in self.history]
        width = max(24, self.console.size.width - PLOT_MARGIN)
        return plotille.plot(
            steps,
            rewards,
            width=width,
            height=PLOT_HEIGHT,
            X_label="step",
            Y_label="reward",
            origin=False,
        )

    def _reward_function_table(self) -> Table:
        table = Table(box=box.SIMPLE, padding=(0, 1), expand=False)
        table.add_column("Reward", overflow="fold")
        table.add_column("Mean", justify="right")
        latest = self.history[-1] if self.history else None
        if latest is None:
            return table

        for name, value in sorted(latest.reward_function_means.items()):
            table.add_row(name, f"{value:.3f}")
        return table

    def _completion_table(self) -> Table:
        table = Table(box=box.SIMPLE, padding=(0, 1), expand=False)
        table.add_column("Reward", justify="right")
        table.add_column("Adv", justify="right")
        table.add_column("Chars", justify="right")
        table.add_column("Completion", overflow="fold")
        for record in self.latest_completions:
            table.add_row(
                f"{record.reward:.3f}",
                f"{record.advantages:.3f}",
                str(len(record.completion)),
                self._compact_completion(record.completion),
            )
        return table

    def _compact_completion(self, completion: str) -> str:
        compact = " ".join(completion.split())
        if len(compact) <= MAX_COMPLETION_CHARS:
            return compact
        return compact[: MAX_COMPLETION_CHARS - 3] + "..."
