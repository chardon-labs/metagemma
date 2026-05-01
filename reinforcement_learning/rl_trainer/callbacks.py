from typing import Protocol

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rl_trainer.types import CompletionRecord, StepMetrics

MAX_HISTORY = 30
RECENT_STEPS = 5
MAX_COMPLETIONS = 3
MAX_COMPLETION_CHARS = 220
PLOT_WIDTH = 64
PLOT_HEIGHT = 10


class TrainerCallback(Protocol):
    def on_step_end(self, metrics: StepMetrics) -> None: ...

    def on_completions(self, records: list[CompletionRecord]) -> None: ...


class PrintCallback:
    def __init__(self) -> None:
        self.history: list[StepMetrics] = []
        self.latest_completions: list[CompletionRecord] = []
        self.live = Live(self._render(), refresh_per_second=4, transient=False)
        self.live.start()

    def on_step_end(self, metrics: StepMetrics) -> None:
        self.history.append(metrics)
        self.history = self.history[-MAX_HISTORY:]
        self.live.update(self._render(), refresh=True)

    def on_completions(self, records: list[CompletionRecord]) -> None:
        self.latest_completions = records[:MAX_COMPLETIONS]
        self.live.update(self._render(), refresh=True)

    def close(self) -> None:
        self.live.stop()

    def _render(self) -> Group:
        return Group(
            Panel(self._summary_table(), title="Training"),
            Panel(self._reward_plot(), title="Reward over steps"),
            Panel(self._history_table(), title=f"Recent steps ({min(len(self.history), RECENT_STEPS)})"),
            Panel(self._reward_function_table(), title="Reward functions"),
            Panel(self._completion_table(), title="Latest completions"),
        )

    def _summary_table(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        latest = self.history[-1] if self.history else None
        if latest is None:
            table.add_row("step", "-")
            table.add_row("reward", "-")
            table.add_row("completion length", "-")
            table.add_row("loss", "-")
            return table

        table.add_row("step", str(latest.step))
        table.add_row("reward mean", f"{latest.reward_mean:.3f}")
        table.add_row("reward std", f"{latest.reward_std:.3f}")
        table.add_row("loss", f"{latest.loss:.4f}")
        table.add_row("completion length", f"{latest.completion_length_mean:.1f}")
        table.add_row("active length", f"{latest.active_completion_length_mean:.1f}")
        table.add_row("loss sequence fraction", f"{latest.loss_sequence_fraction:.2f}")
        table.add_row("learning rate", f"{latest.learning_rate:.2e}")
        table.add_row("grad norm", f"{latest.grad_norm:.3f}")
        return table

    def _history_table(self) -> Table:
        table = Table(expand=True)
        table.add_column("Step", justify="right")
        table.add_column("Reward", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Len", justify="right")
        table.add_column("Active", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Grad", justify="right")
        for metrics in self.history[-RECENT_STEPS:]:
            table.add_row(
                str(metrics.step),
                f"{metrics.reward_mean:.3f}",
                f"{metrics.reward_std:.3f}",
                f"{metrics.completion_length_mean:.1f}",
                f"{metrics.active_completion_length_mean:.1f}",
                f"{metrics.loss:.4f}",
                f"{metrics.grad_norm:.3f}",
            )
        return table

    def _reward_plot(self) -> Text:
        if not self.history:
            return Text("No reward history yet.")

        points = [(metrics.step, metrics.reward_mean) for metrics in self.history]
        min_step = points[0][0]
        max_step = points[-1][0]
        rewards = [reward for _, reward in points]
        min_reward = min(rewards)
        max_reward = max(rewards)
        reward_span = max(max_reward - min_reward, 1e-9)
        step_span = max(max_step - min_step, 1)

        canvas = [[" " for _ in range(PLOT_WIDTH)] for _ in range(PLOT_HEIGHT)]
        mapped_points: list[tuple[int, int]] = []
        for step, reward in points:
            x = round((step - min_step) / step_span * (PLOT_WIDTH - 1))
            y = PLOT_HEIGHT - 1 - round((reward - min_reward) / reward_span * (PLOT_HEIGHT - 1))
            mapped_points.append((x, y))

        for index, (x, y) in enumerate(mapped_points):
            canvas[y][x] = "*"
            if index == 0:
                continue

            prev_x, prev_y = mapped_points[index - 1]
            distance = max(abs(x - prev_x), abs(y - prev_y), 1)
            for offset in range(1, distance):
                interp_x = round(prev_x + (x - prev_x) * offset / distance)
                interp_y = round(prev_y + (y - prev_y) * offset / distance)
                if canvas[interp_y][interp_x] == " ":
                    canvas[interp_y][interp_x] = "."

        rows = []
        for row_index, row in enumerate(canvas):
            reward = max_reward - reward_span * row_index / max(PLOT_HEIGHT - 1, 1)
            rows.append(f"{reward:>8.3f} |{''.join(row)}")

        rows.append(f"{'':>8} +{'-' * PLOT_WIDTH}")
        rows.append(f"{'step':>8}  {min_step:<{PLOT_WIDTH // 2}}{max_step:>{PLOT_WIDTH - (PLOT_WIDTH // 2)}}")
        return Text("\n".join(rows))

    def _reward_function_table(self) -> Table:
        table = Table(expand=True)
        table.add_column("Reward", overflow="fold")
        table.add_column("Mean", justify="right")
        latest = self.history[-1] if self.history else None
        if latest is None:
            return table

        for name, value in sorted(latest.reward_function_means.items()):
            table.add_row(name, f"{value:.3f}")
        return table

    def _completion_table(self) -> Table:
        table = Table(expand=True)
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
