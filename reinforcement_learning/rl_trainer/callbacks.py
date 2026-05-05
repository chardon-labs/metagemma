import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import plotille
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rl_trainer.types import CompletionRecord, RLStepMetrics, StepTimings, ValidationMetrics

MAX_HISTORY = 120
RECENT_STEPS = 5
MAX_COMPLETIONS = 2
MAX_COMPLETION_CHARS = 140
PLOT_HEIGHT = 8
PLOT_MIN_WIDTH = 24
PLOT_MAX_WIDTH = 140
PLOT_PANEL_OVERHEAD = 4
REWARD_Y_MIN = 0.0
REWARD_Y_MAX = 1.0


MetricValueFn = Callable[[RLStepMetrics], float | None]


@dataclass(frozen=True)
class MetricSeries:
    label: str
    y_label: str
    value_fn: MetricValueFn
    y_limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class RecentMetric:
    label: str
    value_fn: MetricValueFn
    precision: int = 3
    signed: bool = False


class TrainerCallback(Protocol):
    def on_step_end(self, metrics: RLStepMetrics) -> ValidationMetrics | None: ...

    def on_completions(self, records: list[CompletionRecord]) -> None: ...


class PrintCallback:
    def __init__(
        self,
        *,
        reward_y_limits: tuple[float, float] | None = (REWARD_Y_MIN, REWARD_Y_MAX),
        metric_series: list[MetricSeries] | None = None,
        recent_metrics: list[RecentMetric] | None = None,
    ) -> None:
        self.metric_series = metric_series or []
        self.recent_metrics = recent_metrics or []
        self.history: list[RLStepMetrics] = []
        self.recent_metric_history: list[dict[str, float | None]] = []
        self.metric_series_history: dict[str, list[tuple[int, float]]] = {
            series.label: [] for series in self.metric_series
        }
        self.validation_history: list[ValidationMetrics] = []
        self.latest_completions: list[CompletionRecord] = []
        self.reward_y_limits = reward_y_limits
        self.console = Console()
        self.live = Live(self._render(), console=self.console, refresh_per_second=4, transient=False)
        self.started = False
        self.closed = False

    def on_step_end(self, metrics: RLStepMetrics) -> None:
        self._start()
        self.history.append(metrics)
        self.recent_metric_history.append(self._capture_recent_metrics(metrics))
        self.history = self.history[-MAX_HISTORY:]
        self.recent_metric_history = self.recent_metric_history[-MAX_HISTORY:]
        self._capture_metric_series(metrics)
        self.live.update(self._render(), refresh=True)

    def on_validation_end(self, metrics: ValidationMetrics) -> None:
        self._start()
        self._append_validation(metrics)
        self.live.update(self._render(), refresh=True)

    def _append_validation(self, metrics: ValidationMetrics) -> None:
        self.validation_history = [
            item
            for item in self.validation_history
            if not (item.step == metrics.step and item.name == metrics.name)
        ]
        self.validation_history.append(metrics)
        self.validation_history = self.validation_history[-MAX_HISTORY:]

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
            *self._metric_series_panels(),
            Panel(
                self._history_table(),
                title=f"Recent ({min(len(self.history), RECENT_STEPS)})",
                padding=(0, 1),
            ),
            Panel(self._completion_table(), title="Latest completions", padding=(0, 1)),
        )

    def _metric_series_panels(self) -> list[Panel]:
        panels: list[Panel] = []
        for series in self.metric_series:
            if self.metric_series_history.get(series.label):
                panels.append(
                    Panel(
                        self._metric_series_plot(series),
                        title=f"{series.label} over steps",
                        padding=(0, 1),
                    )
                )
        return panels

    def _summary_line(self) -> str:
        latest = self.history[-1] if self.history else None
        if latest is None:
            return "step=-  reward=-  len=-  loss=-"

        return (
            f"step={latest.generic.step}  reward={latest.reward.mean:.3f}±{latest.reward.std:.3f}  "
            f"len={latest.generic.completion_length_mean:.1f} active={latest.generic.active_completion_length_mean:.1f}  "
            f"loss_seq={latest.generic.loss_sequence_fraction:.2f}  loss={latest.generic.loss:.4f}  "
            f"lr={latest.generic.learning_rate:.2e} raw_grad={latest.generic.grad_norm:.3f} "
            f"clip={latest.generic.grad_clip_scale:.2f}"
            f"{self._grad_norm_stats_summary(latest)}"
            f"{self._validation_summary(latest.generic.step)}"
            f"{self._reward_group_summary(latest)}"
            f"{self._grpo_summary(latest)}{self._timing_summary(latest)}{self._sync_summary(latest)}"
        )

    def _grad_norm_stats_summary(self, metrics: RLStepMetrics) -> str:
        grad_norms = metrics.generic.grad_norms
        if not grad_norms:
            return ""
        sorted_norms = sorted(grad_norms)
        count = len(sorted_norms)
        midpoint = count // 2
        if count % 2 == 0:
            median = (sorted_norms[midpoint - 1] + sorted_norms[midpoint]) / 2.0
        else:
            median = sorted_norms[midpoint]
        return f" grad_min/med/max={sorted_norms[0]:.2f}/{median:.2f}/{sorted_norms[-1]:.2f}"

    def _validation_summary(self, step: int) -> str:
        if not self.validation_history:
            return ""
        latest = self.validation_history[-1]
        if latest.step != step:
            return ""
        return f" val={latest.reward_mean:.3f}"

    def _reward_group_summary(self, metrics: RLStepMetrics) -> str:
        if metrics.reward_groups is None:
            return ""
        return f" groups={metrics.reward_groups.kept}/{metrics.reward_groups.total}"

    def _grpo_summary(self, metrics: RLStepMetrics) -> str:
        if metrics.generic.grpo_clip_ratio is None:
            return ""
        return f" grpo_clip={metrics.generic.grpo_clip_ratio:.3f}"

    def _timing_summary(self, metrics: RLStepMetrics) -> str:
        timings = metrics.generic.timings
        if timings is None:
            return ""

        return (
            f"  t=roll:{timings.rollout_seconds:.2f}s "
            f"{self._old_logprobs_timing(timings)}"
            f"back:{timings.backward_seconds:.2f}s opt:{timings.optimizer_seconds:.2f}s"
        )

    def _old_logprobs_timing(self, timings: StepTimings) -> str:
        if timings.old_logprobs_seconds <= 0.0:
            return ""
        return f"oldlp:{timings.old_logprobs_seconds:.2f}s "

    def _sync_summary(self, metrics: RLStepMetrics) -> str:
        stats = metrics.generic.rollout_sync_stats
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
        for metric in self.recent_metrics:
            table.add_column(metric.label, justify="right")
        table.add_column("Roll", justify="right")
        table.add_column("Back", justify="right")
        recent_metrics = self.history[-RECENT_STEPS:]
        recent_values = self.recent_metric_history[-RECENT_STEPS:]
        if len(recent_values) < len(recent_metrics):
            recent_values = [{} for _ in recent_metrics]
        for metrics, values in zip(recent_metrics, recent_values, strict=True):
            timings = metrics.generic.timings
            row = [
                str(metrics.generic.step),
                f"{metrics.reward.mean:.3f}",
                f"{metrics.reward.std:.3f}",
                f"{metrics.generic.completion_length_mean:.1f}",
                f"{metrics.generic.active_completion_length_mean:.1f}",
                f"{metrics.generic.loss:.4f}",
                f"{metrics.generic.grad_norm:.3f}",
                f"{metrics.generic.grad_clip_scale:.2f}",
            ]
            row.extend(self._recent_metric_value(metric, values) for metric in self.recent_metrics)
            row.extend(
                [
                    "-" if timings is None else f"{timings.rollout_seconds:.2f}",
                    "-" if timings is None else f"{timings.backward_seconds:.2f}",
                ]
            )
            table.add_row(*row)
        return table

    def _capture_recent_metrics(self, metrics: RLStepMetrics) -> dict[str, float | None]:
        return {metric.label: metric.value_fn(metrics) for metric in self.recent_metrics}

    def _capture_metric_series(self, metrics: RLStepMetrics) -> None:
        for series in self.metric_series:
            value = series.value_fn(metrics)
            if value is None:
                continue
            history = self.metric_series_history.setdefault(series.label, [])
            history.append((metrics.generic.step, value))
            self.metric_series_history[series.label] = history[-MAX_HISTORY:]

    def _recent_metric_value(self, metric: RecentMetric, values: dict[str, float | None]) -> str:
        return self._optional_metric(values.get(metric.label), precision=metric.precision, signed=metric.signed)

    def _optional_metric(self, value: float | None, *, precision: int, signed: bool) -> str:
        if value is None:
            return "-"
        sign = "+" if signed else ""
        return f"{value:{sign}.{precision}f}"

    def _reward_plot(self) -> RenderableType:
        if not self.history and not self.validation_history:
            return "No reward history yet."

        steps = [metrics.generic.step for metrics in self.history]
        rewards = [metrics.reward.mean for metrics in self.history]
        validation_steps = [metrics.step for metrics in self.validation_history]
        validation_rewards = [metrics.reward_mean for metrics in self.validation_history]
        x_min, x_max = self._plot_x_limits(steps + validation_steps)
        max_cells = self._max_plot_cells()
        raw_plot = self._build_reward_plot(
            width=self._initial_plot_width(),
            steps=steps,
            rewards=rewards,
            validation_steps=validation_steps,
            validation_rewards=validation_rewards,
            x_min=x_min,
            x_max=x_max,
            max_cells=max_cells,
        )
        return Text.from_ansi(raw_plot, no_wrap=True, overflow="crop")

    def _metric_series_plot(self, series: MetricSeries) -> RenderableType:
        values = self.metric_series_history.get(series.label, [])
        if not values:
            return f"No {series.label.lower()} history yet."

        steps = [step for step, _value in values]
        series_values = [value for _step, value in values]
        x_min, x_max = self._plot_x_limits(steps)
        raw_plot = self._build_reward_plot(
            width=self._initial_plot_width(),
            steps=steps,
            rewards=series_values,
            validation_steps=[],
            validation_rewards=[],
            x_min=x_min,
            x_max=x_max,
            max_cells=self._max_plot_cells(),
            y_label=series.y_label,
            y_limits=series.y_limits,
        )
        return Text.from_ansi(raw_plot, no_wrap=True, overflow="crop")

    def _max_plot_cells(self) -> int:
        return max(PLOT_MIN_WIDTH, self.console.size.width - PLOT_PANEL_OVERHEAD)

    def _initial_plot_width(self) -> int:
        return min(PLOT_MAX_WIDTH, self._max_plot_cells())

    def _build_reward_plot(
        self,
        *,
        width: int,
        steps: list[int],
        rewards: list[float],
        validation_steps: list[int],
        validation_rewards: list[float],
        x_min: int,
        x_max: int,
        max_cells: int,
        y_label: str = "reward",
        y_limits: tuple[float, float] | None = None,
    ) -> str:
        width = max(PLOT_MIN_WIDTH, width)
        while True:
            raw_plot = self._render_reward_plot(
                width=width,
                steps=steps,
                rewards=rewards,
                validation_steps=validation_steps,
                validation_rewards=validation_rewards,
                x_min=x_min,
                x_max=x_max,
                y_label=y_label,
                y_limits=y_limits,
            )
            longest_line = max(Text.from_ansi(line).cell_len for line in raw_plot.splitlines())
            if longest_line <= max_cells or width <= PLOT_MIN_WIDTH:
                return raw_plot
            width = max(PLOT_MIN_WIDTH, width - max(1, longest_line - max_cells))

    def _render_reward_plot(
        self,
        *,
        width: int,
        steps: list[int],
        rewards: list[float],
        validation_steps: list[int],
        validation_rewards: list[float],
        x_min: int,
        x_max: int,
        y_label: str,
        y_limits: tuple[float, float] | None,
    ) -> str:
        figure = plotille.Figure()
        figure.width = width
        figure.height = PLOT_HEIGHT
        figure.x_label = "step"
        figure.y_label = y_label
        figure.origin = False
        figure.x_ticks_fkt = self._step_tick
        figure.y_ticks_fkt = self._reward_tick
        figure.set_x_limits(min_=x_min, max_=x_max)
        y_min, y_max = y_limits if y_limits is not None else self._plot_y_limits(rewards + validation_rewards)
        figure.set_y_limits(min_=y_min, max_=y_max)

        if steps:
            figure.plot(steps, rewards, label="train")
        if validation_steps:
            figure.scatter(validation_steps, validation_rewards, label="val", marker="x")
        return figure.show(legend=bool(validation_steps))

    def _plot_x_limits(self, steps: list[int]) -> tuple[int, int]:
        first = min(steps)
        last = max(steps)
        if first == last:
            return first, first + 1
        return first, last

    def _plot_y_limits(self, rewards: list[float]) -> tuple[float, float]:
        if self.reward_y_limits is not None:
            return self.reward_y_limits

        minimum = min(rewards)
        maximum = max(rewards)
        if minimum == maximum:
            return minimum - 1.0, maximum + 1.0

        padding = max(1.0, (maximum - minimum) * 0.05)
        return minimum - padding, maximum + padding

    def _step_tick(self, value: int | float | datetime, next_value: int | float | datetime) -> str:
        if isinstance(value, datetime) or isinstance(next_value, datetime):
            return str(value)
        step = round(value)
        tick_width = abs(next_value - value)
        if abs(value - step) <= max(0.05, tick_width / 2):
            return str(int(step))
        return ""

    def _reward_tick(self, value: int | float | datetime, _next_value: int | float | datetime) -> str:
        if isinstance(value, datetime):
            return str(value)
        return f"{value:.2f}"

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


class JSONLLogCallback:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = (log_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self.completions_file = (log_dir / "completions.jsonl").open("a", encoding="utf-8")
        self.closed = False

    def on_step_end(self, metrics: RLStepMetrics) -> None:
        self.metrics_file.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")
        self.metrics_file.flush()

    def on_validation_end(self, metrics: ValidationMetrics) -> None:
        self.metrics_file.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")
        self.metrics_file.flush()

    def on_completions(self, records: list[CompletionRecord]) -> None:
        for record in records:
            self.completions_file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        self.completions_file.flush()

    def close(self) -> None:
        if self.closed:
            return
        self.metrics_file.close()
        self.completions_file.close()
        self.closed = True
