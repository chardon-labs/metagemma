from collections.abc import Iterable

import bitsandbytes as bnb
import torch

from rl_trainer.config import AdamW8BitOptimizerConfig, AdamWOptimizerConfig, MuonOptimizerConfig, OptimizerConfig
from rl_trainer.types import OptimizerLike


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def trainable_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


class CombinedOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("At least one optimizer is required.")
        self.optimizers = optimizers
        self.param_groups = [
            group
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)


class LinearScheduler:
    def __init__(self, optimizer: "OptimizerLike", *, warmup_ratio: float, max_steps: int) -> None:
        self.optimizer = optimizer
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.max_steps = max_steps
        self.step_count = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self._apply_lrs()

    def step(self) -> None:
        self.step_count += 1
        self._apply_lrs()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _apply_lrs(self) -> None:
        factor = self._lr_factor(self.step_count)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def _lr_factor(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        remaining = max(1, self.max_steps - self.warmup_steps)
        return max(0.0, float(self.max_steps - step) / float(remaining))


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> OptimizerLike:
    named_parameters = trainable_named_parameters(model)
    parameters = [parameter for _, parameter in named_parameters]
    if isinstance(config, AdamW8BitOptimizerConfig):
        return bnb.optim.AdamW8bit(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.epsilon,
        )

    if isinstance(config, AdamWOptimizerConfig):
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.epsilon,
        )

    return build_muon(named_parameters, config)


def build_muon(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    config: MuonOptimizerConfig,
) -> OptimizerLike:
    muon_parameters = [
        parameter
        for name, parameter in named_parameters
        if _should_use_muon(name, parameter)
    ]
    adamw_parameters = [
        parameter
        for name, parameter in named_parameters
        if not _should_use_muon(name, parameter)
    ]
    optimizers: list[torch.optim.Optimizer] = []

    if muon_parameters:
        optimizers.append(
            torch.optim.Muon(
                muon_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
                eps=config.epsilon,
                adjust_lr_fn=config.adjust_lr_fn,
            )
        )
    if adamw_parameters:
        optimizers.append(
            torch.optim.AdamW(
                adamw_parameters,
                lr=config.adamw_learning_rate if config.adamw_learning_rate is not None else config.learning_rate,
                weight_decay=config.adamw_weight_decay,
                eps=config.adamw_epsilon,
            )
        )

    return CombinedOptimizer(optimizers)


def _should_use_muon(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim != 2:
        return False

    excluded_markers = (
        "embed",
        "embedding",
        "lm_head",
        "output",
        "score",
    )
    return not any(marker in name for marker in excluded_markers)


def build_linear_scheduler(
    optimizer: OptimizerLike,
    *,
    warmup_ratio: float,
    max_steps: int,
) -> LinearScheduler:
    return LinearScheduler(optimizer, warmup_ratio=warmup_ratio, max_steps=max_steps)


def optimizer_from_parameters(
    parameters: Iterable[torch.nn.Parameter],
    config: OptimizerConfig,
) -> OptimizerLike:
    parameter_list = [parameter for parameter in parameters if parameter.requires_grad]
    if isinstance(config, AdamW8BitOptimizerConfig):
        return bnb.optim.AdamW8bit(
            parameter_list,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.epsilon,
        )

    if isinstance(config, AdamWOptimizerConfig):
        return torch.optim.AdamW(
            parameter_list,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.epsilon,
        )

    return build_muon([(str(index), parameter) for index, parameter in enumerate(parameter_list)], config)
