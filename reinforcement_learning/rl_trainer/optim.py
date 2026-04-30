from collections.abc import Iterable

import bitsandbytes as bnb
import torch

from rl_trainer.config import RLTrainerConfig


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_adamw(model: torch.nn.Module, config: RLTrainerConfig) -> torch.optim.Optimizer:
    parameters = trainable_parameters(model)
    if config.optimizer == "adamw_8bit":
        return bnb.optim.AdamW8bit(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )

    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )


def build_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_ratio: float,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(max_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        remaining = max(1, max_steps - warmup_steps)
        return max(0.0, float(max_steps - step) / float(remaining))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def optimizer_from_parameters(
    parameters: Iterable[torch.nn.Parameter],
    config: RLTrainerConfig,
) -> torch.optim.Optimizer:
    if config.optimizer == "adamw_8bit":
        return bnb.optim.AdamW8bit(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )

    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )
