from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch

from rl_trainer.config import LossType

Message: TypeAlias = dict[str, str]
Completion: TypeAlias = list[Message]
RewardFunction: TypeAlias = Callable[..., Awaitable[list[float | None]]]


@dataclass(frozen=True)
class TrainingExample:
    prompt: list[dict[str, object]]
    fields: dict[str, object]


@dataclass(frozen=True)
class PromptBatch:
    examples: list[TrainingExample]
    prompts: list[list[dict[str, object]]]


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_texts: list[str]


@dataclass(frozen=True)
class RolloutBatch:
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_logprobs: torch.Tensor
    completions: list[Completion]


@dataclass(frozen=True)
class RewardBatch:
    prompts: list[list[dict[str, object]]]
    completions: list[Completion]
    completion_ids: list[list[int]]
    extra_fields: dict[str, list[object]]
    trainer_state: "TrainerState"


@dataclass(frozen=True)
class RewardResult:
    per_function: torch.Tensor
    total: torch.Tensor
    names: list[str]


@dataclass(frozen=True)
class AdvantageBatch:
    rewards: torch.Tensor
    advantages: torch.Tensor


@dataclass(frozen=True)
class LossInput:
    current_logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    completion_mask: torch.Tensor
    epsilon: float
    epsilon_high: float
    delta: float | None
    loss_type: LossType


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    mean_ratio: float
    clip_ratio: float


@dataclass(frozen=True)
class StepMetrics:
    step: int
    loss: float
    reward_mean: float
    reward_std: float
    completion_length_mean: float
    active_completion_length_mean: float
    loss_sequence_fraction: float
    learning_rate: float
    grad_norm: float
    mean_ratio: float
    clip_ratio: float
    reward_function_means: dict[str, float]


@dataclass(frozen=True)
class CompletionRecord:
    prompt: str
    completion: str
    reward: float
    advantages: float


@dataclass
class TrainerState:
    step: int = 0
    examples_seen: int = 0


class DatasetLike(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, object]: ...


class RolloutEngine(Protocol):
    def generate(self, batch: PromptBatch) -> RolloutBatch: ...


class OptimizerFactory(Protocol):
    def __call__(self, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer: ...


class SchedulerFactory(Protocol):
    def __call__(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler: ...


class Environment(Protocol):
    async def reset(self, example: TrainingExample) -> str | None: ...


EnvironmentFactory: TypeAlias = Callable[[], Environment]
