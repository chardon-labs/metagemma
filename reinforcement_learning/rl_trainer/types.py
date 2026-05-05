from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import torch

Message: TypeAlias = dict[str, str]
Completion: TypeAlias = list[Message]


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
class RolloutSyncStats:
    step: int
    synced_tensors: int
    loaded_tensors: int
    synced_bytes: int


@dataclass(frozen=True)
class RolloutBatch:
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    completions: list[Completion]


@dataclass(frozen=True)
class RewardBatch:
    prompts: list[list[dict[str, object]]]
    completions: list[Completion]
    completion_ids: list[list[int]]
    completion_mask: list[list[float]]
    extra_fields: dict[str, list[object]]
    trainer_state: "TrainerState"


RewardFunction: TypeAlias = Callable[[RewardBatch], Awaitable[list[float | None]]]


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
    advantages: torch.Tensor
    completion_mask: torch.Tensor
    normalizer: torch.Tensor | None = None


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor


@dataclass(frozen=True)
class GRPOLossInput:
    current_logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    completion_mask: torch.Tensor
    epsilon: float
    epsilon_high: float
    normalizer: torch.Tensor | None = None


@dataclass(frozen=True)
class GRPOLossOutput:
    loss: torch.Tensor
    clip_ratio: torch.Tensor


@dataclass(frozen=True)
class StepTimings:
    rollout_seconds: float
    reward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    microbatch_seconds: float
    old_logprobs_seconds: float = 0.0


@dataclass(frozen=True)
class GenericStepMetrics:
    step: int
    loss: float
    completion_length_mean: float
    active_completion_length_mean: float
    loss_sequence_fraction: float
    learning_rate: float
    grad_norm: float
    grad_clip_scale: float
    rollout_sync_stats: RolloutSyncStats | None = None
    timings: StepTimings | None = None
    grpo_clip_ratio: float | None = None
    grad_norms: list[float] | None = None


@dataclass(frozen=True)
class RewardStats:
    mean: float
    std: float
    function_means: dict[str, float]


@dataclass(frozen=True)
class RewardGroupStats:
    kept: int
    total: int


@dataclass(frozen=True)
class RLStepMetrics:
    generic: GenericStepMetrics
    reward: RewardStats
    reward_groups: RewardGroupStats | None = None
    unfiltered_reward: RewardStats | None = None


@dataclass(frozen=True)
class ValidationMetrics:
    step: int
    reward_mean: float
    name: str = "validation"


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


class OptimizerLike(Protocol):
    param_groups: list[dict[str, Any]]

    def step(self) -> None: ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...


class SchedulerLike(Protocol):
    def step(self) -> None: ...

    def get_last_lr(self) -> list[float]: ...


class OptimizerFactory(Protocol):
    def __call__(self, parameters: Iterable[torch.nn.Parameter]) -> OptimizerLike: ...


class SchedulerFactory(Protocol):
    def __call__(self, optimizer: OptimizerLike) -> SchedulerLike: ...
