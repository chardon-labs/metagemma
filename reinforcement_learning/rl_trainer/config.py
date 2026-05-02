from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class GRPOAlgorithmConfig:
    name: Literal["grpo"] = "grpo"


@dataclass(frozen=True)
class TPOAlgorithmConfig:
    eta: float = 1.0
    optimization_epochs: int = 4
    name: Literal["tpo"] = "tpo"


AlgorithmConfig: TypeAlias = GRPOAlgorithmConfig | TPOAlgorithmConfig


@dataclass(frozen=True)
class AdamWOptimizerConfig:
    learning_rate: float
    weight_decay: float
    epsilon: float = 1e-8
    name: Literal["adamw"] = "adamw"


@dataclass(frozen=True)
class AdamW8BitOptimizerConfig:
    learning_rate: float
    weight_decay: float
    epsilon: float = 1e-8
    name: Literal["adamw_8bit"] = "adamw_8bit"


@dataclass(frozen=True)
class MuonOptimizerConfig:
    learning_rate: float
    weight_decay: float = 0.0
    momentum: float = 0.95
    nesterov: bool = True
    epsilon: float = 1e-7
    adamw_learning_rate: float | None = None
    adamw_weight_decay: float = 0.0
    adamw_epsilon: float = 1e-8
    name: Literal["muon"] = "muon"


OptimizerConfig: TypeAlias = AdamWOptimizerConfig | AdamW8BitOptimizerConfig | MuonOptimizerConfig


@dataclass(frozen=True)
class RLTrainerConfig:
    warmup_ratio: float
    batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_completion_length: int
    max_steps: int
    logging_steps: int
    save_steps: int
    output_dir: Path
    optimizer: OptimizerConfig
    algorithm: AlgorithmConfig = field(default_factory=GRPOAlgorithmConfig)
    backward_microbatch_size: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    mask_truncated_completions: bool = True
    max_grad_norm: float = 1.0
    seed: int = 3407
    shuffle: bool = True
    use_generation_cache: bool = True
    disable_generation_compile: bool = True
    empty_cache_steps: int | None = 1
    chat_template_kwargs: dict[str, bool] = field(default_factory=dict)

    @property
    def learning_rate(self) -> float:
        return self.optimizer.learning_rate

    @property
    def weight_decay(self) -> float:
        return self.optimizer.weight_decay
