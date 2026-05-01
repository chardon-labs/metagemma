from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


OptimizerType = Literal["adamw_8bit", "adamw"]


@dataclass(frozen=True)
class RLTrainerConfig:
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_completion_length: int
    max_steps: int
    logging_steps: int
    save_steps: int
    output_dir: Path
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    mask_truncated_completions: bool = True
    max_grad_norm: float = 1.0
    seed: int = 3407
    shuffle: bool = True
    optimizer: OptimizerType = "adamw_8bit"
    adam_epsilon: float = 1e-8
    use_generation_cache: bool = True
    disable_generation_compile: bool = True
    empty_cache_steps: int | None = 1
    chat_template_kwargs: dict[str, bool] = field(default_factory=dict)
