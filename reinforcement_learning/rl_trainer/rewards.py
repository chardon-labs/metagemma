import asyncio

import torch

from rl_trainer.types import RewardBatch, RewardFunction, RewardResult


def reward_name(reward_function: RewardFunction) -> str:
    return getattr(reward_function, "__name__", reward_function.__class__.__name__)


async def score_rewards(reward_functions: list[RewardFunction], batch: RewardBatch, device: torch.device) -> RewardResult:
    async def score_one(reward_function: RewardFunction) -> list[float | None]:
        return await reward_function(
            prompts=batch.prompts,
            completions=batch.completions,
            completion_ids=batch.completion_ids,
            trainer_state=batch.trainer_state,
            **batch.extra_fields,
        )

    raw_scores = await asyncio.gather(*(score_one(function) for function in reward_functions))
    columns = []
    for scores in raw_scores:
        values = [float("nan") if score is None else score for score in scores]
        columns.append(torch.tensor(values, dtype=torch.float32, device=device))

    per_function = torch.stack(columns, dim=1)
    total = torch.nan_to_num(per_function, nan=0.0).sum(dim=1)
    return RewardResult(
        per_function=per_function,
        total=total,
        names=[reward_name(function) for function in reward_functions],
    )
