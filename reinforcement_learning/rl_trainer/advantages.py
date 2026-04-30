import torch

from rl_trainer.types import AdvantageBatch


def group_relative_advantages(rewards: torch.Tensor, num_generations: int) -> AdvantageBatch:
    grouped = rewards.view(-1, num_generations)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - means) / (stds + 1e-4)
    return AdvantageBatch(rewards=rewards, advantages=advantages.view(-1))
