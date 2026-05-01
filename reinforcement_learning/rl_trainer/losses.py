import torch

from rl_trainer.types import LossInput, LossOutput


def policy_gradient_loss(inputs: LossInput) -> LossOutput:
    advantages = inputs.advantages.unsqueeze(1)
    per_token_loss = -inputs.current_logprobs * advantages
    mask = inputs.completion_mask
    normalizer = inputs.normalizer if inputs.normalizer is not None else mask.sum().clamp(min=1.0)
    loss = (per_token_loss * mask).sum() / normalizer
    return LossOutput(loss=loss)
