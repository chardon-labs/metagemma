import torch

from rl_trainer.types import GRPOLossInput, GRPOLossOutput, LossInput, LossOutput


def policy_gradient_loss(inputs: LossInput) -> LossOutput:
    advantages = inputs.advantages.unsqueeze(1)
    per_token_loss = -inputs.current_logprobs * advantages
    mask = inputs.completion_mask
    normalizer = inputs.normalizer if inputs.normalizer is not None else mask.sum().clamp(min=1.0)
    loss = (per_token_loss * mask).sum() / normalizer
    return LossOutput(loss=loss)


def grpo_loss(inputs: GRPOLossInput) -> GRPOLossOutput:
    advantages = inputs.advantages.unsqueeze(1)
    log_ratio = inputs.current_logprobs - inputs.old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - inputs.epsilon, 1.0 + inputs.epsilon_high)

    per_token_loss_1 = ratio * advantages
    per_token_loss_2 = clipped_ratio * advantages
    per_token_loss = -torch.minimum(per_token_loss_1, per_token_loss_2)

    mask = inputs.completion_mask
    normalizer = inputs.normalizer if inputs.normalizer is not None else mask.sum().clamp(min=1.0)
    loss = (per_token_loss * mask).sum() / normalizer

    low_clipped = (ratio < 1.0 - inputs.epsilon) & (advantages < 0.0)
    high_clipped = (ratio > 1.0 + inputs.epsilon_high) & (advantages > 0.0)
    clipped = (low_clipped | high_clipped).to(mask.dtype)
    clip_ratio = (clipped * mask).sum() / mask.sum().clamp(min=1.0)
    return GRPOLossOutput(loss=loss, clip_ratio=clip_ratio)
