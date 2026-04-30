import torch

from rl_trainer.types import LossInput, LossOutput


def clipped_policy_loss(inputs: LossInput) -> LossOutput:
    ratio = torch.exp(inputs.current_logprobs - inputs.old_logprobs)
    advantages = inputs.advantages.unsqueeze(1)
    clipped_ratio = torch.clamp(ratio, 1.0 - inputs.epsilon, 1.0 + inputs.epsilon_high)
    if inputs.delta is not None:
        ratio = torch.clamp(ratio, max=inputs.delta)

    per_token_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
    mask = inputs.completion_mask
    if inputs.loss_type == "grpo":
        loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).mean()
    elif inputs.loss_type == "bnpo":
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        raise ValueError(f"Unsupported loss type: {inputs.loss_type}")

    clipped = (ratio - clipped_ratio).abs().gt(1e-8).to(torch.float32)
    active_clipped = (clipped * mask).sum() / mask.sum().clamp(min=1.0)
    active_ratio = (ratio * mask).sum() / mask.sum().clamp(min=1.0)
    return LossOutput(
        loss=loss,
        mean_ratio=float(active_ratio.detach().cpu()),
        clip_ratio=float(active_clipped.detach().cpu()),
    )
