import torch


def logprobs_from_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def completion_mask(completion_ids: torch.Tensor, eos_token_id: int | None, pad_token_id: int) -> torch.Tensor:
    mask = completion_ids.ne(pad_token_id).to(torch.float32)
    if eos_token_id is None:
        return mask

    eos_positions = completion_ids.eq(eos_token_id)
    token_positions = torch.arange(completion_ids.shape[1], device=completion_ids.device).unsqueeze(0)
    first_eos = torch.where(eos_positions, token_positions, completion_ids.shape[1]).min(dim=1).values
    through_eos = token_positions <= first_eos.unsqueeze(1)
    return mask * through_eos.to(torch.float32)
