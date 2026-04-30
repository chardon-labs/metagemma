from typing import Any

import torch

from rl_trainer.tensors import logprobs_from_logits


def policy_logprobs(
    model: Any,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    prompt_ids = prompt_ids.clone()
    prompt_attention_mask = prompt_attention_mask.clone()
    completion_ids = completion_ids.clone()
    completion_mask = completion_mask.clone()
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, completion_mask.to(prompt_attention_mask.dtype)], dim=1)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, prompt_ids.shape[1] - 1 : -1, :]
    logits = logits / temperature
    return logprobs_from_logits(logits, completion_ids)
