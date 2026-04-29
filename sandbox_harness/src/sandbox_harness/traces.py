from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLogprob:
    token_id: int
    logprob: float


@dataclass(frozen=True)
class AssistantTurnTrace:
    message_index: int
    prompt_text: str
    prompt_token_ids: list[int]
    completion_text: str
    completion_token_ids: list[int]
    top_logprobs: list[list[TokenLogprob]]
    finish_reason: str | None
