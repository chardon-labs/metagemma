from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterator, cast

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from confidence_serving.model_loader import LoadedConfidenceModel
from confidence_serving.settings import (
    CONFIDENCE_TOKEN,
    CONFIDENCE_TOKEN_ID,
    ENABLE_THINKING,
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class GenerateRequest:
    messages: Sequence[ChatMessage]
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    repetition_penalty: float = REPETITION_PENALTY
    enable_thinking: bool = ENABLE_THINKING


@dataclass(frozen=True)
class ConfidenceSummary:
    final: float | None
    mean: float | None
    tail10_mean: float | None


@dataclass(frozen=True)
class GenerateResult:
    completion: str
    token_ids: list[int]
    token_confidences: list[float]
    confidence: float | None
    confidence_summary: ConfidenceSummary
    finish_reason: str


@dataclass(frozen=True)
class StreamTokenEvent:
    token_id: int
    text: str
    confidence: float


@dataclass(frozen=True)
class StreamFinalEvent:
    completion: str
    token_ids: list[int]
    token_confidences: list[float]
    confidence: float | None
    confidence_summary: ConfidenceSummary
    finish_reason: str


StreamEvent = StreamTokenEvent | StreamFinalEvent


def _message_dicts(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def render_prompt(tokenizer: PreTrainedTokenizerBase, request: GenerateRequest) -> str:
    rendered = tokenizer.apply_chat_template(
        _message_dicts(request.messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=request.enable_thinking,
    )
    if not isinstance(rendered, str):
        raise TypeError("Expected tokenizer chat template to render a string.")
    return rendered


def _summary(confidences: Sequence[float]) -> ConfidenceSummary:
    if not confidences:
        return ConfidenceSummary(final=None, mean=None, tail10_mean=None)
    tail = confidences[-10:]
    return ConfidenceSummary(
        final=confidences[-1],
        mean=sum(confidences) / len(confidences),
        tail10_mean=sum(tail) / len(tail),
    )


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: Sequence[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    adjusted = logits.clone()
    for token_id in set(token_ids):
        token_logit = adjusted[:, token_id]
        adjusted[:, token_id] = torch.where(token_logit < 0, token_logit * penalty, token_logit / penalty)
    return adjusted


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be greater than 0.")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits.float(), dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    filtered = logits.clone()
    remove_indices = sorted_indices[sorted_remove]
    filtered[:, remove_indices] = -torch.inf
    return filtered


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")

    scaled = logits / temperature
    filtered = _top_p_filter(scaled, top_p)
    probs = F.softmax(filtered.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate_confidence_stream(loaded: LoadedConfidenceModel, request: GenerateRequest) -> Iterator[StreamEvent]:
    if request.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    tokenizer = loaded.tokenizer
    prompt = render_prompt(tokenizer, request)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = cast(torch.Tensor, encoded["input_ids"]).to(loaded.device)
    attention_mask = cast(torch.Tensor, encoded["attention_mask"]).to(loaded.device)

    generated_ids: list[int] = []
    token_confidences: list[float] = []
    finish_reason = "length"
    eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        for _ in range(request.max_new_tokens):
            outputs = loaded.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = cast(torch.Tensor, outputs.logits[:, -1, :])
            confidence = torch.sigmoid(logits[:, CONFIDENCE_TOKEN_ID].float())
            token_confidences.append(float(confidence.item()))

            sampling_logits = logits.clone()
            sampling_logits[:, CONFIDENCE_TOKEN_ID] = -torch.inf
            sampling_logits = _apply_repetition_penalty(
                sampling_logits,
                [*input_ids[0].tolist(), *generated_ids],
                request.repetition_penalty,
            )
            next_token = _sample_next_token(sampling_logits, request.temperature, request.top_p)
            next_token_id = int(next_token.item())
            if eos_token_id is not None and next_token_id == eos_token_id:
                finish_reason = "stop"
                break

            generated_ids.append(next_token_id)
            token_text = cast(str, tokenizer.decode([next_token_id], skip_special_tokens=True))
            token_text = token_text.replace(CONFIDENCE_TOKEN, "")
            yield StreamTokenEvent(
                token_id=next_token_id,
                text=token_text,
                confidence=float(confidence.item()),
            )
            next_token_tensor = next_token.reshape(1, 1).to(device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
            attention_mask = torch.ones_like(input_ids)

    text = cast(str, tokenizer.decode(generated_ids, skip_special_tokens=True))
    text = text.replace(CONFIDENCE_TOKEN, "")
    summary = _summary(token_confidences[: len(generated_ids)])
    yield StreamFinalEvent(
        completion=text,
        token_ids=generated_ids,
        token_confidences=token_confidences[: len(generated_ids)],
        confidence=summary.final,
        confidence_summary=summary,
        finish_reason=finish_reason,
    )


def generate_with_confidence(loaded: LoadedConfidenceModel, request: GenerateRequest) -> GenerateResult:
    final_event: StreamFinalEvent | None = None
    for event in generate_confidence_stream(loaded, request):
        if isinstance(event, StreamFinalEvent):
            final_event = event
    if final_event is None:
        raise RuntimeError("Generation stream ended without a final event.")
    return GenerateResult(
        completion=final_event.completion,
        token_ids=final_event.token_ids,
        token_confidences=final_event.token_confidences,
        confidence=final_event.confidence,
        confidence_summary=final_event.confidence_summary,
        finish_reason=final_event.finish_reason,
    )
