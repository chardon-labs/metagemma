from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from inference_server.model_loader import LoadedConfidenceModel
from inference_server.settings import (
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
    n: int = 1


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
class GenerateBatchResult:
    completions: list[GenerateResult]


@dataclass(frozen=True)
class StreamTokenEvent:
    index: int
    token_id: int
    text: str
    confidence: float


@dataclass(frozen=True)
class StreamFinalEvent:
    index: int
    completion: str
    token_ids: list[int]
    token_confidences: list[float]
    confidence: float | None
    confidence_summary: ConfidenceSummary
    finish_reason: str


@dataclass(frozen=True)
class StreamBatchFinalEvent:
    completions: list[GenerateResult]


StreamEvent = StreamTokenEvent | StreamFinalEvent | StreamBatchFinalEvent
STOP_TOKEN_NAMES = (
    "eos_token",
    "eot_token",
    "etr_token",
    "eoc_token",
)


class _StreamEnded:
    pass


_STREAM_ENDED = _StreamEnded()


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

    remove_mask = torch.zeros_like(sorted_remove).scatter(dim=-1, index=sorted_indices, src=sorted_remove)
    return logits.masked_fill(remove_mask, -torch.inf)


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")

    scaled = logits / temperature
    filtered = _top_p_filter(scaled, top_p)
    probs = F.softmax(filtered.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    token_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        token_ids.add(int(tokenizer.eos_token_id))

    for token_name in STOP_TOKEN_NAMES:
        token = getattr(tokenizer, token_name, None)
        if not isinstance(token, str):
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.add(token_id)

    return token_ids


def _finish_result(
    *,
    tokenizer: PreTrainedTokenizerBase,
    index: int,
    generated_ids: list[int],
    token_confidences: list[float],
    finish_reason: str,
) -> StreamFinalEvent:
    text = cast(str, tokenizer.decode(generated_ids, skip_special_tokens=True))
    text = text.replace(CONFIDENCE_TOKEN, "")
    summary = _summary(token_confidences)
    return StreamFinalEvent(
        index=index,
        completion=text,
        token_ids=generated_ids,
        token_confidences=token_confidences,
        confidence=summary.final,
        confidence_summary=summary,
        finish_reason=finish_reason,
    )


def generate_confidence_stream(loaded: LoadedConfidenceModel, request: GenerateRequest) -> Iterator[StreamEvent]:
    if request.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if request.n <= 0:
        raise ValueError("n must be positive.")

    tokenizer = loaded.tokenizer
    prompt = render_prompt(tokenizer, request)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = cast(torch.Tensor, encoded["input_ids"]).to(loaded.device).repeat(request.n, 1)
    attention_mask = cast(torch.Tensor, encoded["attention_mask"]).to(loaded.device).repeat(request.n, 1)

    generated_ids = [[] for _ in range(request.n)]
    token_confidences = [[] for _ in range(request.n)]
    repetition_token_ids = [input_ids[index].tolist() for index in range(request.n)]
    finished = [False for _ in range(request.n)]
    finish_reasons = ["length" for _ in range(request.n)]
    stop_ids = stop_token_ids(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = next(iter(stop_ids), 0)

    with torch.inference_mode():
        model_input_ids = input_ids
        past_key_values = None
        for _ in range(request.max_new_tokens):
            if all(finished):
                break

            outputs = loaded.model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = cast(torch.Tensor, outputs.logits[:, -1, :])
            confidence = torch.sigmoid(logits[:, CONFIDENCE_TOKEN_ID].float())

            sampling_logits = logits.clone()
            sampling_logits[:, CONFIDENCE_TOKEN_ID] = -torch.inf
            for index in range(request.n):
                if finished[index]:
                    continue
                sampling_logits[index : index + 1] = _apply_repetition_penalty(
                    sampling_logits[index : index + 1],
                    repetition_token_ids[index],
                    request.repetition_penalty,
                )

            next_token = _sample_next_token(sampling_logits, request.temperature, request.top_p)
            confidence_values = [float(value) for value in confidence.tolist()]
            next_token_ids = [int(token_id) for token_id in next_token.tolist()]
            next_column = torch.full(
                (request.n, 1),
                pad_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            next_attention = torch.zeros((request.n, 1), device=attention_mask.device, dtype=attention_mask.dtype)

            for index in range(request.n):
                if finished[index]:
                    continue

                confidence_value = confidence_values[index]
                next_token_id = next_token_ids[index]
                if next_token_id in stop_ids:
                    finished[index] = True
                    finish_reasons[index] = "stop"
                    continue

                generated_ids[index].append(next_token_id)
                token_confidences[index].append(confidence_value)
                repetition_token_ids[index].append(next_token_id)
                next_column[index, 0] = next_token_id
                next_attention[index, 0] = 1

                token_text = cast(str, tokenizer.decode([next_token_id], skip_special_tokens=True))
                token_text = token_text.replace(CONFIDENCE_TOKEN, "")
                yield StreamTokenEvent(
                    index=index,
                    token_id=next_token_id,
                    text=token_text,
                    confidence=confidence_value,
                )

            input_ids = torch.cat([input_ids, next_column], dim=-1)
            attention_mask = torch.cat([attention_mask, next_attention], dim=-1)
            model_input_ids = next_column

    final_events = [
        _finish_result(
            tokenizer=tokenizer,
            index=index,
            generated_ids=generated_ids[index],
            token_confidences=token_confidences[index],
            finish_reason=finish_reasons[index],
        )
        for index in range(request.n)
    ]
    for event in final_events:
        yield event
    yield StreamBatchFinalEvent(
        completions=[
            GenerateResult(
                completion=event.completion,
                token_ids=event.token_ids,
                token_confidences=event.token_confidences,
                confidence=event.confidence,
                confidence_summary=event.confidence_summary,
                finish_reason=event.finish_reason,
            )
            for event in final_events
        ]
    )


def _next_stream_event(iterator: Iterator[StreamEvent]) -> StreamEvent | _StreamEnded:
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_ENDED


async def async_generate_confidence_stream(
    loaded: LoadedConfidenceModel,
    request: GenerateRequest,
) -> AsyncIterator[StreamEvent]:
    iterator = generate_confidence_stream(loaded, request)
    while True:
        event = await asyncio.to_thread(_next_stream_event, iterator)
        if isinstance(event, _StreamEnded):
            break
        yield event


def generate_with_confidence(loaded: LoadedConfidenceModel, request: GenerateRequest) -> GenerateResult:
    final_event: StreamBatchFinalEvent | None = None
    for event in generate_confidence_stream(loaded, request):
        if isinstance(event, StreamBatchFinalEvent):
            final_event = event
    if final_event is None:
        raise RuntimeError("Generation stream ended without a final event.")
    if len(final_event.completions) != 1:
        raise ValueError("generate_with_confidence only supports requests with n=1.")
    return final_event.completions[0]


def generate_batch_with_confidence(loaded: LoadedConfidenceModel, request: GenerateRequest) -> GenerateBatchResult:
    final_event: StreamBatchFinalEvent | None = None
    for event in generate_confidence_stream(loaded, request):
        if isinstance(event, StreamBatchFinalEvent):
            final_event = event
    if final_event is None:
        raise RuntimeError("Generation stream ended without a final event.")
    return GenerateBatchResult(completions=final_event.completions)


async def async_generate_batch_with_confidence(
    loaded: LoadedConfidenceModel,
    request: GenerateRequest,
) -> GenerateBatchResult:
    final_event: StreamBatchFinalEvent | None = None
    async for event in async_generate_confidence_stream(loaded, request):
        if isinstance(event, StreamBatchFinalEvent):
            final_event = event
    if final_event is None:
        raise RuntimeError("Generation stream ended without a final event.")
    return GenerateBatchResult(completions=final_event.completions)


async def async_generate_with_confidence(loaded: LoadedConfidenceModel, request: GenerateRequest) -> GenerateResult:
    result = await async_generate_batch_with_confidence(loaded, request)
    if len(result.completions) != 1:
        raise ValueError("async_generate_with_confidence only supports requests with n=1.")
    return result.completions[0]
