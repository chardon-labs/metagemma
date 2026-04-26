from __future__ import annotations

import asyncio
import logging
import json
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Literal, cast

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from confidence_serving.generate import (
    ChatMessage,
    GenerateBatchResult,
    GenerateRequest,
    GenerateResult,
    StreamBatchFinalEvent,
    StreamFinalEvent,
    StreamTokenEvent,
    async_generate_batch_with_confidence,
    async_generate_confidence_stream,
)
from confidence_serving.model_loader import LoadedConfidenceModel, load_confidence_model
from confidence_serving.settings import HOST, MAX_NEW_TOKENS, PORT, REPETITION_PENALTY, TEMPERATURE, TOP_P


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

MODEL: LoadedConfidenceModel | None = None
MODEL_LOCK = asyncio.Lock()


class ChatMessagePayload(BaseModel):
    role: str
    content: str


class CompletionRequestPayload(BaseModel):
    messages: list[ChatMessagePayload]
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, gt=0)
    temperature: float = Field(default=TEMPERATURE, ge=0.0)
    top_p: float = Field(default=TOP_P, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=REPETITION_PENALTY, gt=0.0)
    enable_thinking: bool = True
    n: int = Field(default=1, gt=0)


class ConfidenceSummaryPayload(BaseModel):
    final: float | None
    mean: float | None
    tail10_mean: float | None


class CompletionChoicePayload(BaseModel):
    index: int
    completion: str
    confidence: float | None
    token_confidences: list[float]
    token_ids: list[int]
    confidence_summary: ConfidenceSummaryPayload
    finish_reason: str


class CompletionResponsePayload(BaseModel):
    completion: str
    confidence: float | None
    token_confidences: list[float]
    token_ids: list[int]
    confidence_summary: ConfidenceSummaryPayload
    finish_reason: str
    completions: list[CompletionChoicePayload]


class StreamTokenPayload(BaseModel):
    type: Literal["token"] = "token"
    index: int
    token_id: int
    text: str
    confidence: float


class StreamFinalPayload(BaseModel):
    type: Literal["final"] = "final"
    index: int
    completion: str
    confidence: float | None
    token_confidences: list[float]
    token_ids: list[int]
    confidence_summary: ConfidenceSummaryPayload
    finish_reason: str


class StreamBatchFinalPayload(BaseModel):
    type: Literal["batch_final"] = "batch_final"
    completions: list[CompletionChoicePayload]


def _to_generate_request(payload: CompletionRequestPayload) -> GenerateRequest:
    return GenerateRequest(
        messages=[ChatMessage(role=message.role, content=message.content) for message in payload.messages],
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        repetition_penalty=payload.repetition_penalty,
        enable_thinking=payload.enable_thinking,
        n=payload.n,
    )


def _to_choice(index: int, result: GenerateResult) -> CompletionChoicePayload:
    return CompletionChoicePayload(
        index=index,
        completion=result.completion,
        confidence=result.confidence,
        token_confidences=result.token_confidences,
        token_ids=result.token_ids,
        confidence_summary=ConfidenceSummaryPayload(
            final=result.confidence_summary.final,
            mean=result.confidence_summary.mean,
            tail10_mean=result.confidence_summary.tail10_mean,
        ),
        finish_reason=result.finish_reason,
    )


def _to_response(result: GenerateBatchResult) -> CompletionResponsePayload:
    choices = [_to_choice(index, completion) for index, completion in enumerate(result.completions)]
    first = choices[0]
    return CompletionResponsePayload(
        completion=first.completion,
        confidence=first.confidence,
        token_confidences=first.token_confidences,
        token_ids=first.token_ids,
        confidence_summary=first.confidence_summary,
        finish_reason=first.finish_reason,
        completions=choices,
    )


async def _stream_response(loaded: LoadedConfidenceModel, request: GenerateRequest) -> AsyncIterator[str]:
    async with MODEL_LOCK:
        async for event in async_generate_confidence_stream(loaded, request):
            if isinstance(event, StreamTokenEvent):
                payload = StreamTokenPayload(
                    index=event.index,
                    token_id=event.token_id,
                    text=event.text,
                    confidence=event.confidence,
                )
            elif isinstance(event, StreamFinalEvent):
                payload = StreamFinalPayload(
                    index=event.index,
                    completion=event.completion,
                    confidence=event.confidence,
                    token_confidences=event.token_confidences,
                    token_ids=event.token_ids,
                    confidence_summary=ConfidenceSummaryPayload(
                        final=event.confidence_summary.final,
                        mean=event.confidence_summary.mean,
                        tail10_mean=event.confidence_summary.tail10_mean,
                    ),
                    finish_reason=event.finish_reason,
                )
            elif isinstance(event, StreamBatchFinalEvent):
                payload = StreamBatchFinalPayload(
                    completions=[
                        _to_choice(index, completion) for index, completion in enumerate(event.completions)
                    ],
                )
            else:
                raise TypeError(f"Unexpected stream event: {event}")
            yield json.dumps(payload.model_dump(), ensure_ascii=False) + "\n"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL
    MODEL = load_confidence_model()
    yield
    MODEL = None


app = FastAPI(title="Unicorn Mafia Confidence Serving", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded" if MODEL is not None else "loading"}


@app.post("/complete")
async def complete(payload: CompletionRequestPayload) -> CompletionResponsePayload:
    loaded = MODEL
    if loaded is None:
        raise RuntimeError("Model is not loaded.")
    request = _to_generate_request(payload)
    async with MODEL_LOCK:
        result = await async_generate_batch_with_confidence(loaded, request)
    return _to_response(result)


@app.post("/complete/stream")
async def complete_stream(payload: CompletionRequestPayload) -> StreamingResponse:
    loaded = MODEL
    if loaded is None:
        raise RuntimeError("Model is not loaded.")
    request = _to_generate_request(payload)
    return StreamingResponse(_stream_response(loaded, request), media_type="application/x-ndjson")


def main() -> None:
    uvicorn.run(cast(str, "confidence_serving.server:app"), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
