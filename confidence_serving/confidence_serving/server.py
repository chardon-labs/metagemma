from __future__ import annotations

import logging
import json
from contextlib import asynccontextmanager
from threading import Lock
from collections.abc import Iterator
from typing import Literal, cast

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from confidence_serving.generate import (
    ChatMessage,
    GenerateRequest,
    GenerateResult,
    StreamFinalEvent,
    StreamTokenEvent,
    generate_confidence_stream,
    generate_with_confidence,
)
from confidence_serving.model_loader import LoadedConfidenceModel, load_confidence_model
from confidence_serving.settings import HOST, MAX_NEW_TOKENS, PORT, REPETITION_PENALTY, TEMPERATURE, TOP_P


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

MODEL: LoadedConfidenceModel | None = None
MODEL_LOCK = Lock()


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


class ConfidenceSummaryPayload(BaseModel):
    final: float | None
    mean: float | None
    tail10_mean: float | None


class CompletionResponsePayload(BaseModel):
    completion: str
    confidence: float | None
    token_confidences: list[float]
    token_ids: list[int]
    confidence_summary: ConfidenceSummaryPayload
    finish_reason: str


class StreamTokenPayload(BaseModel):
    type: Literal["token"] = "token"
    token_id: int
    text: str
    confidence: float


class StreamFinalPayload(BaseModel):
    type: Literal["final"] = "final"
    completion: str
    confidence: float | None
    token_confidences: list[float]
    token_ids: list[int]
    confidence_summary: ConfidenceSummaryPayload
    finish_reason: str


def _to_generate_request(payload: CompletionRequestPayload) -> GenerateRequest:
    return GenerateRequest(
        messages=[ChatMessage(role=message.role, content=message.content) for message in payload.messages],
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        repetition_penalty=payload.repetition_penalty,
        enable_thinking=payload.enable_thinking,
    )


def _to_response(result: GenerateResult) -> CompletionResponsePayload:
    return CompletionResponsePayload(
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


def _stream_response(loaded: LoadedConfidenceModel, request: GenerateRequest) -> Iterator[str]:
    with MODEL_LOCK:
        for event in generate_confidence_stream(loaded, request):
            if isinstance(event, StreamTokenEvent):
                payload = StreamTokenPayload(
                    token_id=event.token_id,
                    text=event.text,
                    confidence=event.confidence,
                )
            elif isinstance(event, StreamFinalEvent):
                payload = StreamFinalPayload(
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
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded" if MODEL is not None else "loading"}


@app.post("/complete")
def complete(payload: CompletionRequestPayload) -> CompletionResponsePayload:
    loaded = MODEL
    if loaded is None:
        raise RuntimeError("Model is not loaded.")
    request = _to_generate_request(payload)
    with MODEL_LOCK:
        result = generate_with_confidence(loaded, request)
    return _to_response(result)


@app.post("/complete/stream")
def complete_stream(payload: CompletionRequestPayload) -> StreamingResponse:
    loaded = MODEL
    if loaded is None:
        raise RuntimeError("Model is not loaded.")
    request = _to_generate_request(payload)
    return StreamingResponse(_stream_response(loaded, request), media_type="application/x-ndjson")


def main() -> None:
    uvicorn.run(cast(str, "confidence_serving.server:app"), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
