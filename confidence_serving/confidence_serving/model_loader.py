from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from peft import PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from confidence_serving.settings import (
    ARTIFACT_DIR,
    BASE_MODEL_ID,
    CONFIDENCE_TOKEN,
    CONFIDENCE_TOKEN_ID,
    TORCH_DTYPE,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedConfidenceModel:
    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def _torch_dtype() -> torch.dtype | str:
    if TORCH_DTYPE == "auto":
        return "auto"
    if TORCH_DTYPE == "float16":
        return torch.float16
    if TORCH_DTYPE == "bfloat16":
        return torch.bfloat16
    if TORCH_DTYPE == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {TORCH_DTYPE}")


def verify_confidence_token(tokenizer: PreTrainedTokenizerBase) -> None:
    token_id = tokenizer.convert_tokens_to_ids(CONFIDENCE_TOKEN)
    if token_id != CONFIDENCE_TOKEN_ID:
        raise ValueError(f"{CONFIDENCE_TOKEN} resolved to token id {token_id}, expected {CONFIDENCE_TOKEN_ID}.")


def _load_confidence_row(artifact_dir: Path) -> torch.Tensor:
    row_path = artifact_dir / "confidence_lm_head_row.pt"
    row = torch.load(row_path, map_location="cpu")
    if not isinstance(row, torch.Tensor):
        raise TypeError(f"Expected {row_path} to contain a torch.Tensor.")
    if row.ndim != 1:
        raise ValueError(f"Expected confidence row to be rank 1, got shape {tuple(row.shape)}.")
    return row


def load_confidence_model() -> LoadedConfidenceModel:
    LOGGER.info("Loading tokenizer from %s", ARTIFACT_DIR)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(ARTIFACT_DIR, trust_remote_code=True),
    )
    verify_confidence_token(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading base model %s", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=_torch_dtype(),
        device_map="auto",
        trust_remote_code=True,
    )

    LOGGER.info("Loading PEFT adapter from %s", ARTIFACT_DIR)
    model = cast(nn.Module, PeftModel.from_pretrained(base_model, ARTIFACT_DIR))
    model.eval()

    confidence_row = _load_confidence_row(ARTIFACT_DIR)
    get_output_embeddings = getattr(model, "get_output_embeddings")
    lm_head = cast(nn.Embedding | nn.Linear, get_output_embeddings())
    with torch.no_grad():
        lm_head.weight[CONFIDENCE_TOKEN_ID].copy_(
            confidence_row.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
        )

    device = lm_head.weight.device
    LOGGER.info("Loaded model on %s", device)
    return LoadedConfidenceModel(model=model, tokenizer=tokenizer, device=device)
