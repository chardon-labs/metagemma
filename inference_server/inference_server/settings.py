from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, TypedDict, cast


class ProjectSettings(TypedDict):
    model_id: str
    tokenizer_id: str
    confidence_token: str
    confidence_token_id: int
    trace_dir: str
    output_dir: str


REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "project_settings.json"


def load_project_settings() -> ProjectSettings:
    return cast(ProjectSettings, json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))


PROJECT_SETTINGS = load_project_settings()
BASE_MODEL_ID = PROJECT_SETTINGS["model_id"]
ARTIFACT_DIR = REPO_ROOT / PROJECT_SETTINGS["output_dir"]
CONFIDENCE_TOKEN = PROJECT_SETTINGS["confidence_token"]
CONFIDENCE_TOKEN_ID = PROJECT_SETTINGS["confidence_token_id"]

MAX_NEW_TOKENS = 2048
TEMPERATURE = 1.0
TOP_P = 0.95
REPETITION_PENALTY = 1.0
ENABLE_THINKING = True

HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
PORT = 8010
AUTH_TOKEN = os.environ.get("INFERENCE_AUTH_TOKEN") or None

TorchDTypeName = Literal["auto", "float16", "bfloat16", "float32"]
TORCH_DTYPE: TorchDTypeName = "bfloat16"

AttentionImplementationName = Literal["eager", "sdpa", "flash_attention_2", "flash_attention_3"]
ATTENTION_IMPLEMENTATION: AttentionImplementationName = "sdpa"
