from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from confidence_trace import extract_prediction, math_verify_label


_HASH_FINAL_RE = re.compile(r"####\s*\(?([A-Z])\)?", re.IGNORECASE)
_ANSWER_IS_RE = re.compile(r"answer is \(?([A-Z])\)?", re.IGNORECASE)
_ANSWER_COLON_RE = re.compile(r"answer\s*:\s*\$?\(?([A-Z])\)?\$?", re.IGNORECASE)
_LABEL_FINAL_RE = re.compile(r"####\s*([A-Z][A-Z _-]+)", re.IGNORECASE)
_LABEL_COLON_RE = re.compile(r"answer\s*:\s*([A-Z][A-Z _-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ScoreResult:
    label: int
    extracted_prediction: str
    normalized_prediction: str
    normalized_gold: str
    scorer: str
    score_error: str | None = None


def _normalize_letter(value: str) -> str:
    return value.strip().upper()


def _valid_letters(problem: Mapping[str, Any]) -> list[str]:
    labels = problem.get("choice_labels", [])
    if isinstance(labels, Sequence) and not isinstance(labels, str):
        return [str(label).upper() for label in labels]
    return []


def _extract_final_letter(text: str, valid_letters: Sequence[str]) -> str:
    for pattern in (_HASH_FINAL_RE, _ANSWER_IS_RE, _ANSWER_COLON_RE):
        match = pattern.search(text)
        if match is not None:
            candidate = _normalize_letter(match.group(1))
            if candidate in valid_letters:
                return candidate

    matches = re.findall(r"\b([A-Z])\b", text.upper(), re.DOTALL)
    for candidate in reversed(matches):
        if candidate in valid_letters:
            return candidate
    return ""


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ").strip().upper())


def _extract_final_label(text: str, labels: Sequence[str]) -> str:
    normalized_labels = {_normalize_label(label): _normalize_label(label) for label in labels}
    for pattern in (_LABEL_FINAL_RE, _LABEL_COLON_RE):
        match = pattern.search(text)
        if match is None:
            continue
        candidate = _normalize_label(match.group(1))
        if candidate in normalized_labels:
            return normalized_labels[candidate]

    normalized_text = _normalize_label(text)
    for label in normalized_labels:
        if re.search(rf"\b{re.escape(label)}\b", normalized_text):
            return normalized_labels[label]
    return ""


def _score_math(completion: str, gold_answer: str, scorer: str) -> ScoreResult:
    extracted = extract_prediction(completion)
    try:
        label = math_verify_label(completion, gold_answer)
        error = None
    except Exception as exc:
        label = 0
        error = f"{type(exc).__name__}: {exc}"
    return ScoreResult(
        label=label,
        extracted_prediction=extracted,
        normalized_prediction=extracted,
        normalized_gold=gold_answer,
        scorer=scorer,
        score_error=error,
    )


def _score_multiple_choice(completion: str, problem: Mapping[str, Any], scorer: str) -> ScoreResult:
    valid_letters = _valid_letters(problem)
    gold = _normalize_letter(str(problem["gold_answer"]))
    prediction = _extract_final_letter(completion, valid_letters)
    return ScoreResult(
        label=int(prediction == gold),
        extracted_prediction=prediction,
        normalized_prediction=prediction,
        normalized_gold=gold,
        scorer=scorer,
    )


def _score_label(completion: str, problem: Mapping[str, Any], scorer: str) -> ScoreResult:
    labels = [str(label) for label in problem.get("choice_labels", [])]
    gold = _normalize_label(str(problem["gold_answer"]))
    prediction = _extract_final_label(completion, labels)
    return ScoreResult(
        label=int(prediction == gold),
        extracted_prediction=prediction,
        normalized_prediction=prediction,
        normalized_gold=gold,
        scorer=scorer,
    )


def score_completion(completion: str, problem: Mapping[str, Any]) -> ScoreResult:
    scorer = str(problem["scorer"])
    if scorer == "math_verify":
        return _score_math(completion, str(problem["gold_answer"]), scorer)
    if scorer == "multiple_choice_exact":
        return _score_multiple_choice(completion, problem, scorer)
    if scorer == "label_exact":
        return _score_label(completion, problem, scorer)
    raise ValueError(f"Unknown scorer: {scorer}")
