from __future__ import annotations

from inference_server.generate import ChatMessage, GenerateRequest, generate_batch_with_confidence, generate_with_confidence
from inference_server.model_loader import load_confidence_model
from inference_server.settings import CONFIDENCE_TOKEN, CONFIDENCE_TOKEN_ID


def main() -> None:
    loaded = load_confidence_model()
    token_id = loaded.tokenizer.convert_tokens_to_ids(CONFIDENCE_TOKEN)
    if token_id != CONFIDENCE_TOKEN_ID:
        raise AssertionError(f"{CONFIDENCE_TOKEN} resolved to {token_id}, expected {CONFIDENCE_TOKEN_ID}.")

    result = generate_with_confidence(
        loaded,
        GenerateRequest(
            messages=[ChatMessage(role="user", content="What is 2 + 2? Answer briefly.")],
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            enable_thinking=False,
        ),
    )
    if CONFIDENCE_TOKEN in result.completion:
        raise AssertionError(f"Generated text contains {CONFIDENCE_TOKEN}.")
    if len(result.token_confidences) != len(result.token_ids):
        raise AssertionError("token_confidences length does not match token_ids length.")
    for confidence in result.token_confidences:
        if not 0.0 <= confidence <= 1.0:
            raise AssertionError(f"Confidence out of range: {confidence}")

    batch_result = generate_batch_with_confidence(
        loaded,
        GenerateRequest(
            messages=[ChatMessage(role="user", content="What is 2 + 2? Answer briefly.")],
            max_new_tokens=8,
            temperature=0.7,
            top_p=1.0,
            enable_thinking=False,
            n=2,
        ),
    )
    if len(batch_result.completions) != 2:
        raise AssertionError(f"Expected 2 completions, got {len(batch_result.completions)}.")

    print(result.completion)
    print({"confidence": result.confidence, "finish_reason": result.finish_reason})


if __name__ == "__main__":
    main()
