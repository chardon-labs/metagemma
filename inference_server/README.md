# Inference Server

FastAPI inference server and dashboard for confidence-aware generation.

The server loads:

- base model from `../project_settings.json`
- adapter/tokenizer/confidence row/position row from `../project_settings.json` `output_dir`
- confidence and position token ids from `../project_settings.json`

The confidence for each generated token is computed before sampling:

```python
confidence = sigmoid(logits[:, 6])
position = sigmoid(logits[:, 7])
```

Token ids `6` and `7` are then suppressed before choosing the next generated token.

Sampling is deterministic by default. `INFERENCE_SEED` defaults to `42`, and each
request may also pass a `seed` field. The same prompt and sampling settings
produce the same completions for the same seed.

## Run

```bash
cd inference_server
uv run python -m inference_server
```

Local dashboard:

```text
http://127.0.0.1:8010
```

For a public Vast.ai deployment, bind to all interfaces and set a shared token:

```bash
cd inference_server
INFERENCE_HOST=0.0.0.0 INFERENCE_AUTH_TOKEN='replace-with-a-secret' uv run python -m inference_server
```

Share the mapped Vast URL with the token query parameter:

```text
http://PUBLIC_IP:PUBLIC_PORT/?token=replace-with-a-secret
```

The UI stores the token locally and sends it as a bearer token for API requests. API clients can also pass `Authorization: Bearer ...`.

Example request:

```bash
curl -s http://127.0.0.1:8010/complete \
  -H 'authorization: Bearer replace-with-a-secret' \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Solve the following math problem. Give the final answer in the format: #### <answer>\n\nHow many distinct arrangements of the letters in MISSISSIPPI have no two S's adjacent?"}],
    "max_new_tokens": 2048,
    "temperature": 1.0,
    "top_p": 1,
    "enable_thinking": false,
    "n": 1,
    "seed": 42
  }'
```

Set `n` greater than `1` to sample multiple completions from the same rendered prompt in one model batch. The response keeps the original top-level single-completion fields for compatibility and also includes all samples:

```json
{
  "completion": "...",
  "confidence": 0.91,
  "token_confidences": [0.82, 0.91],
  "token_positions": [0.02, 0.04],
  "token_ids": [123, 456],
  "confidence_summary": {"final": 0.91, "mean": 0.865, "tail10_mean": 0.865},
  "finish_reason": "stop",
  "completions": [
    {
      "index": 0,
      "completion": "...",
      "confidence": 0.91,
      "token_confidences": [0.82, 0.91],
      "token_positions": [0.02, 0.04],
      "token_ids": [123, 456],
      "confidence_summary": {"final": 0.91, "mean": 0.865, "tail10_mean": 0.865},
      "finish_reason": "stop"
    }
  ]
}
```

Streaming endpoint:

```text
POST /complete/stream
```

It returns newline-delimited JSON. Token events arrive as:

```json
{"type":"token","index":0,"token_id":123,"text":" answer","confidence":0.82,"position":0.02}
```

Each completion emits a final event:

```json
{"type":"final","index":0,"completion":"...","confidence":0.91,"token_confidences":[0.82,0.91],"token_positions":[0.02,0.04],"token_ids":[123,456],"confidence_summary":{"final":0.91,"mean":0.865,"tail10_mean":0.865},"finish_reason":"stop"}
```

The stream ends with `{"type":"batch_final","completions":[...]}`.

## Frontend

The frontend is served by FastAPI at `/` and calls `/complete/stream` on the same origin.

The UI source is TypeScript in `static/app.ts`. Rebuild the served JavaScript after frontend edits:

```bash
cd inference_server
bun install
bun run build
```

## Smoke Test

```bash
cd inference_server
uv run inference-smoke-test
```
