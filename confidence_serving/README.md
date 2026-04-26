# Confidence Serving

Minimal Hugging Face/PEFT inference server that emits completions with per-token confidence.

The server loads:

- base model from `../project_settings.json`
- adapter/tokenizer/confidence row from `../project_settings.json` `output_dir`
- confidence token/token id from `../project_settings.json`

The confidence for each generated token is computed before sampling:

```python
confidence = sigmoid(logits[:, 6])
```

Token id `6` is then suppressed before choosing the next generated token.

## Run

```bash
cd confidence_serving
uv run confidence-server
```

Server URL:

```text
http://127.0.0.1:8010
```

Example request:

```bash
curl -s http://127.0.0.1:8010/complete \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
    "max_new_tokens": 32,
    "temperature": 0,
    "top_p": 1,
    "enable_thinking": false
  }'
```

Streaming endpoint:

```text
POST /complete/stream
```

It returns newline-delimited JSON. Token events arrive as:

```json
{"type":"token","token_id":123,"text":" answer","confidence":0.82}
```

The last event is:

```json
{"type":"final","completion":"...","confidence":0.91,"token_confidences":[0.82,0.91],"token_ids":[123,456],"confidence_summary":{"final":0.91,"mean":0.865,"tail10_mean":0.865},"finish_reason":"stop"}
```

## Frontend

Open:

```text
../confidence_frontend/index.html
```

The frontend calls `http://127.0.0.1:8010/complete/stream` and plots token confidence as the response streams.

## Smoke Test

```bash
cd confidence_serving
uv run confidence-smoke-test
```
