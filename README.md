# MetaGemma

Steps to replicate the results.

## 1. Generate traces

```bash
cd data_generation
uv run python generate_trace.py
```

This reads `../project_settings.json` and writes trace shards to the configured
`trace_dir`, currently:

```text
traces/gemma-4-E2B-it-mixed-confidence-1200
```

To add only newly configured dataset examples to an existing trace snapshot:

```bash
cd data_generation
uv run python generate_trace.py --append
```

## 2. Fine-tune

```bash
cd fine_tuning
uv run python finetune.py
```

This reads the generated traces from `trace_dir` and writes the adapter,
tokenizer, config, and `confidence_lm_head_row.pt` to the configured
`output_dir`, currently:

```text
outputs/gemma-4-E2B-it-mixed-confidence-3
```

## 3. Launch inference

```bash
cd inference_server
uv run python -m inference_server
```

Open:

```text
http://127.0.0.1:8010
```

For a public server:

```bash
cd inference_server
INFERENCE_HOST=0.0.0.0 INFERENCE_AUTH_TOKEN='replace-with-a-secret' uv run python -m inference_server
```

Then open the mapped URL with the token:

```text
http://PUBLIC_IP:PUBLIC_PORT/?token=replace-with-a-secret
```

## Sync remote artifacts

```bash
./scripts/sync_remote_artifacts.sh
```

This pulls the configured `trace_dir` and `output_dir` from the remote machine into
local `data/`, preserving their relative paths.
