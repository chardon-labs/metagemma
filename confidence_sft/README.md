# Confidence SFT

Steps to replicate the confidence and relative sequence-position model.

Configuration lives in `project_settings.json`. Paths in that file are relative
to `confidence_sft/`.

## 1. Generate traces

```bash
cd confidence_sft/data_generation
uv run python generate_trace.py
```

This writes trace shards to the configured `trace_dir`, currently:

```text
traces/gemma-4-E2B-it-mixed-confidence-4
```

To add only newly configured dataset examples to an existing trace snapshot:

```bash
cd confidence_sft/data_generation
uv run python generate_trace.py --append
```

Coding-agent traces use the shared sandbox harness in `packages/sandbox_harness`:

```bash
cd confidence_sft/data_generation
uv run python generate_coding_traces.py
```

## 2. Fine-tune

```bash
cd confidence_sft/fine_tuning
uv run python finetune.py
```

This reads the generated traces from `trace_dir` and writes the adapter,
tokenizer, config, `confidence_lm_head_row.pt`, and `position_lm_head_row.pt` to
the configured `output_dir`, currently:

```text
outputs/gemma-4-E2B-it-mixed-confidence-4
```

## 3. Launch inference

```bash
cd confidence_sft/inference_server
uv run python -m inference_server
```

Open:

```text
http://127.0.0.1:8010
```

For a public server:

```bash
cd confidence_sft/inference_server
INFERENCE_HOST=0.0.0.0 INFERENCE_AUTH_TOKEN='replace-with-a-secret' uv run python -m inference_server
```

Then open the mapped URL with the token:

```text
http://PUBLIC_IP:PUBLIC_PORT/?token=replace-with-a-secret
```

## Sync remote artifacts

```bash
./scripts/confidence_sft/sync_artifacts.sh
```

This pulls the configured `trace_dir`, configured `output_dir`, and optional
coding-agent traces from the remote `confidence_sft/` folder into local
`data/confidence_sft/`, preserving their relative paths.
