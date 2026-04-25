# Unicorn Mafia Confidence Fine-Tuning

This repo now has two separate `uv` projects so vLLM-based data generation and TRL/Torch
fine-tuning can move independently.

## Step 1: Split Projects

Generate confidence traces with the vLLM project:

```bash
cd data_generation
uv sync
uv run python generate_trace.py
```

Fine-tune from those traces with the fine-tuning project:

```bash
cd fine_tuning
uv sync
uv run python finetune.py
```

Both scripts keep shared artifacts at the repo root:

- `traces/` for generated trace shards
- `outputs/` for fine-tuned adapters and model artifacts

The old GRPO/RL loop has been removed.

## Step 2: Gemma 4

Defaults now target `google/gemma-4-E2B-it`. The tokenizer assets are stored in
`tokenizers/gemma-4-E2B-it/`, and `project_settings.json` selects `<unused0>` as the
confidence token. In the Gemma 4 E2B tokenizer, `<unused0>` resolves to token id `6`.

## Step 3: More Evaluation Data

After that, add 4-5 verifiable datasets beyond GSM8K, covering math, knowledge, and
hallucination-sensitive tasks, then raise generation length enough for those tasks.
