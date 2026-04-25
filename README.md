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

Next, switch the defaults to `gemma-4-E2B-it` and inspect the tokenizer to choose the
confidence token id instead of assuming Gemma 3's `<unused0>` id.

## Step 3: More Evaluation Data

After that, add 4-5 verifiable datasets beyond GSM8K, covering math, knowledge, and
hallucination-sensitive tasks, then raise generation length enough for those tasks.
