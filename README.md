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

The fine-tuning script defaults to memory-saving settings: train/eval batch size 1
and gradient checkpointing enabled.

Both scripts keep shared artifacts at the repo root:

- `traces/` for generated trace shards
- `outputs/` for fine-tuned adapters and model artifacts

The old GRPO/RL loop has been removed.

## Step 2: Gemma 4

Defaults now target `google/gemma-4-E2B-it`. The tokenizer assets are stored in
`tokenizers/gemma-4-E2B-it/`, and `project_settings.json` selects `<unused0>` as the
confidence token. In the Gemma 4 E2B tokenizer, `<unused0>` resolves to token id `6`.

## Step 3: More Evaluation Data

Trace generation now uses a mixed verifiable dataset registry:

- GSM8K
- MATH-500
- MMLU-Pro
- ARC-Challenge
- TruthfulQA MC1
- FEVER

Generation enables Gemma thinking and caps each saved trace at 4096 total tokens
including prompt and completion.

Default generation samples up to 1200 SFT problems and 240 eval problems per
dataset, with 1 generation per problem. The SFT split renders two thirds of each
dataset with thinking enabled and the remaining third with thinking disabled.
Adjust `DATASET_SAMPLE_COUNTS` in
`data_generation/dataset_specs.py` or `NUM_GENERATIONS` in
`data_generation/generate_trace.py` for larger runs.
