# Gemma 3 GRPO and Confidence SFT on GSM8K

Minimal TRL GRPO training script plus an experimental confidence-logit SFT path for
`google/gemma-3-1b-it` on `openai/gsm8k`.

```bash
uv run python train.py
```

The default run is intentionally small (`2048` train samples, `250` steps, LoRA enabled) so it is usable as a bootstrap experiment. Scale it up with flags:
edit the constants at the top of `train.py`.

Current defaults:

- `NUM_GENERATIONS = 16`
- `GENERATION_BATCH_SIZE = 16`
- `USE_VLLM = True`
- `VLLM_GPU_MEMORY_UTILIZATION = 0.6`
- `ENABLE_THINKING = True`
- reward functions: `math_verify_reward`, `exact_match_reward`, and `format_reward`

vLLM is enabled by default. TRL currently supports vLLM versions from `0.12.0` to `0.18.0`, so the project pins `vllm>=0.17.0,<=0.18.0`.

On the server, after syncing:

```bash
uv sync --refresh-package vllm
uv run python -c "import vllm; print(vllm.__version__)"
```

The reward tracker prints one line per question group, not per flattened reward call.

## Confidence SFT

The confidence SFT experiment trains the model so `sigmoid(logits[..., 6])` acts as a
running confidence that the final answer will verify as correct. Gemma's `<unused0>`
token is expected to resolve to token id `6`; both scripts check this before running.

First generate trace shards with vLLM:

```bash
uv run python generate_trace.py
```

Configuration is hardcoded as constants at the top of `generate_trace.py`. Defaults use a
small test split:

- `250` shuffled GSM8K train problems for SFT
- `50` shuffled GSM8K train problems for eval
- `16` completions per problem
- `20` top logprobs per generated token
- token id `6` suppressed during generation and removed from KL targets

Then run the confidence SFT:

```bash
uv run python finetune.py
```

`finetune.py` is also configured by constants at the top of the file. It uses LoRA by
default and trains a small row-6 confidence vector separately, then copies it into
`lm_head.weight[6]` at save time. This avoids allocating Adam state for Gemma's full tied
embedding/lm-head matrix. With PEFT enabled, the output directory contains the adapter plus
`unused0_lm_head_row.pt`; set `SAVE_FULL_MODEL = True` for a merged full-model save.

Default SFT batching is `TRAIN_BATCH_SIZE = 4` with `GRADIENT_ACCUMULATION_STEPS = 4`,
so each optimizer step sees `16` completion traces. Confidence metrics include final-token
and last-10%-of-tokens accuracy, balanced accuracy, AUROC, and Brier score.
`BALANCE_TRAIN_BATCHES = True` arranges each microbatch to contain both correct and
incorrect traces when both classes exist, without oversampling or changing epoch class
counts.

Trace files are written as paired shards:

```text
trace-00000.meta.parquet
trace-00000.arrays.npz
```

The Parquet metadata has one row per completion, including offsets into the arrays. The NPZ
payload stores fixed-size tensors:

```text
prompt_token_ids: int32[total_prompt_tokens]
completion_token_ids: int32[total_completion_tokens]
top_logprob_token_ids: int32[total_completion_tokens, 20]
top_logprobs: float32[total_completion_tokens, 20]
top_logprob_mask: bool[total_completion_tokens, 20]
```

For gated Google Gemma access, authenticate with Hugging Face or export `HF_TOKEN` in your
shell; do not commit tokens to the repo.

## Logging

Weights & Biases logging is enabled by default with `REPORT_TO = "wandb"` in `train.py`.
Before the first online run, authenticate once:

```bash
uv run wandb login
```

Training metrics emitted by TRL/Hugging Face Trainer are reported to wandb automatically.
The custom reward tracker also logs:

- `reward/exact_match_count`
- `reward/exact_match_total`
- `reward/exact_match_rate`
- `reward/predictions` tables with expected, gold, prediction, and correctness rows

Set these environment variables when needed:

```bash
WANDB_PROJECT=unicorn-mafia-grpo uv run python train.py
WANDB_MODE=offline uv run python train.py
WANDB_DISABLED=true uv run python train.py
```

For a terminal UI, run W&B LEET from the same working directory in another SSH/tmux pane:

```bash
uv run wandb beta leet run
```

LEET reads `./wandb/latest-run/` by default and updates from the local `.wandb` file while
training is running. You can also point it at a specific run directory or transaction log:

```bash
uv run wandb beta leet run ./wandb/run-20250813_124246-n67z9ude
uv run wandb beta leet run ./wandb/run-20250813_124246-n67z9ude/run-n67z9ude.wandb
```

The training script disables tqdm by default (`DISABLE_TQDM = True`) so LEET is the main
progress surface instead of an in-process progress bar.

## Fast Path Dependencies

`flash-linear-attention` is installed by default. On Linux, `torch` is pinned to the PyTorch CUDA 12.8 wheel index via `tool.uv.sources`.

`causal-conv1d` is optional because it may need to compile against the local CUDA toolkit; install it only when the system CUDA toolkit matches the CUDA version used by PyTorch.

```bash
uv sync
uv sync --extra conv1d
```

If `causal-conv1d` fails with a CUDA mismatch, keep the default `uv sync` install and do not enable the `conv1d` extra. Pinning the PyTorch wheel to CUDA 12.8 does not change the system `nvcc`; check it with:

```bash
uv run python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

## Provenance

The trainer shape follows Hugging Face TRL's `GRPOTrainer` examples: a dataset with a `prompt` column, custom reward functions, and `GRPOConfig`.

The GSM8K answer extraction follows the dataset convention where final answers appear after `####`.

The prompt is stored as `system`/`user` chat messages so TRL can apply the model tokenizer's chat template. Thinking mode is enabled by default through `chat_template_kwargs={"enable_thinking": True}`. The system message is a minimal local instruction that asks for GSM8K's `#### <answer>` final-answer format. The bootstrap hyperparameters are local defaults, not copied from a benchmark recipe. Treat them as a starting point to tune.
