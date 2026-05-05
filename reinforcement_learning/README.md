# Reinforcement Learning

Standalone Sudoku RL/GRPO experiment.

Run experiment entrypoints from this directory so their relative `outputs/...`
paths stay inside `reinforcement_learning/`:

```bash
cd reinforcement_learning
uv run python experiments/grpo_single.py
```

The local machine has no NVIDIA GPU, so training entrypoints should run on the
remote CUDA machine after syncing the repo with `scripts/remote/sync_remote.sh`.

Sync remote logs and diagnostics:

```bash
./scripts/reinforcement_learning/sync_diagnostics.sh
```
