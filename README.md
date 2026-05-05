# MetaGemma

This repo now keeps the two active experiments separated:

- `confidence_sft/`: trace generation, supervised fine-tuning, and inference for the model that emits confidence and relative sequence-position metrics.
- `reinforcement_learning/`: standalone Sudoku RL/GRPO experiment.
- `packages/sandbox_harness/`: reusable coding-agent sandbox package used by confidence SFT trace generation.
- `scripts/remote/`: shared Vast.ai sync/bootstrap/SSH helpers.
- `scripts/confidence_sft/` and `scripts/reinforcement_learning/`: experiment-specific artifact sync helpers.

## Confidence SFT

See `confidence_sft/README.md`.

Common commands:

```bash
cd confidence_sft/data_generation
uv run python generate_trace.py

cd ../fine_tuning
uv run python finetune.py

cd ../inference_server
uv run python -m inference_server
```

Sync configured confidence artifacts from the remote machine:

```bash
./scripts/confidence_sft/sync_artifacts.sh
```

## Reinforcement Learning

The RL experiment is self-contained under `reinforcement_learning/`.

Sync RL diagnostics from the remote machine:

```bash
./scripts/reinforcement_learning/sync_diagnostics.sh
```

## Remote Workflow

```bash
./scripts/remote/update_remote_instance.sh
./scripts/remote/sync_remote.sh
./scripts/remote/bootstrap_remote.sh
./scripts/remote/ssh_remote.sh
```
