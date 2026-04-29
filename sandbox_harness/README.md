# Agent Sandbox Harness

Reusable Python package for running coding-agent episodes against isolated
filesystem and command sandboxes.

The model loop belongs to the caller. This package provides:

- Pi-compatible `read`, `bash`, `edit`, and `write` tool schemas
- Pi-parity default system prompt construction
- sandbox backend interfaces with Bubblewrap and PRoot implementations
- per-episode workspace setup and cleanup
- JSONL episode summaries plus per-episode artifacts
- token trace data structures for later SFT or RL conversion

The package intentionally has no model, dataset, or fine-tuning dependencies.

Install from a caller project with:

```bash
uv add ../sandbox_harness
```
