# Sandbox Harness Requirements

## Purpose

Build a reusable Python wheel for running coding-agent episodes against isolated sandboxes. It must be installable from the existing data-generation project and from a future RL project with different dependencies.

## Naming

- Do not use project-specific names in package, module, artifact, or documentation names.
- Use neutral names such as `sandbox_harness` for the source package and `agent-sandbox-harness` or `coding-sandbox-harness` for the wheel.

## Packaging

- Use `uv` for project and package workflows.
- Prefer `uv_build` as the PEP 517 build backend unless it blocks a required feature.
- Keep dependencies minimal and independent of data-generation, fine-tuning, and inference environments.
- Do not depend on `vllm`, `torch`, `datasets`, or fine-tuning libraries from the harness package.

## Execution Model

- The model loop runs outside the sandbox in the caller's environment.
- The sandbox is only responsible for filesystem and command/tool effects.
- The same local model used by data generation should produce the agent turns.
- The harness must preserve token-level generation data, including token ids and top-k logprobs, for training.

## Pi Compatibility

- Match Pi's default coding-agent interface during training and inference.
- Reuse or mirror Pi's default system prompt and tool schemas.
- Default active tools are `read`, `bash`, `edit`, and `write`.
- Do not expose additional exploration tools such as `grep`, `find`, or `ls`; callers should install and invoke those utilities through `bash` when needed.
- Treat Pi's OpenAI-compatible message/tool format as the high-level episode schema.
- Treat the Gemma chat-template rendering as the token-level training format.

## Tool Serialization

- Pi sends tools to OpenAI-compatible servers as function tools.
- Assistant tool calls are represented as OpenAI `assistant.tool_calls` at the API layer.
- Tool results are represented as `role: "tool"` messages at the API layer.
- For `gemma-4-E2B-it`, the tokenizer chat template renders these into native tags:
  - tool declaration: `<|tool>declaration:name{...}<tool|>`
  - assistant call: `<|tool_call>call:name{...}<tool_call|>`
  - tool result: `<|tool_response>response:name{...}<tool_response|>`
- The harness should generate and parse this native rendered form when collecting token traces.

## Sandbox Requirements

- Provide a common backend interface so callers can switch sandbox implementations without changing the agent/tool loop.
- Implement `BubblewrapBackend` for training/RL and `ProotBackend` for locked-down inference containers.
- Each episode gets a fresh isolated filesystem.
- Mount only the episode workspace as writable inside the sandbox.
- Do not provide network access from sandboxes; networking is outside the harness requirements.
- Support wall-clock timeouts for all commands.
- Support CPU, memory, file-size, and process-count limits where the host permits it.
- Assume accidental resource abuse, not a malicious adversary.

## Bubblewrap Backend

- Use where `bwrap` can create user, mount, PID, and filesystem namespaces.
- Use a read-only rootfs with a per-episode writable workspace and isolated temporary directory.
- Fail fast with an actionable error if namespace creation is blocked by the host/container runtime.

## PRoot Backend

- Use where nested namespaces are unavailable.
- Emulate a rootfs and bind the per-episode workspace into it.
- Do not rely on Docker-in-Docker, host sysctls, privileged container flags, or Bubblewrap.
- Treat PRoot as a filesystem isolation aid, not a strong security boundary.
- Detect whether `ptrace` is blocked and report a clear unsupported-backend error if PRoot cannot run.

## Dataset Bootstrap Target

- Start with Python-only toy tasks.
- Prefer EvalPlus HumanEval+/MBPP+ transformed into small repositories:
  - `README.md` with the task prompt
  - `solution.py` or package source file
  - visible pytest tests
  - stronger hidden verifier tests
- Use QuixBugs Python as an additional smoke-test repair benchmark.
- Defer SWE-bench Lite until the sandbox and episode runner are stable because its repository and dependency setup is heavier.

## Episode Output

- Store coding episodes separately from the existing single-turn trace shards.
- Use JSONL for episode metadata and transcript summaries.
- Store artifacts per episode, including final diff, verifier output, and full transcript.
- Keep enough information to later convert episodes into supervised traces or RL rollouts.

## Repository Constraints

- Do not modify anything under `repos/`; those checkouts are references only.
- Keep the current single-turn data-generation path unchanged.
