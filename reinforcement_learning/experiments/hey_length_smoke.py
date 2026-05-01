from pathlib import Path
from typing import Any

import torch
from unsloth import FastVisionModel

from rl_trainer import PrintCallback, RLTrainer, RLTrainerConfig
from rl_trainer.generation import TransformersRolloutEngine
from rl_trainer.types import CompletionRecord, PromptBatch, RewardBatch, StepMetrics, TrainingExample

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 8192
MAX_COMPLETION_LENGTH = 256
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
LORA_RANK = 16

PROMPT_TEXT = "hey"
ENABLE_THINKING = True
DATASET_SIZE = 1000
MAX_STEPS = 100
OUTPUT_DIR = Path("outputs/hey_length_smoke")
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
PERIODIC_EVAL_STEPS = 10


def build_prompt() -> list[dict[str, object]]:
    return [{"role": "user", "content": [{"type": "text", "text": PROMPT_TEXT}]}]


class HeyDataset:
    def __init__(self, *, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _index: int) -> dict[str, object]:
        return {"prompt": build_prompt()}


async def short_completion_reward(batch: RewardBatch) -> list[float | None]:
    return [-sum(mask) for mask in batch.completion_mask]


class LengthEvalCallback:
    def __init__(self, *, rollout_engine: TransformersRolloutEngine) -> None:
        self.rollout_engine = rollout_engine

    def on_step_end(self, metrics: StepMetrics) -> None:
        if metrics.step % PERIODIC_EVAL_STEPS != 0:
            return

        rollout = self.rollout_engine.generate(
            PromptBatch(
                examples=[TrainingExample(prompt=build_prompt(), fields={})],
                prompts=[build_prompt()],
            )
        )
        lengths_tensor = rollout.completion_mask.sum(dim=1).detach().cpu()
        lengths = [float(length) for length in lengths_tensor.tolist()]
        print(
            f"eval_step={metrics.step} "
            f"completion_len_mean={sum(lengths) / len(lengths):.1f} "
            f"completion_len_min={min(lengths):.0f} "
            f"completion_len_max={max(lengths):.0f}",
            flush=True,
        )

    def on_completions(self, records: list[CompletionRecord]) -> None:
        del records


def load_model_and_tokenizer() -> tuple[Any, Any]:
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_STATE,
    )
    return model, tokenizer


def build_training_config() -> RLTrainerConfig:
    return RLTrainerConfig(
        temperature=1.0,
        learning_rate=5e-5,
        weight_decay=0.0,
        warmup_ratio=0.0,
        logging_steps=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=0,
        output_dir=OUTPUT_DIR,
        mask_truncated_completions=False,
        max_grad_norm=1.0,
        seed=RANDOM_STATE,
        shuffle=True,
        optimizer="adamw",
        empty_cache_steps=1,
        chat_template_kwargs={"enable_thinking": ENABLE_THINKING},
    )


def print_training_config(config: RLTrainerConfig) -> None:
    print(
        "hey_length_smoke_config "
        f"generations={config.num_generations} lr={config.learning_rate:.2e} "
        f"temperature={config.temperature:.2f} max_completion={config.max_completion_length} "
        f"thinking={ENABLE_THINKING} mask_truncated={config.mask_truncated_completions}",
        flush=True,
    )


def main() -> None:
    config = build_training_config()
    print_training_config(config)
    model, tokenizer = load_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout_engine = TransformersRolloutEngine(model, tokenizer, config, device)

    dataset = HeyDataset(size=DATASET_SIZE)
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_functions=[short_completion_reward],
        config=config,
        rollout_engine=rollout_engine,
        callbacks=[PrintCallback(), LengthEvalCallback(rollout_engine=rollout_engine)],
    )
    trainer.train()

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()
