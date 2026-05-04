from dataclasses import replace
from pathlib import Path

import torch
from experiments import muon_curriculum as base
from rl_trainer import GRPOAlgorithmConfig, JSONLLogCallback, PrintCallback, RLTrainer, RLTrainerConfig, TrainerCallback

OUTPUT_DIR = Path("outputs/grpo_curriculum")
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
GRPO_EPSILON = 0.2
GRPO_MINI_BATCH_SIZE = 8

_build_muon_training_config = base.build_training_config


def build_training_config() -> RLTrainerConfig:
    config = _build_muon_training_config()
    return replace(
        config,
        output_dir=OUTPUT_DIR,
        algorithm=GRPOAlgorithmConfig(
            epsilon=GRPO_EPSILON,
            num_iterations=1,
            mini_batch_size=GRPO_MINI_BATCH_SIZE,
        ),
    )


def print_training_config(config: RLTrainerConfig) -> None:
    print(
        "grpo_curriculum_config "
        f"batch_size={config.batch_size} generations={config.num_generations} lr={base.LEARNING_RATE:.2e} "
        "optimizer=muon muon_adjust_lr=match_rms_adamw "
        f"grpo_epsilon={GRPO_EPSILON:.2f} grpo_mini_batch={GRPO_MINI_BATCH_SIZE} "
        f"backward_microbatch={config.backward_microbatch_size} "
        f"weight_decay={config.weight_decay:.3g} temperature={config.temperature:.2f} "
        f"max_completion={config.max_completion_length} "
        f"mask_truncated={config.mask_truncated_completions} "
        f"vllm_sync_steps={base.VLLM_SYNC_STEPS}",
        flush=True,
    )


def main() -> None:
    config = build_training_config()
    print_training_config(config)
    model, tokenizer = base.load_model_and_tokenizer()
    rollout_engine = base.build_vllm_engine(base.MODEL_NAME, tokenizer, config, sync_steps=base.VLLM_SYNC_STEPS)
    curriculum = base.SudokuCurriculum()
    dataset = base.SudokuDataset(
        size=base.DATASET_SIZE,
        curriculum=curriculum,
        seed=base.RANDOM_STATE,
    )

    print("Dataset sample:")
    print(dataset[0])

    callbacks: list[TrainerCallback] = [
        PrintCallback(),
        JSONLLogCallback(OUTPUT_DIR / "logs"),
        base.CurriculumCallback(curriculum),
    ]

    trainer = RLTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_functions=base.SUDOKU_REWARD_FUNCTIONS,
        config=config,
        rollout_engine=rollout_engine,
        callbacks=callbacks,
    )
    trainer.train()

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    del trainer
    del rollout_engine
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
