from dataclasses import replace
from pathlib import Path
import random

import torch
from experiments import muon_single as base
from rl_trainer import GRPOAlgorithmConfig, JSONLLogCallback, PrintCallback, RLTrainer, RLTrainerConfig

OUTPUT_DIR = Path("outputs/grpo_single")
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
        "grpo_single_config "
        f"generations={config.num_generations} lr={base.LEARNING_RATE:.2e} "
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
    rng = random.Random(base.RANDOM_STATE)
    model, tokenizer = base.load_model_and_tokenizer()
    rollout_engine = base.build_vllm_engine(base.MODEL_NAME, tokenizer, config, sync_steps=base.VLLM_SYNC_STEPS)

    puzzle = base.select_smoke_puzzle(rollout_engine=rollout_engine, rng=rng)
    base.print_puzzle(puzzle)

    base_solves = base.evaluate_puzzle(
        rollout_engine=rollout_engine,
        puzzle=puzzle,
        completion_count=base.EVAL_COMPLETIONS,
    )
    print(f"base_exact_solve_rate={base_solves}/{base.EVAL_COMPLETIONS}")

    dataset = base.SinglePuzzleDataset(puzzle=puzzle, size=base.DATASET_SIZE)
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_functions=base.SUDOKU_REWARD_FUNCTIONS,
        config=config,
        rollout_engine=rollout_engine,
        callbacks=[
            PrintCallback(),
            JSONLLogCallback(OUTPUT_DIR / "logs"),
            base.SudokuEvalCallback(rollout_engine=rollout_engine, puzzle=puzzle),
        ],
    )
    trainer.train()

    trained_solves = base.evaluate_puzzle(
        rollout_engine=rollout_engine,
        puzzle=puzzle,
        completion_count=base.EVAL_COMPLETIONS,
    )
    print(f"trained_exact_solve_rate={trained_solves}/{base.EVAL_COMPLETIONS}")

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    del trainer
    del rollout_engine
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
