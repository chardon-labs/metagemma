from pathlib import Path

import torch

from experiments.utils.support import (
    SudokuValidationCallback,
    build_curriculum_print_callback,
    build_vllm_engine,
    load_model_and_tokenizer,
    load_sudoku_validation_puzzles,
)
from rl_trainer import GRPOAlgorithmConfig, JSONLLogCallback, MuonOptimizerConfig, RLTrainer, RLTrainerConfig
from rl_trainer.callbacks import TrainerCallback
from tasks.sudoku import SUDOKU_REWARD_FUNCTIONS, CurriculumCallback, SudokuCurriculum, SudokuDataset

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 2048
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
FULL_FINETUNING = True
DATASET_SIZE = 1000
MAX_STEPS = 240
OUTPUT_DIR = Path("outputs/grpo_curriculum")
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
LEARNING_RATE = 1e-5
INITIAL_DIFFICULTY = 0.35
MAX_GRAD_NORM = 5.0
VALIDATION_SET_PATH = Path(__file__).resolve().parent / "fixtures" / "sudoku_validation_128.json"
VALIDATION_STEPS = 20
VALIDATION_COMPLETIONS_PER_PUZZLE = 1

GRPO_EPSILON = 0.2
GRPO_MINI_BATCH_SIZE = 16

VLLM_GPU_MEMORY_UTILIZATION = 0.20
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_ENFORCE_EAGER = True
VLLM_SYNC_STEPS = 1
VLLM_SYNC_BACKEND = "inprocess"
VLLM_SYNC_CHUNK_BYTES = 8 * 1024 * 1024 * 1024


def build_training_config() -> RLTrainerConfig:
    return RLTrainerConfig(
        warmup_ratio=0.0,
        warmup_steps=10,
        logging_steps=1,
        batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=16,
        backward_microbatch_size=8,
        max_seq_length=MAX_SEQ_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=0,
        output_dir=OUTPUT_DIR,
        optimizer=MuonOptimizerConfig(
            learning_rate=LEARNING_RATE,
            weight_decay=0.0,
            adjust_lr_fn="match_rms_adamw",
        ),
        algorithm=GRPOAlgorithmConfig(
            epsilon=GRPO_EPSILON,
            num_iterations=1,
            mini_batch_size=GRPO_MINI_BATCH_SIZE,
        ),
        temperature=1.0,
        mask_truncated_completions=False,
        max_grad_norm=MAX_GRAD_NORM,
        seed=RANDOM_STATE,
        shuffle=True,
        empty_cache_steps=1,
    )


def print_training_config(config: RLTrainerConfig) -> None:
    print(
        "grpo_curriculum_config "
        f"batch_size={config.batch_size} generations={config.num_generations} lr={LEARNING_RATE:.2e} "
        "optimizer=muon muon_adjust_lr=match_rms_adamw "
        f"grpo_epsilon={GRPO_EPSILON:.2f} grpo_mini_batch={GRPO_MINI_BATCH_SIZE} "
        f"backward_microbatch={config.backward_microbatch_size} "
        f"max_grad_norm={config.max_grad_norm:.1f} "
        f"weight_decay={config.weight_decay:.3g} temperature={config.temperature:.2f} "
        f"max_seq={config.max_seq_length} "
        f"mask_truncated={config.mask_truncated_completions} "
        f"vllm_sync_steps={VLLM_SYNC_STEPS}",
        flush=True,
    )


def main() -> None:
    config = build_training_config()
    print_training_config(config)
    model, tokenizer = load_model_and_tokenizer(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
        full_finetuning=FULL_FINETUNING,
    )
    rollout_engine = build_vllm_engine(
        model_name_or_path=MODEL_NAME,
        tokenizer=tokenizer,
        config=config,
        sync_steps=VLLM_SYNC_STEPS,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        enforce_eager=VLLM_ENFORCE_EAGER,
        sync_chunk_bytes=VLLM_SYNC_CHUNK_BYTES,
        sync_backend=VLLM_SYNC_BACKEND,
    )
    curriculum = SudokuCurriculum(difficulty=INITIAL_DIFFICULTY)
    dataset = SudokuDataset(
        size=DATASET_SIZE,
        curriculum=curriculum,
        seed=RANDOM_STATE,
    )

    print("Dataset sample:")
    print(dataset[0])

    callbacks: list[TrainerCallback] = [
        SudokuValidationCallback(
            rollout_engine=rollout_engine,
            puzzles=load_sudoku_validation_puzzles(VALIDATION_SET_PATH),
            eval_steps=VALIDATION_STEPS,
            completions_per_puzzle=VALIDATION_COMPLETIONS_PER_PUZZLE,
        ),
        CurriculumCallback(curriculum),
        build_curriculum_print_callback(curriculum),
        JSONLLogCallback(OUTPUT_DIR / "logs"),
    ]

    trainer = RLTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_functions=SUDOKU_REWARD_FUNCTIONS,
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
