import random
from pathlib import Path

import torch

from experiments.utils.support import (
    SudokuEvalCallback,
    build_vllm_engine,
    evaluate_puzzle,
    load_model_and_tokenizer,
    print_puzzle,
    select_smoke_puzzle,
)
from rl_trainer import JSONLLogCallback, MuonOptimizerConfig, PrintCallback, ReinforceAlgorithmConfig, RLTrainer, RLTrainerConfig
from tasks.sudoku import SUDOKU_REWARD_FUNCTIONS, SinglePuzzleDataset

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 2048
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
FULL_FINETUNING = True

PUZZLE_DIFFICULTY = 0.35
CANDIDATE_COMPLETIONS = 128
MIN_ACCEPTED_SOLVES = 32
MAX_ACCEPTED_SOLVES = 96
MAX_PUZZLE_CANDIDATES = 32
EVAL_COMPLETIONS = 64
PERIODIC_EVAL_STEPS = 10
PERIODIC_EVAL_COMPLETIONS = 128

DATASET_SIZE = 1000
MAX_STEPS = 300
OUTPUT_DIR = Path("outputs/muon_single")
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
LEARNING_RATE = 1e-5

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
        batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=128,
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
        algorithm=ReinforceAlgorithmConfig(),
        temperature=1.0,
        mask_truncated_completions=False,
        max_grad_norm=1.0,
        seed=RANDOM_STATE,
        shuffle=True,
        empty_cache_steps=1,
    )


def print_training_config(config: RLTrainerConfig) -> None:
    print(
        "muon_single_config "
        f"generations={config.num_generations} lr={LEARNING_RATE:.2e} "
        "muon_adjust_lr=match_rms_adamw "
        f"backward_microbatch={config.backward_microbatch_size} "
        f"weight_decay={config.weight_decay:.3g} temperature={config.temperature:.2f} "
        f"max_seq={config.max_seq_length} "
        f"mask_truncated={config.mask_truncated_completions} "
        f"vllm_sync_steps={VLLM_SYNC_STEPS}",
        flush=True,
    )


def main() -> None:
    config = build_training_config()
    print_training_config(config)
    rng = random.Random(RANDOM_STATE)
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

    puzzle = select_smoke_puzzle(
        rollout_engine=rollout_engine,
        rng=rng,
        puzzle_difficulty=PUZZLE_DIFFICULTY,
        candidate_completions=CANDIDATE_COMPLETIONS,
        min_accepted_solves=MIN_ACCEPTED_SOLVES,
        max_accepted_solves=MAX_ACCEPTED_SOLVES,
        max_puzzle_candidates=MAX_PUZZLE_CANDIDATES,
    )
    print_puzzle(puzzle)

    base_solves = evaluate_puzzle(
        rollout_engine=rollout_engine,
        puzzle=puzzle,
        completion_count=EVAL_COMPLETIONS,
    )
    print(f"base_exact_solve_rate={base_solves}/{EVAL_COMPLETIONS}")

    dataset = SinglePuzzleDataset(puzzle=puzzle, size=DATASET_SIZE)
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_functions=SUDOKU_REWARD_FUNCTIONS,
        config=config,
        rollout_engine=rollout_engine,
        callbacks=[
            PrintCallback(),
            JSONLLogCallback(OUTPUT_DIR / "logs"),
            SudokuEvalCallback(
                rollout_engine=rollout_engine,
                puzzle=puzzle,
                periodic_eval_steps=PERIODIC_EVAL_STEPS,
                periodic_eval_completions=PERIODIC_EVAL_COMPLETIONS,
            ),
        ],
    )
    trainer.train()

    trained_solves = evaluate_puzzle(
        rollout_engine=rollout_engine,
        puzzle=puzzle,
        completion_count=EVAL_COMPLETIONS,
    )
    print(f"trained_exact_solve_rate={trained_solves}/{EVAL_COMPLETIONS}")

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    del trainer
    del rollout_engine
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
