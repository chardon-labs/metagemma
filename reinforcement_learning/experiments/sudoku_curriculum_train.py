from pathlib import Path
from typing import Any

import torch
from unsloth import FastVisionModel

from rl_trainer import PrintCallback, RLTrainer, RLTrainerConfig, TrainerCallback
from rl_trainer.generation import VLLMRolloutEngine
from tasks.sudoku import (
    SUDOKU_REWARD_FUNCTIONS,
    CurriculumCallback,
    SudokuCurriculum,
    SudokuDataset,
)

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 8192
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
FULL_FINETUNING = True
DATASET_SIZE = 1000
MAX_STEPS = 60
OUTPUT_DIR = Path("outputs/sudoku_curriculum_train")
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
MAX_COMPLETION_LENGTH = 2048

VLLM_GPU_MEMORY_UTILIZATION = 0.20
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_ENFORCE_EAGER = True
VLLM_SYNC_STEPS = 1
VLLM_SYNC_BACKEND = "inprocess"
VLLM_SYNC_CHUNK_BYTES = 8 * 1024 * 1024 * 1024


def load_model_and_tokenizer() -> tuple[Any, Any]:
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
        full_finetuning=FULL_FINETUNING,
    )
    if FULL_FINETUNING:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return model, tokenizer


def build_training_config() -> RLTrainerConfig:
    return RLTrainerConfig(
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.03,
        logging_steps=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=128,
        backward_microbatch_size=8,
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
    )


def build_vllm_engine(
    model_name_or_path: str,
    tokenizer: Any,
    config: RLTrainerConfig,
    *,
    sync_steps: int = 0,
) -> VLLMRolloutEngine:
    return VLLMRolloutEngine(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        config=config,
        device=torch.device("cuda"),
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        enforce_eager=VLLM_ENFORCE_EAGER,
        sync_steps=sync_steps,
        sync_chunk_bytes=VLLM_SYNC_CHUNK_BYTES,
        sync_backend=VLLM_SYNC_BACKEND,
    )


def print_training_config(config: RLTrainerConfig) -> None:
    print(
        "curriculum_training_config "
        f"generations={config.num_generations} lr={config.learning_rate:.2e} "
        f"backward_microbatch={config.backward_microbatch_size} "
        f"weight_decay={config.weight_decay:.3g} temperature={config.temperature:.2f} "
        f"max_completion={config.max_completion_length} "
        f"mask_truncated={config.mask_truncated_completions} "
        f"vllm_sync_steps={VLLM_SYNC_STEPS}",
        flush=True,
    )


def main() -> None:
    config = build_training_config()
    print_training_config(config)
    model, tokenizer = load_model_and_tokenizer()
    rollout_engine = build_vllm_engine(MODEL_NAME, tokenizer, config, sync_steps=VLLM_SYNC_STEPS)
    curriculum = SudokuCurriculum()
    dataset = SudokuDataset(
        size=DATASET_SIZE,
        curriculum=curriculum,
        seed=RANDOM_STATE,
    )

    print("Dataset sample:")
    print(dataset[0])

    callbacks: list[TrainerCallback] = [PrintCallback(), CurriculumCallback(curriculum)]

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
