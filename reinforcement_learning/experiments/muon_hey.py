from pathlib import Path

from experiments.utils.support import HeyDataset, build_vllm_engine, load_model_and_tokenizer, short_completion_reward
from rl_trainer import JSONLLogCallback, MuonOptimizerConfig, PrintCallback, ReinforceAlgorithmConfig, RLTrainer, RLTrainerConfig

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 2048
MAX_COMPLETION_LENGTH = 256
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
FULL_FINETUNING = True

PROMPT_TEXT = "hey"
ENABLE_THINKING = True
DATASET_SIZE = 1000
MAX_STEPS = 100
OUTPUT_DIR = Path("outputs/muon_hey")
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
        logging_steps=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=128,
        backward_microbatch_size=8,
        max_seq_length=MAX_SEQ_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
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
        chat_template_kwargs={"enable_thinking": ENABLE_THINKING},
    )


def print_training_config(config: RLTrainerConfig) -> None:
    print(
        "muon_hey_config "
        f"generations={config.num_generations} lr={LEARNING_RATE:.2e} "
        "muon_adjust_lr=match_rms_adamw "
        f"backward_microbatch={config.backward_microbatch_size} "
        f"temperature={config.temperature:.2f} max_seq={config.max_seq_length} "
        f"max_completion={config.max_completion_length} "
        f"thinking={ENABLE_THINKING} mask_truncated={config.mask_truncated_completions} "
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
        quiet=True,
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

    dataset = HeyDataset(size=DATASET_SIZE, prompt_text=PROMPT_TEXT)
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_functions=[short_completion_reward],
        config=config,
        rollout_engine=rollout_engine,
        callbacks=[PrintCallback(reward_y_limits=None), JSONLLogCallback(OUTPUT_DIR / "logs")],
    )
    trainer.train()

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()
