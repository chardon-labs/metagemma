from pathlib import Path

import unsloth
from safetensors import safe_open
from transformers import TextStreamer
from unsloth import FastVisionModel

from rl_trainer import PrintCallback, RLTrainer, RLTrainerConfig, TrainerCallback
from tasks.sudoku import (
    SUDOKU_REWARD_FUNCTIONS,
    CurriculumCallback,
    SudokuCurriculum,
    SudokuDataset,
)

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 8192
LORA_RANK = 32
RANDOM_STATE = 3407
LOAD_IN_4BIT = False
FAST_INFERENCE = False
DATASET_SIZE = 1000
MAX_STEPS = 60
OUTPUT_DIR = Path("outputs")
ADAPTER_DIR = Path("gemma_4_lora")
MAX_COMPLETION_LENGTH = 2048


def load_model_and_tokenizer():
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
        weight_decay=0.001,
        warmup_ratio=0.1,
        logging_steps=1,
        batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=8,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=100,
        output_dir=OUTPUT_DIR,
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="bnpo",
        mask_truncated_completions=True,
    )


def run_base_generation(model, tokenizer, dataset: SudokuDataset) -> None:
    sample = dataset[0]
    prompt = sample["prompt"]
    if not isinstance(prompt, list):
        raise TypeError("Expected prompt list.")

    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text=text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    print("=" * 50)
    print("BASE MODEL OUTPUT")
    print("=" * 50)
    model.generate(
        **inputs,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
        max_new_tokens=512,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )


def verify_saved_lora(adapter_dir: Path) -> None:
    adapter_path = adapter_dir / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt") as tensors:
        for key in tensors.keys():
            tensor = tensors.get_tensor(key)
            zero_fraction = (tensor == 0).sum().item() / tensor.numel()
            if zero_fraction == 1.0:
                raise ValueError(f"{key} is all zeros")


def run_trained_generation(model, tokenizer, dataset: SudokuDataset) -> None:
    sample = dataset[0]
    prompt = sample["prompt"]
    if not isinstance(prompt, list):
        raise TypeError("Expected prompt list.")

    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    model.generate(
        **tokenizer(images=None, text=text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=MAX_COMPLETION_LENGTH,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )


def main() -> None:
    model, tokenizer = load_model_and_tokenizer()
    curriculum = SudokuCurriculum()
    dataset = SudokuDataset(
        size=DATASET_SIZE,
        curriculum=curriculum,
        seed=RANDOM_STATE,
    )

    print("Dataset sample:")
    print(dataset[0])

    run_base_generation(model, tokenizer, dataset)
    callbacks: list[TrainerCallback] = [PrintCallback(), CurriculumCallback(curriculum)]

    trainer = RLTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_functions=SUDOKU_REWARD_FUNCTIONS,
        config=build_training_config(),
        callbacks=callbacks,
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    verify_saved_lora(ADAPTER_DIR)
    run_trained_generation(model, tokenizer, dataset)


if __name__ == "__main__":
    main()
