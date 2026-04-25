from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

from datasets import Dataset, load_dataset
from math_verify import parse, verify


MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "outputs/qwen3-0.5b-gsm8k-grpo"
MAX_TRAIN_SAMPLES = 2048
MAX_EVAL_SAMPLES = 128
MAX_COMPLETION_LENGTH = 512
NUM_GENERATIONS = 16
GENERATION_BATCH_SIZE = 16
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-6
WARMUP_STEPS = 8
LR_SCHEDULER_TYPE = "cosine"
MAX_STEPS = 250
NUM_TRAIN_EPOCHS = 1.0
LOGGING_STEPS = 5
SAVE_STEPS = 50
EVAL_STEPS: int | None = None
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
ENABLE_THINKING = True
REPETITION_PENALTY = 1.0
SEED = 42
BF16 = True
FP16 = False
USE_PEFT = True
USE_VLLM = True
VLLM_GPU_MEMORY_UTILIZATION = 0.6
PUSH_TO_HUB = False
HUB_MODEL_ID: str | None = None
REPORT_TO = "wandb"
WANDB_PROJECT = "unicorn-mafia-grpo"
WANDB_ENTITY: str | None = None
WANDB_RUN_NAME: str | None = None
WANDB_LOG_MODEL = "false"
WANDB_WATCH = "false"
LOG_REWARD_TRACKING = True
LOG_REWARD_TABLES = True
REWARD_TABLE_LOG_EVERY = 1
DISABLE_TQDM = True
LOG_WANDB_LEET_HINT = True

SYSTEM_PROMPT = (
    "Solve the following math problem. Give the final answer in the format: "
    "#### <answer>"
)

Completion: TypeAlias = str | Sequence[Mapping[str, Any]]
ChatMessage: TypeAlias = dict[str, str]
RewardFunc: TypeAlias = Callable[..., list[float | None]]

_GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?")
_reward_group_index = 0
_wandb_import_warning_shown = False

LOGGER = logging.getLogger(__name__)


from peft import LoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer


def configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def report_targets() -> list[str]:
    if isinstance(REPORT_TO, str):
        values = [target.strip().lower() for target in REPORT_TO.split(",")]
    else:
        values = [str(target).strip().lower() for target in REPORT_TO]
    return [target for target in values if target and target != "none"]


def wandb_enabled() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False
    return "wandb" in report_targets()


def configure_wandb_environment() -> None:
    if not wandb_enabled():
        return

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    os.environ.setdefault("WANDB_NAME", WANDB_RUN_NAME or Path(OUTPUT_DIR).name)
    os.environ.setdefault("WANDB_LOG_MODEL", WANDB_LOG_MODEL)
    os.environ.setdefault("WANDB_WATCH", WANDB_WATCH)
    if WANDB_ENTITY is not None:
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)


def run_config() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "output_dir": OUTPUT_DIR,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "num_generations": NUM_GENERATIONS,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "max_steps": MAX_STEPS,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "enable_thinking": ENABLE_THINKING,
        "repetition_penalty": REPETITION_PENALTY,
        "seed": SEED,
        "bf16": BF16,
        "fp16": FP16,
        "use_peft": USE_PEFT,
        "use_vllm": USE_VLLM,
        "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
    }


def initialize_wandb() -> None:
    if not wandb_enabled():
        return

    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb reporting is enabled but wandb is not installed.")
        return

    if wandb.run is None:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY"),
            name=os.environ.get("WANDB_NAME", WANDB_RUN_NAME or Path(OUTPUT_DIR).name),
            config=run_config(),
        )

    wandb.define_metric("reward/question_group")
    wandb.define_metric("reward/*", step_metric="reward/question_group")


def log_wandb_leet_hint() -> None:
    if not LOG_WANDB_LEET_HINT or get_wandb_run() is None:
        return

    LOGGER.info(
        "Monitor this run in a separate terminal with: uv run wandb beta leet run"
    )


def get_wandb_run() -> Any | None:
    global _wandb_import_warning_shown

    if not wandb_enabled():
        return None

    try:
        import wandb
    except ImportError:
        if not _wandb_import_warning_shown:
            LOGGER.warning("wandb reporting is enabled but wandb is not installed.")
            _wandb_import_warning_shown = True
        return None

    return wandb.run


def log_to_wandb(data: Mapping[str, Any], *, step: int | None = None, commit: bool = True) -> None:
    if get_wandb_run() is None:
        return

    import wandb

    wandb.log(dict(data), step=step, commit=commit)


def extract_gsm8k_answer(answer: str) -> str:
    match = _GSM8K_FINAL_RE.search(answer)
    if match is not None:
        return normalize_number(match.group(1))

    numbers = _NUMBER_RE.findall(answer)
    return normalize_number(numbers[-1]) if numbers else answer.strip()


def normalize_number(value: str) -> str:
    return value.replace(",", "").strip()


def format_prompt(question: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def prepare_split(split: str, max_samples: int | None) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def map_example(example: Mapping[str, Any]) -> dict[str, str | list[ChatMessage]]:
        question = str(example["question"])
        answer = str(example["answer"])
        return {
            "prompt": format_prompt(question),
            "answer": extract_gsm8k_answer(answer),
        }

    mapped = dataset.map(map_example, remove_columns=dataset.column_names)
    return cast(Dataset, mapped)


def completion_text(completion: Completion | Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence):
        pieces: list[str] = []
        for message in completion:
            if isinstance(message, Mapping):
                pieces.append(str(message.get("content", "")))
            else:
                pieces.append(str(message))
        return "\n".join(pieces)
    return str(completion)


def extract_prediction(text: str) -> str:
    match = _GSM8K_FINAL_RE.search(text)
    if match is not None:
        return normalize_number(match.group(1))

    parsed = parse(text)
    if parsed:
        return str(parsed[0])

    numbers = _NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else text.strip()


def format_predictions(predictions: Sequence[str], max_items: int = 8) -> str:
    unique_predictions = list(dict.fromkeys(predictions))
    shown = ", ".join(repr(prediction) for prediction in unique_predictions[:max_items])
    remaining = len(unique_predictions) - max_items
    if remaining > 0:
        shown = f"{shown}, ... (+{remaining} more)"
    return f"[{shown}]"


def log_reward_tracking(gold_answers: Sequence[str], predictions: Sequence[str]) -> None:
    global _reward_group_index

    if not LOG_REWARD_TRACKING:
        return

    for start in range(0, len(predictions), NUM_GENERATIONS):
        group_gold = gold_answers[start : start + NUM_GENERATIONS]
        group_predictions = predictions[start : start + NUM_GENERATIONS]
        if not group_gold:
            continue

        _reward_group_index += 1
        expected = normalize_number(group_gold[0])
        correct_count = sum(
            prediction == normalize_number(gold)
            for gold, prediction in zip(group_gold, group_predictions, strict=True)
        )
        group_size = len(group_predictions)
        accuracy = correct_count / group_size if group_size else 0.0
        LOGGER.info(
            "[question %s] extracted_correct=%s/%s expected=%r predictions=%s",
            _reward_group_index,
            correct_count,
            group_size,
            expected,
            format_predictions(group_predictions, max_items=NUM_GENERATIONS),
        )
        log_reward_group(
            question_group=_reward_group_index,
            expected=expected,
            group_gold=group_gold,
            group_predictions=group_predictions,
            correct_count=correct_count,
            group_size=group_size,
            accuracy=accuracy,
        )


def log_reward_group(
    *,
    question_group: int,
    expected: str,
    group_gold: Sequence[str],
    group_predictions: Sequence[str],
    correct_count: int,
    group_size: int,
    accuracy: float,
) -> None:
    payload: dict[str, Any] = {
        "reward/question_group": question_group,
        "reward/exact_match_count": correct_count,
        "reward/exact_match_total": group_size,
        "reward/exact_match_rate": accuracy,
        "reward/expected": expected,
    }

    if LOG_REWARD_TABLES and question_group % REWARD_TABLE_LOG_EVERY == 0 and get_wandb_run() is not None:
        import wandb

        table = wandb.Table(columns=["question_group", "expected", "gold", "prediction", "correct"])
        for gold, prediction in zip(group_gold, group_predictions, strict=True):
            normalized_gold = normalize_number(gold)
            table.add_data(
                question_group,
                expected,
                normalized_gold,
                prediction,
                prediction == normalized_gold,
            )
        payload["reward/predictions"] = table

    log_to_wandb(payload)


def exact_match_reward(completions: Sequence[Completion], answer: Sequence[str], **_: Any) -> list[float | None]:
    rewards: list[float | None] = []
    predictions: list[str] = []
    correct_count = 0
    for completion, gold_answer in zip(completions, answer, strict=True):
        predicted = extract_prediction(completion_text(completion))
        predictions.append(predicted)
        is_correct = predicted == normalize_number(gold_answer)
        if is_correct:
            correct_count += 1
        rewards.append(1.0 if is_correct else 0.0)

    log_reward_tracking(answer, predictions)
    return rewards


def math_verify_reward(completions: Sequence[Completion], answer: Sequence[str], **_: Any) -> list[float | None]:
    rewards: list[float | None] = []
    for completion, gold_answer in zip(completions, answer, strict=True):
        predicted_text = completion_text(completion)
        gold = parse(f"#### {gold_answer}")
        target = parse(predicted_text)
        rewards.append(1.0 if gold and target and verify(gold, target) else 0.0)
    return rewards


def format_reward(completions: Sequence[Completion], **_: Any) -> list[float | None]:
    rewards: list[float | None] = []
    for completion in completions:
        text = completion_text(completion)
        rewards.append(0.2 if _GSM8K_FINAL_RE.search(text) is not None else 0.0)
    return rewards


def build_peft_config(enabled: bool) -> LoraConfig | None:
    if not enabled:
        return None

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def build_trainer() -> GRPOTrainer:
    train_dataset = prepare_split("train", MAX_TRAIN_SAMPLES)
    eval_dataset = prepare_split("test", MAX_EVAL_SAMPLES) if MAX_EVAL_SAMPLES else None

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        generation_batch_size=GENERATION_BATCH_SIZE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps" if EVAL_STEPS is not None else "no",
        eval_steps=EVAL_STEPS,
        bf16=BF16,
        fp16=FP16,
        gradient_checkpointing=True,
        use_vllm=USE_VLLM,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        chat_template_kwargs={"enable_thinking": ENABLE_THINKING},
        repetition_penalty=REPETITION_PENALTY,
        report_to=REPORT_TO,
        seed=SEED,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_ID,
        log_completions=False,
        run_name=os.environ.get("WANDB_NAME", WANDB_RUN_NAME or Path(OUTPUT_DIR).name),
        disable_tqdm=DISABLE_TQDM,
        model_init_kwargs={"dtype": "auto", "trust_remote_code": True},
    )

    return GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=[math_verify_reward, exact_match_reward, format_reward],
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(USE_PEFT),
    )


def main() -> None:
    configure_logging()
    configure_wandb_environment()
    initialize_wandb()
    log_wandb_leet_hint()
    LOGGER.info("Starting GRPO training run: %s", run_config())
    trainer = build_trainer()
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    if get_wandb_run() is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
