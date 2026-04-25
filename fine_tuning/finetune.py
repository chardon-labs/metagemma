from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from confidence_trace import (
    CONFIDENCE_TOKEN_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TRACE_DIR,
    configure_logging,
    load_manifest,
    read_trace_metadata,
    verify_confidence_token,
)


LOGGER = logging.getLogger(__name__)

MODEL_ID = DEFAULT_MODEL_ID
TOKENIZER_ID: str | None = DEFAULT_TOKENIZER_ID
TRACE_DIR = Path(DEFAULT_TRACE_DIR)
OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR)
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
CONFIDENCE_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
EPOCHS = 1.0
MAX_STEPS = 0
WARMUP_STEPS = 0
KL_WEIGHT = 1.0
BCE_WEIGHT = 1.0
SEED = 42
LOG_EVERY_STEPS = 10
EVAL_EVERY_STEPS = 100
MAX_GRAD_NORM = 1.0
BF16 = True
FP16 = False
USE_PEFT = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
REPORT_TO = "wandb"
WANDB_PROJECT = "unicorn-mafia-grpo"
WANDB_RUN_NAME: str | None = None
SAVE_FULL_MODEL = False
BALANCE_TRAIN_BATCHES = True


@dataclass(frozen=True)
class FinetuneConfig:
    model_id: str
    tokenizer_id: str | None = TOKENIZER_ID
    trace_dir: Path = TRACE_DIR
    output_dir: Path = OUTPUT_DIR
    train_batch_size: int = TRAIN_BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = LEARNING_RATE
    confidence_learning_rate: float = CONFIDENCE_LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    epochs: float = EPOCHS
    max_steps: int = MAX_STEPS
    warmup_steps: int = WARMUP_STEPS
    kl_weight: float = KL_WEIGHT
    bce_weight: float = BCE_WEIGHT
    seed: int = SEED
    log_every_steps: int = LOG_EVERY_STEPS
    eval_every_steps: int = EVAL_EVERY_STEPS
    max_grad_norm: float = MAX_GRAD_NORM
    bf16: bool = BF16
    fp16: bool = FP16
    use_peft: bool = USE_PEFT
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    report_to: str = REPORT_TO
    wandb_project: str = WANDB_PROJECT
    wandb_run_name: str | None = WANDB_RUN_NAME
    save_full_model: bool = SAVE_FULL_MODEL
    balance_train_batches: bool = BALANCE_TRAIN_BATCHES


@dataclass(frozen=True)
class ConfidenceMetricBatch:
    final_probs: torch.Tensor
    final_labels: torch.Tensor
    tail_probs: torch.Tensor
    tail_labels: torch.Tensor


def config_dict(config: FinetuneConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["trace_dir"] = str(config.trace_dir)
    payload["output_dir"] = str(config.output_dir)
    return payload


class TraceDataset(Dataset[dict[str, Any]]):
    def __init__(self, *, trace_dir: Path, split: str) -> None:
        self.trace_dir = trace_dir
        self.split = split
        self.samples: list[dict[str, Any]] = []
        manifest = load_manifest(trace_dir)

        for shard in manifest.shards:
            meta_path = trace_dir / shard["meta_path"]
            arrays_path = trace_dir / shard["arrays_path"]
            rows = read_trace_metadata(meta_path)
            arrays = np.load(arrays_path)
            prompt_token_ids = arrays["prompt_token_ids"]
            completion_token_ids = arrays["completion_token_ids"]
            top_logprob_token_ids = arrays["top_logprob_token_ids"]
            top_logprobs = arrays["top_logprobs"]
            top_logprob_mask = arrays["top_logprob_mask"]

            for row in rows:
                if row["split"] != split or int(row["token_length"]) <= 0:
                    continue

                prompt_start = int(row["prompt_token_start"])
                prompt_end = prompt_start + int(row["prompt_token_length"])
                token_start = int(row["token_start"])
                token_end = token_start + int(row["token_length"])
                self.samples.append(
                    {
                        "row_id": int(row["row_id"]),
                        "problem_id": int(row["problem_id"]),
                        "math_verify_label": float(row["math_verify_label"]),
                        "prompt_token_ids": prompt_token_ids[prompt_start:prompt_end].copy(),
                        "completion_token_ids": completion_token_ids[token_start:token_end].copy(),
                        "top_logprob_token_ids": top_logprob_token_ids[token_start:token_end].copy(),
                        "top_logprobs": top_logprobs[token_start:token_end].copy(),
                        "top_logprob_mask": top_logprob_mask[token_start:token_end].copy(),
                    }
                )

        LOGGER.info("Loaded %s %s samples from %s", len(self.samples), split, trace_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def label_summary(self) -> dict[str, float]:
        total = len(self.samples)
        correct = sum(int(sample["math_verify_label"]) for sample in self.samples)
        incorrect = total - correct
        correct_rate = correct / total if total else float("nan")
        majority_accuracy = max(correct, incorrect) / total if total else float("nan")
        return {
            "samples": float(total),
            "correct": float(correct),
            "incorrect": float(incorrect),
            "correct_rate": correct_rate,
            "majority_class_accuracy": majority_accuracy,
        }


class BalancedBinaryBatchSampler(BatchSampler):
    def __init__(
        self,
        *,
        labels: list[int],
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 2:
            raise ValueError("BalancedBinaryBatchSampler requires batch_size >= 2.")

        positives = [index for index, label in enumerate(labels) if label == 1]
        negatives = [index for index, label in enumerate(labels) if label == 0]
        if not positives or not negatives:
            raise ValueError("BalancedBinaryBatchSampler requires both positive and negative samples.")

        super().__init__(
            RandomSampler(range(len(labels))),
            batch_size=batch_size,
            drop_last=drop_last,
        )
        self.positives = positives
        self.negatives = negatives
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self) -> Any:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        positives = self.positives.copy()
        negatives = self.negatives.copy()
        rng.shuffle(positives)
        rng.shuffle(negatives)

        positive_cursor = 0
        negative_cursor = 0
        batches: list[list[int]] = []
        positive_target = self.batch_size // 2
        negative_target = self.batch_size - positive_target

        while positive_cursor < len(positives) or negative_cursor < len(negatives):
            batch: list[int] = []
            take_positive = min(positive_target, len(positives) - positive_cursor)
            take_negative = min(negative_target, len(negatives) - negative_cursor)
            batch.extend(positives[positive_cursor : positive_cursor + take_positive])
            batch.extend(negatives[negative_cursor : negative_cursor + take_negative])
            positive_cursor += take_positive
            negative_cursor += take_negative

            while len(batch) < self.batch_size and positive_cursor < len(positives):
                batch.append(positives[positive_cursor])
                positive_cursor += 1
            while len(batch) < self.batch_size and negative_cursor < len(negatives):
                batch.append(negatives[negative_cursor])
                negative_cursor += 1

            if len(batch) == self.batch_size or (batch and not self.drop_last):
                rng.shuffle(batch)
                batches.append(batch)

        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        sample_count = len(self.positives) + len(self.negatives)
        if self.drop_last:
            return sample_count // self.batch_size
        return math.ceil(sample_count / self.batch_size)


def dtype_from_config(config: FinetuneConfig) -> torch.dtype | str:
    if config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if config.fp16 and torch.cuda.is_available():
        return torch.float16
    return "auto"


def build_peft_model(model: nn.Module, config: FinetuneConfig) -> nn.Module:
    if not config.use_peft:
        return model

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


def collate_trace_batch(samples: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    max_input_len = max(len(sample["prompt_token_ids"]) + len(sample["completion_token_ids"]) for sample in samples)
    max_target_len = max(len(sample["completion_token_ids"]) for sample in samples)
    logprobs_k = int(samples[0]["top_logprob_token_ids"].shape[1])

    input_ids = torch.full((len(samples), max_input_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(samples), max_input_len), dtype=torch.long)
    target_positions = torch.zeros((len(samples), max_target_len), dtype=torch.long)
    target_mask = torch.zeros((len(samples), max_target_len), dtype=torch.bool)
    teacher_token_ids = torch.full((len(samples), max_target_len, logprobs_k), -1, dtype=torch.long)
    teacher_logprobs = torch.full((len(samples), max_target_len, logprobs_k), -torch.inf, dtype=torch.float32)
    teacher_mask = torch.zeros((len(samples), max_target_len, logprobs_k), dtype=torch.bool)
    labels = torch.zeros((len(samples), max_target_len), dtype=torch.float32)

    for batch_index, sample in enumerate(samples):
        prompt_ids = sample["prompt_token_ids"].astype(np.int64, copy=False)
        completion_ids = sample["completion_token_ids"].astype(np.int64, copy=False)
        sequence = np.concatenate([prompt_ids, completion_ids])
        prompt_len = len(prompt_ids)
        target_len = len(completion_ids)
        if prompt_len == 0:
            raise ValueError("Prompt token length cannot be zero.")

        input_ids[batch_index, : len(sequence)] = torch.from_numpy(sequence)
        attention_mask[batch_index, : len(sequence)] = 1
        positions = torch.arange(prompt_len - 1, prompt_len + target_len - 1, dtype=torch.long)
        target_positions[batch_index, :target_len] = positions
        target_mask[batch_index, :target_len] = True
        teacher_token_ids[batch_index, :target_len] = torch.from_numpy(
            sample["top_logprob_token_ids"].astype(np.int64, copy=False)
        )
        teacher_logprobs[batch_index, :target_len] = torch.from_numpy(
            sample["top_logprobs"].astype(np.float32, copy=False)
        )
        teacher_mask[batch_index, :target_len] = torch.from_numpy(sample["top_logprob_mask"])
        labels[batch_index, :target_len] = float(sample["math_verify_label"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_positions": target_positions,
        "target_mask": target_mask,
        "teacher_token_ids": teacher_token_ids,
        "teacher_logprobs": teacher_logprobs,
        "teacher_mask": teacher_mask,
        "labels": labels,
    }


def selected_token_state(
    tensor: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    batch_indices = torch.arange(tensor.shape[0], device=tensor.device).unsqueeze(1)
    return tensor[batch_indices, target_positions]


def binary_classification_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if probs.numel() == 0:
        return {
            "acc": float("nan"),
            "bal_acc": float("nan"),
            "auroc": float("nan"),
            "brier": float("nan"),
        }

    labels_bool = labels >= 0.5
    predictions = probs >= 0.5
    accuracy = (predictions == labels_bool).float().mean()
    brier = torch.square(probs - labels).mean()

    positives = labels_bool
    negatives = ~labels_bool
    has_positive = positives.any()
    has_negative = negatives.any()
    if has_positive:
        true_positive_rate = predictions[positives].float().mean()
    else:
        true_positive_rate = torch.tensor(float("nan"), device=probs.device)
    if has_negative:
        true_negative_rate = (~predictions[negatives]).float().mean()
    else:
        true_negative_rate = torch.tensor(float("nan"), device=probs.device)
    if has_positive and has_negative:
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
        order = torch.argsort(probs)
        ranks = torch.empty_like(probs, dtype=torch.float32)
        ranks[order] = torch.arange(1, probs.numel() + 1, device=probs.device, dtype=torch.float32)
        positive_rank_sum = ranks[positives].sum()
        positive_count = positives.float().sum()
        negative_count = negatives.float().sum()
        auroc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    else:
        balanced_accuracy = torch.tensor(float("nan"), device=probs.device)
        auroc = torch.tensor(float("nan"), device=probs.device)

    return {
        "acc": float(accuracy.detach().cpu()),
        "bal_acc": float(balanced_accuracy.detach().cpu()),
        "auroc": float(auroc.detach().cpu()),
        "brier": float(brier.detach().cpu()),
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.scalar_sums: dict[str, float] = {}
        self.scalar_count = 0
        self.final_probs: list[torch.Tensor] = []
        self.final_labels: list[torch.Tensor] = []
        self.tail_probs: list[torch.Tensor] = []
        self.tail_labels: list[torch.Tensor] = []

    def update(self, metrics: dict[str, float], confidence_metrics: ConfidenceMetricBatch) -> None:
        for key, value in metrics.items():
            if not math.isnan(value):
                self.scalar_sums[key] = self.scalar_sums.get(key, 0.0) + value
        self.scalar_count += 1
        if confidence_metrics.final_probs.numel() > 0:
            self.final_probs.append(confidence_metrics.final_probs)
            self.final_labels.append(confidence_metrics.final_labels)
        if confidence_metrics.tail_probs.numel() > 0:
            self.tail_probs.append(confidence_metrics.tail_probs)
            self.tail_labels.append(confidence_metrics.tail_labels)

    def summary(self) -> dict[str, float]:
        metrics = {
            key: value / max(1, self.scalar_count)
            for key, value in self.scalar_sums.items()
        }
        if self.final_probs:
            final_probs = torch.cat(self.final_probs)
            final_labels = torch.cat(self.final_labels)
            final_metrics = binary_classification_metrics(final_probs, final_labels)
            metrics.update(
                {
                    "conf_final": float(final_probs.mean()),
                    "acc_final": final_metrics["acc"],
                    "bal_acc_final": final_metrics["bal_acc"],
                    "auroc_final": final_metrics["auroc"],
                    "brier_final": final_metrics["brier"],
                }
            )
        if self.tail_probs:
            tail_probs = torch.cat(self.tail_probs)
            tail_labels = torch.cat(self.tail_labels)
            tail_metrics = binary_classification_metrics(tail_probs, tail_labels)
            metrics.update(
                {
                    "acc_tail10": tail_metrics["acc"],
                    "bal_acc_tail10": tail_metrics["bal_acc"],
                    "auroc_tail10": tail_metrics["auroc"],
                    "brier_tail10": tail_metrics["brier"],
                }
            )
        return metrics

    def reset(self) -> None:
        self.scalar_sums = {}
        self.scalar_count = 0
        self.final_probs = []
        self.final_labels = []
        self.tail_probs = []
        self.tail_labels = []


def compute_loss(
    *,
    model: nn.Module,
    confidence_row: nn.Parameter,
    batch: dict[str, torch.Tensor],
    kl_weight: float,
    bce_weight: float,
) -> tuple[torch.Tensor, dict[str, float], ConfidenceMetricBatch]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    target_positions = batch["target_positions"]
    target_mask = batch["target_mask"]
    teacher_token_ids = batch["teacher_token_ids"]
    teacher_logprobs = batch["teacher_logprobs"]
    teacher_mask = batch["teacher_mask"]
    labels = batch["labels"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    logits = selected_token_state(outputs.logits, target_positions).float()
    hidden = selected_token_state(outputs.hidden_states[-1], target_positions).float()
    confidence_logits = torch.matmul(hidden, confidence_row.float())

    valid_teacher_tokens = teacher_mask.any(dim=-1)
    kl_token_mask = target_mask & valid_teacher_tokens
    safe_teacher_ids = teacher_token_ids.clamp_min(0)

    kl_logits = logits.clone()
    kl_logits[..., CONFIDENCE_TOKEN_ID] = torch.finfo(kl_logits.dtype).min
    student_logprobs = F.log_softmax(kl_logits, dim=-1)
    student_teacher_logprobs = torch.gather(student_logprobs, dim=-1, index=safe_teacher_ids)

    teacher_logprobs = teacher_logprobs.masked_fill(~teacher_mask, -torch.inf)
    teacher_log_norm = torch.logsumexp(teacher_logprobs, dim=-1, keepdim=True)
    teacher_logq = teacher_logprobs - teacher_log_norm
    teacher_q = torch.exp(teacher_logq).masked_fill(~teacher_mask, 0.0)

    token_ce = -(teacher_q * student_teacher_logprobs.masked_fill(~teacher_mask, 0.0)).sum(dim=-1)
    token_entropy = -(teacher_q * teacher_logq.masked_fill(~teacher_mask, 0.0)).sum(dim=-1)
    token_kl = token_ce - token_entropy
    if kl_token_mask.any():
        kl_loss = token_kl[kl_token_mask].mean()
    else:
        kl_loss = token_kl.sum() * 0.0

    bce_per_token = F.binary_cross_entropy_with_logits(confidence_logits, labels, reduction="none")
    bce_loss = bce_per_token[target_mask].mean()
    loss = kl_weight * kl_loss + bce_weight * bce_loss

    with torch.no_grad():
        confidence_probs = torch.sigmoid(confidence_logits)
        final_confidences = []
        tail_confidences = []
        final_labels = []
        tail_label_values = []
        for row_index in range(target_mask.shape[0]):
            valid_indices = torch.nonzero(target_mask[row_index], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            final_index = valid_indices[-1]
            final_confidences.append(confidence_probs[row_index, final_index])
            final_labels.append(labels[row_index, final_index])
            tail_len = max(1, math.ceil(valid_indices.numel() * 0.1))
            tail_indices = valid_indices[-tail_len:]
            tail_confidences.append(confidence_probs[row_index, tail_indices])
            tail_label_values.append(labels[row_index, tail_indices])
        if final_confidences:
            final_confidence_tensor = torch.stack(final_confidences)
            final_label_tensor = torch.stack(final_labels)
            final_confidence_mean = final_confidence_tensor.mean()
            tail_confidence_tensor = torch.cat(tail_confidences)
            tail_label_tensor = torch.cat(tail_label_values)
        else:
            final_confidence_tensor = torch.empty((0,), device=logits.device)
            final_label_tensor = torch.empty((0,), device=logits.device)
            final_confidence_mean = torch.tensor(float("nan"), device=logits.device)
            tail_confidence_tensor = torch.empty((0,), device=logits.device)
            tail_label_tensor = torch.empty((0,), device=logits.device)

        metrics = {
            "loss": float(loss.detach().cpu()),
            "kl": float(kl_loss.detach().cpu()),
            "bce": float(bce_loss.detach().cpu()),
            "conf_final": float(final_confidence_mean.detach().cpu()),
        }
        confidence_metric_batch = ConfidenceMetricBatch(
            final_probs=final_confidence_tensor.detach().cpu(),
            final_labels=final_label_tensor.detach().cpu(),
            tail_probs=tail_confidence_tensor.detach().cpu(),
            tail_labels=tail_label_tensor.detach().cpu(),
        )

    return loss, metrics, confidence_metric_batch


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def initialize_wandb(config: FinetuneConfig) -> Any | None:
    if config.report_to != "wandb" or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None

    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb reporting requested but wandb is not installed.")
        return None

    os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    if config.wandb_run_name is not None:
        os.environ.setdefault("WANDB_NAME", config.wandb_run_name)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", config.wandb_project),
        name=os.environ.get("WANDB_NAME"),
        config=config_dict(config),
    )
    LOGGER.info("Monitor this run in a separate terminal with: uv run wandb beta leet run")
    return run


def log_metrics(wandb_run: Any | None, metrics: dict[str, float], *, step: int, prefix: str) -> None:
    payload = {f"{prefix}/{key}": value for key, value in metrics.items()}
    LOGGER.info("%s step=%s %s", prefix, step, payload)
    if wandb_run is not None:
        import wandb

        wandb.log(payload, step=step)


def log_dataset_summary(wandb_run: Any | None, dataset: TraceDataset, *, prefix: str) -> None:
    summary = dataset.label_summary()
    payload = {f"{prefix}/{key}": value for key, value in summary.items()}
    LOGGER.info("%s label summary: %s", prefix, payload)
    if wandb_run is not None:
        import wandb

        wandb.log(payload, step=0)


def train_dataloader(
    *,
    dataset: TraceDataset,
    tokenizer: PreTrainedTokenizerBase,
    config: FinetuneConfig,
) -> DataLoader[dict[str, torch.Tensor]]:
    collate_fn = lambda samples: collate_trace_batch(samples, pad_token_id=int(tokenizer.pad_token_id))
    if config.balance_train_batches and config.train_batch_size >= 2:
        labels = [int(sample["math_verify_label"]) for sample in dataset.samples]
        if 0 in labels and 1 in labels:
            return DataLoader(
                dataset,
                batch_sampler=BalancedBinaryBatchSampler(
                    labels=labels,
                    batch_size=config.train_batch_size,
                    seed=config.seed,
                ),
                collate_fn=collate_fn,
            )
        LOGGER.warning("Requested balanced batches, but train split does not contain both classes.")

    return DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    confidence_row: nn.Parameter,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    config: FinetuneConfig,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    for batch in dataloader:
        loss, metrics, confidence_metrics = compute_loss(
            model=model,
            confidence_row=confidence_row,
            batch=to_device(batch, device),
            kl_weight=config.kl_weight,
            bce_weight=config.bce_weight,
        )
        del loss
        accumulator.update(metrics, confidence_metrics)
    model.train()
    return accumulator.summary()


def optimizer_groups(model: nn.Module, confidence_row: nn.Parameter, config: FinetuneConfig) -> list[dict[str, Any]]:
    model_params = [param for param in model.parameters() if param.requires_grad]
    return [
        {
            "params": model_params,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "params": [confidence_row],
            "lr": config.confidence_learning_rate,
            "weight_decay": 0.0,
        },
    ]


def save_outputs(
    *,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    confidence_row: nn.Parameter,
    config: FinetuneConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_any = cast(Any, model)
    lm_head = model_any.get_output_embeddings()
    with torch.no_grad():
        lm_head.weight[CONFIDENCE_TOKEN_ID].copy_(confidence_row.to(lm_head.weight.device, dtype=lm_head.weight.dtype))

    torch.save(confidence_row.detach().cpu(), config.output_dir / "confidence_lm_head_row.pt")
    (config.output_dir / "confidence_sft_config.json").write_text(
        json.dumps(config_dict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tokenizer.save_pretrained(config.output_dir)

    if config.save_full_model and hasattr(model_any, "merge_and_unload"):
        merged = model_any.merge_and_unload()
        lm_head = merged.get_output_embeddings()
        with torch.no_grad():
            lm_head.weight[CONFIDENCE_TOKEN_ID].copy_(confidence_row.to(lm_head.weight.device, dtype=lm_head.weight.dtype))
        merged.save_pretrained(config.output_dir)
    elif not config.use_peft or config.save_full_model:
        model_any.save_pretrained(config.output_dir)
    else:
        model_any.save_pretrained(config.output_dir)
        LOGGER.info(
            "Saved PEFT adapter plus confidence_lm_head_row.pt. Copy that row into lm_head.weight[%s] when loading.",
            CONFIDENCE_TOKEN_ID,
        )


def train() -> None:
    configure_logging()
    config = FinetuneConfig(model_id=MODEL_ID)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    tokenizer_id = config.tokenizer_id or config.model_id
    wandb_run = initialize_wandb(config)

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True),
    )
    verify_confidence_token(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TraceDataset(trace_dir=config.trace_dir, split="sft")
    eval_dataset = TraceDataset(trace_dir=config.trace_dir, split="eval")
    log_dataset_summary(wandb_run, train_dataset, prefix="train_baseline")
    if len(eval_dataset) > 0:
        log_dataset_summary(wandb_run, eval_dataset, prefix="eval_baseline")

    train_loader = train_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        config=config,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=lambda samples: collate_trace_batch(samples, pad_token_id=int(tokenizer.pad_token_id)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        torch_dtype=dtype_from_config(config),
    )
    model = build_peft_model(model, config)
    model.to(device)
    model.train()

    model_any = cast(Any, model)
    lm_head = model_any.get_output_embeddings()
    input_embeddings = model_any.get_input_embeddings()
    tied = lm_head.weight.data_ptr() == input_embeddings.weight.data_ptr()
    LOGGER.info("lm_head/input embeddings tied: %s", tied)
    with torch.no_grad():
        lm_head.weight[CONFIDENCE_TOKEN_ID].zero_()

    confidence_row = nn.Parameter(
        torch.zeros(
            lm_head.weight.shape[1],
            device=device,
            dtype=lm_head.weight.dtype,
        )
    )
    optimizer = torch.optim.AdamW(optimizer_groups(model, confidence_row, config))

    total_update_steps = (
        config.max_steps
        if config.max_steps > 0
        else math.ceil(len(train_loader) * config.epochs / config.gradient_accumulation_steps)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, config.warmup_steps)) if config.warmup_steps else 1.0,
    )

    global_step = 0
    running = MetricAccumulator()
    running_count = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_update_steps:
        for batch_index, batch in enumerate(train_loader):
            del batch_index
            loss, metrics, confidence_metrics = compute_loss(
                model=model,
                confidence_row=confidence_row,
                batch=to_device(batch, device),
                kl_weight=config.kl_weight,
                bce_weight=config.bce_weight,
            )
            (loss / config.gradient_accumulation_steps).backward()
            running.update(metrics, confidence_metrics)
            running_count += 1

            if running_count % config.gradient_accumulation_steps != 0:
                continue

            trainable_params = [param for group in optimizer.param_groups for param in group["params"]]
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % config.log_every_steps == 0:
                averaged = running.summary()
                averaged["learning_rate"] = float(scheduler.get_last_lr()[0])
                averaged["confidence_learning_rate"] = float(scheduler.get_last_lr()[1])
                log_metrics(wandb_run, averaged, step=global_step, prefix="train")
                running.reset()
                running_count = 0

            if len(eval_dataset) > 0 and config.eval_every_steps > 0 and global_step % config.eval_every_steps == 0:
                eval_metrics = evaluate(
                    model=model,
                    confidence_row=confidence_row,
                    dataloader=eval_loader,
                    device=device,
                    config=config,
                )
                log_metrics(wandb_run, eval_metrics, step=global_step, prefix="eval")

            if global_step >= total_update_steps:
                break

    if running_count:
        averaged = running.summary()
        log_metrics(wandb_run, averaged, step=global_step, prefix="train")

    if len(eval_dataset) > 0:
        eval_metrics = evaluate(
            model=model,
            confidence_row=confidence_row,
            dataloader=eval_loader,
            device=device,
            config=config,
        )
        log_metrics(wandb_run, eval_metrics, step=global_step, prefix="eval")

    save_outputs(
        model=model,
        tokenizer=tokenizer,
        confidence_row=confidence_row,
        config=config,
    )

    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    train()
