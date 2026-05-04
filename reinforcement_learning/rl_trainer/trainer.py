# Direct policy-gradient trainer with group-relative advantages.

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch

from rl_trainer.advantages import group_relative_advantages
from rl_trainer.callbacks import PrintCallback, TrainerCallback
from rl_trainer.config import GRPOAlgorithmConfig, ReinforceAlgorithmConfig, RLTrainerConfig
from rl_trainer.data import iter_batches, make_prompt_batch
from rl_trainer.generation import TransformersRolloutEngine
from rl_trainer.logprobs import policy_logprobs
from rl_trainer.losses import grpo_loss, policy_gradient_loss
from rl_trainer.optim import build_linear_scheduler, build_optimizer
from rl_trainer.rewards import score_rewards
from rl_trainer.types import (
    CompletionRecord,
    GRPOLossInput,
    LossInput,
    OptimizerFactory,
    PromptBatch,
    RewardBatch,
    RewardFunction,
    RewardResult,
    RolloutBatch,
    RolloutEngine,
    RolloutSyncStats,
    SchedulerFactory,
    StepMetrics,
    StepTimings,
    TrainingExample,
    TrainerState,
)


@dataclass(frozen=True)
class _MicrobatchResult:
    loss: float
    metrics: StepMetrics
    timings: StepTimings
    grad_norm: torch.Tensor | None = None
    grpo_clip_ratio: float | None = None


class RLTrainer:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        reward_functions: list[RewardFunction],
        config: RLTrainerConfig,
        rollout_engine: RolloutEngine | None = None,
        optimizer_factory: OptimizerFactory | None = None,
        scheduler_factory: SchedulerFactory | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.reward_functions = reward_functions
        self.config = config
        if config.backward_microbatch_size is not None and config.backward_microbatch_size <= 0:
            raise ValueError("backward_microbatch_size must be positive when set.")
        self._validate_algorithm_config()
        self.optimizer_updates_per_step = self._compute_optimizer_updates_per_step()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.rollout_engine = rollout_engine or TransformersRolloutEngine(model, tokenizer, config, self.device)
        self.optimizer = (
            optimizer_factory(self.model.parameters())
            if optimizer_factory is not None
            else build_optimizer(self.model, config.optimizer)
        )
        self.scheduler = (
            scheduler_factory(self.optimizer)
            if scheduler_factory is not None
            else build_linear_scheduler(
                self.optimizer,
                warmup_ratio=config.warmup_ratio,
                max_steps=config.max_steps * self.optimizer_updates_per_step,
            )
        )
        self.callbacks = callbacks or [PrintCallback()]
        self.state = TrainerState()
        self.minibatch_generator = torch.Generator().manual_seed(config.seed)

    def _validate_algorithm_config(self) -> None:
        algorithm = self.config.algorithm
        if isinstance(algorithm, GRPOAlgorithmConfig):
            if algorithm.epsilon < 0.0:
                raise ValueError("GRPO epsilon must be non-negative.")
            if algorithm.epsilon_high is not None and algorithm.epsilon_high < 0.0:
                raise ValueError("GRPO epsilon_high must be non-negative when set.")
            if algorithm.num_iterations <= 0:
                raise ValueError("GRPO num_iterations must be positive.")
            if algorithm.mini_batch_size is not None and algorithm.mini_batch_size <= 0:
                raise ValueError("GRPO mini_batch_size must be positive when set.")
            if self.config.gradient_accumulation_steps != 1:
                raise ValueError("GRPO currently expects gradient_accumulation_steps=1.")

    def _compute_optimizer_updates_per_step(self) -> int:
        algorithm = self.config.algorithm
        if not isinstance(algorithm, GRPOAlgorithmConfig):
            return 1

        rollout_size = self.config.batch_size * self.config.num_generations
        mini_batch_size = algorithm.mini_batch_size or rollout_size
        return ceil(rollout_size / mini_batch_size) * algorithm.num_iterations

    def train(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        batches = iter_batches(
            self.train_dataset,
            self.config.batch_size,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
        )
        self.optimizer.zero_grad(set_to_none=True)

        try:
            self._train_loop(batches)
        finally:
            self._close_callbacks()

    def _train_loop(self, batches: Iterator[list[TrainingExample]]) -> None:
        while self.state.step < self.config.max_steps:
            if isinstance(self.config.algorithm, GRPOAlgorithmConfig):
                result = self._train_grpo_batch(next(batches))
                self._emit_step_metrics(
                    accumulated_loss=result.loss,
                    latest_metrics=result.metrics,
                    grad_norm=result.grad_norm if result.grad_norm is not None else torch.tensor(0.0),
                    timings=result.timings,
                    loss_divisor=1.0,
                    grpo_clip_ratio=result.grpo_clip_ratio,
                )

                if self.config.save_steps > 0 and self.state.step % self.config.save_steps == 0:
                    self.save_checkpoint(self.config.output_dir / f"checkpoint-{self.state.step}")
                continue

            accumulated_loss = 0.0
            latest_metrics: StepMetrics | None = None
            accumulated_timings = StepTimings(
                rollout_seconds=0.0,
                reward_seconds=0.0,
                backward_seconds=0.0,
                optimizer_seconds=0.0,
                microbatch_seconds=0.0,
                old_logprobs_seconds=0.0,
            )

            for _ in range(self.config.gradient_accumulation_steps):
                result = self._train_microbatch(next(batches))
                accumulated_loss += result.loss
                latest_metrics = result.metrics
                accumulated_timings = self._add_timings(accumulated_timings, result.timings)

            optimizer_start = perf_counter()
            grad_norm = self._optimizer_step()
            optimizer_seconds = perf_counter() - optimizer_start
            accumulated_timings = StepTimings(
                rollout_seconds=accumulated_timings.rollout_seconds,
                reward_seconds=accumulated_timings.reward_seconds,
                backward_seconds=accumulated_timings.backward_seconds,
                optimizer_seconds=optimizer_seconds,
                microbatch_seconds=accumulated_timings.microbatch_seconds,
            )
            self._emit_step_metrics(
                accumulated_loss=accumulated_loss,
                latest_metrics=latest_metrics,
                grad_norm=grad_norm,
                timings=self._average_timings(accumulated_timings),
                loss_divisor=float(self.config.gradient_accumulation_steps),
            )

            if self.config.save_steps > 0 and self.state.step % self.config.save_steps == 0:
                self.save_checkpoint(self.config.output_dir / f"checkpoint-{self.state.step}")

    def _train_microbatch(self, examples: list[TrainingExample]) -> _MicrobatchResult:
        if not isinstance(self.config.algorithm, ReinforceAlgorithmConfig):
            raise TypeError("_train_microbatch is only used for REINFORCE training.")

        microbatch_start = perf_counter()
        prompt_batch = make_prompt_batch(examples)

        rollout_start = perf_counter()
        rollout = self.rollout_engine.generate(prompt_batch)
        rollout_seconds = perf_counter() - rollout_start
        if self.config.empty_cache_steps is not None:
            self._empty_cuda_cache()

        reward_start = perf_counter()
        reward_result = asyncio.run(
            self._score(
                prompt_batch,
                rollout.completions,
                rollout.completion_ids,
                rollout.completion_mask,
            )
        )
        reward_seconds = perf_counter() - reward_start
        advantages = group_relative_advantages(reward_result.total, self.config.num_generations)

        backward_start = perf_counter()
        self.model.train()
        loss_mask = self._loss_mask(rollout.completion_ids, rollout.completion_mask)
        loss_normalizer = loss_mask.sum().clamp(min=1.0)
        loss = self._backward_rollout_chunks(
            rollout=rollout,
            advantages=advantages.advantages.detach(),
            loss_mask=loss_mask,
            loss_normalizer=loss_normalizer,
        )
        backward_seconds = perf_counter() - backward_start

        if self._should_log():
            self._log_completions(prompt_batch, rollout.completions, reward_result.total, advantages.advantages)

        self.state.examples_seen += len(prompt_batch.examples)
        return _MicrobatchResult(
            loss=loss,
            metrics=self._metrics(
                loss=loss,
                reward_result=reward_result,
                completion_mask=rollout.completion_mask,
                loss_mask=loss_mask,
            ),
            timings=StepTimings(
                rollout_seconds=rollout_seconds,
                reward_seconds=reward_seconds,
                backward_seconds=backward_seconds,
                optimizer_seconds=0.0,
                microbatch_seconds=perf_counter() - microbatch_start,
            ),
        )

    def _train_grpo_batch(self, examples: list[TrainingExample]) -> _MicrobatchResult:
        algorithm = self.config.algorithm
        if not isinstance(algorithm, GRPOAlgorithmConfig):
            raise TypeError("_train_grpo_batch is only used for GRPO training.")

        microbatch_start = perf_counter()
        prompt_batch = make_prompt_batch(examples)

        rollout_start = perf_counter()
        rollout = self.rollout_engine.generate(prompt_batch)
        rollout_seconds = perf_counter() - rollout_start
        if self.config.empty_cache_steps is not None:
            self._empty_cuda_cache()

        reward_start = perf_counter()
        reward_result = asyncio.run(
            self._score(
                prompt_batch,
                rollout.completions,
                rollout.completion_ids,
                rollout.completion_mask,
            )
        )
        reward_seconds = perf_counter() - reward_start
        advantages = group_relative_advantages(reward_result.total, self.config.num_generations)

        old_logprobs_start = perf_counter()
        old_logprobs = self._rollout_logprobs(rollout)
        old_logprobs_seconds = perf_counter() - old_logprobs_start
        loss_mask = self._loss_mask(rollout.completion_ids, rollout.completion_mask)
        backward_seconds = 0.0

        if self._should_log():
            self._log_completions(prompt_batch, rollout.completions, reward_result.total, advantages.advantages)

        total_loss = 0.0
        total_clip_ratio = 0.0
        total_grad_norm = 0.0
        optimizer_update_count = 0
        optimizer_seconds = 0.0
        trainable_count = rollout.completion_ids.shape[0]
        mini_batch_size = algorithm.mini_batch_size or trainable_count
        epsilon_high = algorithm.epsilon if algorithm.epsilon_high is None else algorithm.epsilon_high

        self.model.train()
        for _ in range(algorithm.num_iterations):
            indices = torch.randperm(trainable_count, generator=self.minibatch_generator).to(self.device)
            for start in range(0, trainable_count, mini_batch_size):
                stop = min(start + mini_batch_size, trainable_count)
                mini_batch_indices = indices[start:stop]
                self.optimizer.zero_grad(set_to_none=True)

                backward_start = perf_counter()
                loss, clip_ratio = self._backward_grpo_minibatch(
                    rollout=rollout,
                    old_logprobs=old_logprobs,
                    advantages=advantages.advantages.detach(),
                    loss_mask=loss_mask,
                    indices=mini_batch_indices,
                    epsilon=algorithm.epsilon,
                    epsilon_high=epsilon_high,
                )
                backward_seconds += perf_counter() - backward_start

                optimizer_start = perf_counter()
                grad_norm = self._optimizer_update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_seconds += perf_counter() - optimizer_start

                total_loss += loss
                total_clip_ratio += clip_ratio
                total_grad_norm += float(grad_norm.detach().cpu())
                optimizer_update_count += 1

        self.state.step += 1
        if self.config.empty_cache_steps is not None and self.state.step % self.config.empty_cache_steps == 0:
            self._empty_cuda_cache()

        sync_start = perf_counter()
        self._sync_rollout_engine()
        optimizer_seconds += perf_counter() - sync_start

        update_count = max(1, optimizer_update_count)
        self.state.examples_seen += len(prompt_batch.examples)
        average_loss = total_loss / update_count
        average_clip_ratio = total_clip_ratio / update_count
        average_grad_norm = torch.tensor(total_grad_norm / update_count, device=self.device)
        return _MicrobatchResult(
            loss=average_loss,
            metrics=self._metrics(
                loss=average_loss,
                reward_result=reward_result,
                completion_mask=rollout.completion_mask,
                loss_mask=loss_mask,
            ),
            timings=StepTimings(
                rollout_seconds=rollout_seconds,
                reward_seconds=reward_seconds,
                backward_seconds=backward_seconds,
                optimizer_seconds=optimizer_seconds,
                microbatch_seconds=perf_counter() - microbatch_start,
                old_logprobs_seconds=old_logprobs_seconds,
            ),
            grad_norm=average_grad_norm,
            grpo_clip_ratio=average_clip_ratio,
        )

    def _backward_rollout_chunks(
        self,
        *,
        rollout: RolloutBatch,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_normalizer: torch.Tensor,
    ) -> float:
        chunk_size = self.config.backward_microbatch_size or rollout.completion_ids.shape[0]
        accumulated_loss = 0.0
        for start in range(0, rollout.completion_ids.shape[0], chunk_size):
            stop = min(start + chunk_size, rollout.completion_ids.shape[0])
            current_logprobs = policy_logprobs(
                self.model,
                rollout.prompt_ids[start:stop],
                rollout.prompt_attention_mask[start:stop],
                rollout.completion_ids[start:stop],
                rollout.completion_mask[start:stop],
                self.config.temperature,
            )
            self._assert_finite_tensor("current logprobs", current_logprobs)
            loss_output = policy_gradient_loss(
                LossInput(
                    current_logprobs=current_logprobs,
                    advantages=advantages[start:stop],
                    completion_mask=loss_mask[start:stop],
                    normalizer=loss_normalizer,
                )
            )
            self._assert_finite_tensor("loss", loss_output.loss)
            scaled_loss = loss_output.loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += float(loss_output.loss.detach().cpu())

        return accumulated_loss

    def _backward_grpo_minibatch(
        self,
        *,
        rollout: RolloutBatch,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        indices: torch.Tensor,
        epsilon: float,
        epsilon_high: float,
    ) -> tuple[float, float]:
        chunk_size = self.config.backward_microbatch_size or indices.shape[0]
        loss_normalizer = loss_mask[indices].sum().clamp(min=1.0)
        accumulated_loss = 0.0
        clip_numerator = 0.0
        clip_denominator = 0.0
        for start in range(0, indices.shape[0], chunk_size):
            stop = min(start + chunk_size, indices.shape[0])
            chunk_indices = indices[start:stop]
            current_logprobs = policy_logprobs(
                self.model,
                rollout.prompt_ids[chunk_indices],
                rollout.prompt_attention_mask[chunk_indices],
                rollout.completion_ids[chunk_indices],
                rollout.completion_mask[chunk_indices],
                self.config.temperature,
            )
            self._assert_finite_tensor("current logprobs", current_logprobs)
            chunk_loss_mask = loss_mask[chunk_indices]
            loss_output = grpo_loss(
                GRPOLossInput(
                    current_logprobs=current_logprobs,
                    old_logprobs=old_logprobs[chunk_indices],
                    advantages=advantages[chunk_indices],
                    completion_mask=chunk_loss_mask,
                    normalizer=loss_normalizer,
                    epsilon=epsilon,
                    epsilon_high=epsilon_high,
                )
            )
            self._assert_finite_tensor("loss", loss_output.loss)
            loss_output.loss.backward()
            accumulated_loss += float(loss_output.loss.detach().cpu())

            active_tokens = float(chunk_loss_mask.sum().detach().cpu())
            clip_numerator += float(loss_output.clip_ratio.detach().cpu()) * active_tokens
            clip_denominator += active_tokens

        clip_ratio = clip_numerator / max(1.0, clip_denominator)
        return accumulated_loss, clip_ratio

    def _rollout_logprobs(self, rollout: RolloutBatch) -> torch.Tensor:
        chunk_size = self.config.backward_microbatch_size or rollout.completion_ids.shape[0]
        was_training = self.model.training
        self.model.eval()
        chunks: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                for start in range(0, rollout.completion_ids.shape[0], chunk_size):
                    stop = min(start + chunk_size, rollout.completion_ids.shape[0])
                    logprobs = policy_logprobs(
                        self.model,
                        rollout.prompt_ids[start:stop],
                        rollout.prompt_attention_mask[start:stop],
                        rollout.completion_ids[start:stop],
                        rollout.completion_mask[start:stop],
                        self.config.temperature,
                    )
                    self._assert_finite_tensor("old logprobs", logprobs)
                    chunks.append(logprobs.detach())
        finally:
            if was_training:
                self.model.train()

        return torch.cat(chunks, dim=0)

    def _optimizer_step(self) -> torch.Tensor:
        grad_norm = self._optimizer_update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1
        if self.config.empty_cache_steps is not None and self.state.step % self.config.empty_cache_steps == 0:
            self._empty_cuda_cache()
        self._sync_rollout_engine()
        return grad_norm

    def _optimizer_update(self) -> torch.Tensor:
        self._assert_finite_trainable_parameters("before optimizer step")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self._assert_finite_trainable_parameters("after optimizer step")
        return grad_norm

    def _emit_step_metrics(
        self,
        *,
        accumulated_loss: float,
        latest_metrics: StepMetrics | None,
        grad_norm: torch.Tensor,
        timings: StepTimings,
        loss_divisor: float,
        grpo_clip_ratio: float | None = None,
    ) -> None:
        if latest_metrics is None or not self._should_log():
            return

        metrics = StepMetrics(
            step=self.state.step,
            loss=accumulated_loss / loss_divisor,
            reward_mean=latest_metrics.reward_mean,
            reward_std=latest_metrics.reward_std,
            completion_length_mean=latest_metrics.completion_length_mean,
            active_completion_length_mean=latest_metrics.active_completion_length_mean,
            loss_sequence_fraction=latest_metrics.loss_sequence_fraction,
            learning_rate=self._learning_rate(),
            grad_norm=float(grad_norm.detach().cpu()),
            grad_clip_scale=self._grad_clip_scale(grad_norm),
            reward_function_means=latest_metrics.reward_function_means,
            rollout_sync_stats=self._rollout_sync_stats(),
            timings=timings,
            grpo_clip_ratio=grpo_clip_ratio,
        )
        for callback in self.callbacks:
            callback.on_step_end(metrics)

    def _add_timings(self, left: StepTimings, right: StepTimings) -> StepTimings:
        return StepTimings(
            rollout_seconds=left.rollout_seconds + right.rollout_seconds,
            reward_seconds=left.reward_seconds + right.reward_seconds,
            backward_seconds=left.backward_seconds + right.backward_seconds,
            optimizer_seconds=left.optimizer_seconds + right.optimizer_seconds,
            microbatch_seconds=left.microbatch_seconds + right.microbatch_seconds,
            old_logprobs_seconds=left.old_logprobs_seconds + right.old_logprobs_seconds,
        )

    def _average_timings(self, timings: StepTimings) -> StepTimings:
        microbatch_count = max(1, self.config.gradient_accumulation_steps)
        return StepTimings(
            rollout_seconds=timings.rollout_seconds / microbatch_count,
            reward_seconds=timings.reward_seconds / microbatch_count,
            backward_seconds=timings.backward_seconds / microbatch_count,
            optimizer_seconds=timings.optimizer_seconds,
            microbatch_seconds=timings.microbatch_seconds / microbatch_count,
            old_logprobs_seconds=timings.old_logprobs_seconds / microbatch_count,
        )

    async def _score(
        self,
        prompt_batch: PromptBatch,
        completions: list[list[dict[str, str]]],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> RewardResult:
        repeated_prompts = [
            prompt
            for prompt in prompt_batch.prompts
            for _ in range(self.config.num_generations)
        ]
        extra_fields = {
            key: [example.fields[key] for example in prompt_batch.examples for _ in range(self.config.num_generations)]
            for key in prompt_batch.examples[0].fields
        }
        reward_batch = RewardBatch(
            prompts=repeated_prompts,
            completions=completions,
            completion_ids=completion_ids.detach().cpu().tolist(),
            completion_mask=completion_mask.detach().cpu().tolist(),
            extra_fields=extra_fields,
            trainer_state=self.state,
        )
        return await score_rewards(self.reward_functions, reward_batch, self.device)

    def _loss_mask(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
        if not self.config.mask_truncated_completions:
            return completion_mask

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", eos_token_id)
        if eos_token_id is None or pad_token_id is None:
            return completion_mask

        terminal_ids = torch.tensor([eos_token_id, pad_token_id], device=completion_ids.device)
        last_ids = completion_ids[:, -1]
        terminated = last_ids.unsqueeze(1).eq(terminal_ids).any(dim=1)
        return completion_mask * terminated.unsqueeze(1).to(completion_mask.dtype)

    def _metrics(
        self,
        *,
        loss: float,
        reward_result: RewardResult,
        completion_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> StepMetrics:
        reward_function_means = {
            name: float(torch.nanmean(reward_result.per_function[:, index]).detach().cpu())
            for index, name in enumerate(reward_result.names)
        }
        lengths = completion_mask.sum(dim=1)
        active_lengths = loss_mask.sum(dim=1)
        return StepMetrics(
            step=self.state.step,
            loss=loss,
            reward_mean=float(reward_result.total.mean().detach().cpu()),
            reward_std=float(reward_result.total.std().detach().cpu()),
            completion_length_mean=float(lengths.mean().detach().cpu()),
            active_completion_length_mean=float(active_lengths.mean().detach().cpu()),
            loss_sequence_fraction=float(active_lengths.gt(0).to(torch.float32).mean().detach().cpu()),
            learning_rate=self._learning_rate(),
            grad_norm=0.0,
            grad_clip_scale=1.0,
            reward_function_means=reward_function_means,
        )

    def _log_completions(
        self,
        prompt_batch: PromptBatch,
        completions: list[list[dict[str, str]]],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                **self.config.chat_template_kwargs,
            )
            for prompt in prompt_batch.prompts
            for _ in range(self.config.num_generations)
        ]
        records = [
            CompletionRecord(
                prompt=prompt,
                completion=completion[0]["content"],
                reward=float(reward.detach().cpu()),
                advantages=float(advantage.detach().cpu()),
            )
            for prompt, completion, reward, advantage in zip(prompt_texts, completions, rewards, advantages, strict=True)
        ]
        for callback in self.callbacks:
            callback.on_completions(records)

    def _should_log(self) -> bool:
        return self.config.logging_steps > 0 and self.state.step % self.config.logging_steps == 0

    def _learning_rate(self) -> float:
        return float(cast(float, self.scheduler.get_last_lr()[0]))

    def _grad_clip_scale(self, grad_norm: torch.Tensor) -> float:
        raw_grad_norm = float(grad_norm.detach().cpu())
        return min(1.0, self.config.max_grad_norm / (raw_grad_norm + 1e-6))

    def _sync_rollout_engine(self) -> None:
        sync_after_optimizer_step = getattr(self.rollout_engine, "sync_after_optimizer_step", None)
        if sync_after_optimizer_step is None:
            return
        sync_after_optimizer_step(model=self.model, tokenizer=self.tokenizer, step=self.state.step)

    def _rollout_sync_stats(self) -> RolloutSyncStats | None:
        stats = getattr(self.rollout_engine, "last_sync_stats", None)
        if isinstance(stats, RolloutSyncStats):
            return stats
        return None

    def _close_callbacks(self) -> None:
        for callback in self.callbacks:
            close_callback = getattr(callback, "close", None)
            if callable(close_callback):
                close_callback()

    def _assert_finite_trainable_parameters(self, location: str) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Non-finite trainable parameter `{name}` detected {location}.")

    def _assert_finite_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite {name} detected.")

    def _empty_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
