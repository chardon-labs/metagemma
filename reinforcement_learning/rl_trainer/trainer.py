# Algorithmic reference: reinforcement_learning/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py

import asyncio
import copy
from pathlib import Path
from typing import Any, cast

import torch

from rl_trainer.advantages import group_relative_advantages
from rl_trainer.callbacks import PrintCallback, TrainerCallback
from rl_trainer.config import RLTrainerConfig
from rl_trainer.data import iter_batches, make_prompt_batch
from rl_trainer.generation import TransformersRolloutEngine
from rl_trainer.logprobs import policy_logprobs
from rl_trainer.losses import clipped_policy_loss
from rl_trainer.optim import build_adamw, build_linear_scheduler
from rl_trainer.rewards import score_rewards
from rl_trainer.types import (
    CompletionRecord,
    EnvironmentFactory,
    LossInput,
    OptimizerFactory,
    PromptBatch,
    RewardBatch,
    RewardFunction,
    RewardResult,
    RolloutEngine,
    SchedulerFactory,
    StepMetrics,
    TrainingExample,
    TrainerState,
)


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
        environment_factory: EnvironmentFactory | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.reward_functions = reward_functions
        self.config = config
        self.environment_factory = environment_factory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.rollout_engine = rollout_engine or TransformersRolloutEngine(model, tokenizer, config, self.device)
        self.optimizer = (
            optimizer_factory(self.model.parameters())
            if optimizer_factory is not None
            else build_adamw(self.model, config)
        )
        self.scheduler = (
            scheduler_factory(self.optimizer)
            if scheduler_factory is not None
            else build_linear_scheduler(self.optimizer, warmup_ratio=config.warmup_ratio, max_steps=config.max_steps)
        )
        self.callbacks = callbacks or [PrintCallback()]
        self.state = TrainerState()

    def train(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        batches = iter_batches(
            self.train_dataset,
            self.config.batch_size,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
        )
        self.optimizer.zero_grad(set_to_none=True)

        while self.state.step < self.config.max_steps:
            accumulated_loss = 0.0
            latest_metrics: StepMetrics | None = None

            for _ in range(self.config.gradient_accumulation_steps):
                prompt_batch = asyncio.run(self._prepare_prompt_batch(next(batches)))
                rollout = self.rollout_engine.generate(prompt_batch)
                if self.config.empty_cache_steps is not None:
                    self._empty_cuda_cache()
                reward_result = asyncio.run(self._score(prompt_batch, rollout.completions, rollout.completion_ids))
                advantages = group_relative_advantages(reward_result.total, self.config.num_generations)

                self.model.train()
                current_logprobs = policy_logprobs(
                    self.model,
                    rollout.prompt_ids,
                    rollout.prompt_attention_mask,
                    rollout.completion_ids,
                    rollout.completion_mask,
                    self.config.temperature,
                )
                self._assert_finite_tensor("current logprobs", current_logprobs)
                completion_mask = self._loss_mask(rollout.completion_ids, rollout.completion_mask)
                loss_output = clipped_policy_loss(
                    LossInput(
                        current_logprobs=current_logprobs,
                        old_logprobs=current_logprobs.detach(),
                        advantages=advantages.advantages.detach(),
                        completion_mask=completion_mask,
                        epsilon=self.config.epsilon,
                        epsilon_high=self.config.effective_epsilon_high,
                        delta=self.config.delta,
                        loss_type=self.config.loss_type,
                    )
                )
                self._assert_finite_tensor("loss", loss_output.loss)
                scaled_loss = loss_output.loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                accumulated_loss += float(loss_output.loss.detach().cpu())

                latest_metrics = self._metrics(
                    loss=float(loss_output.loss.detach().cpu()),
                    reward_result=reward_result,
                    advantages=advantages.advantages,
                    completion_mask=rollout.completion_mask,
                    loss_mask=completion_mask,
                    mean_ratio=loss_output.mean_ratio,
                    clip_ratio=loss_output.clip_ratio,
                )
                if self._should_log():
                    self._log_completions(prompt_batch, rollout.completions, reward_result.total, advantages.advantages)

                self.state.examples_seen += len(prompt_batch.examples)

            self._assert_finite_trainable_parameters("before optimizer step")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            self._assert_finite_trainable_parameters("after optimizer step")
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.step += 1
            if self.config.empty_cache_steps is not None and self.state.step % self.config.empty_cache_steps == 0:
                self._empty_cuda_cache()
            self._sync_rollout_engine()

            if latest_metrics is not None and self._should_log():
                latest_metrics = StepMetrics(
                    step=self.state.step,
                    loss=accumulated_loss / self.config.gradient_accumulation_steps,
                    reward_mean=latest_metrics.reward_mean,
                    reward_std=latest_metrics.reward_std,
                    completion_length_mean=latest_metrics.completion_length_mean,
                    active_completion_length_mean=latest_metrics.active_completion_length_mean,
                    loss_sequence_fraction=latest_metrics.loss_sequence_fraction,
                    learning_rate=self._learning_rate(),
                    grad_norm=float(grad_norm.detach().cpu()),
                    mean_ratio=latest_metrics.mean_ratio,
                    clip_ratio=latest_metrics.clip_ratio,
                    reward_function_means=latest_metrics.reward_function_means,
                )
                for callback in self.callbacks:
                    callback.on_step_end(latest_metrics)

            if self.config.save_steps > 0 and self.state.step % self.config.save_steps == 0:
                self.save_checkpoint(self.config.output_dir / f"checkpoint-{self.state.step}")

    async def _score(
        self,
        prompt_batch: PromptBatch,
        completions: list[list[dict[str, str]]],
        completion_ids: torch.Tensor,
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
            extra_fields=extra_fields,
            trainer_state=self.state,
        )
        return await score_rewards(self.reward_functions, reward_batch, self.device)

    async def _prepare_prompt_batch(self, examples: list[TrainingExample]) -> PromptBatch:
        if self.environment_factory is None:
            return make_prompt_batch(examples)

        environments = [self.environment_factory() for _ in examples]
        observations = await asyncio.gather(
            *(environment.reset(example) for environment, example in zip(environments, examples, strict=True))
        )
        updated_examples = []
        for example, observation in zip(examples, observations, strict=True):
            if observation is None:
                updated_examples.append(example)
                continue

            prompt = copy.deepcopy(example.prompt)
            if not prompt:
                raise ValueError("Environment observations require non-empty prompts.")

            content = prompt[-1].get("content")
            if isinstance(content, str):
                prompt[-1]["content"] = content + observation
            elif isinstance(content, list):
                content_blocks = cast(list[dict[str, object]], content)
                content_blocks.append({"type": "text", "text": observation})
            else:
                raise TypeError("Prompt content must be either text or content blocks.")

            updated_examples.append(TrainingExample(prompt=prompt, fields=example.fields))

        return make_prompt_batch(updated_examples)

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
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        mean_ratio: float,
        clip_ratio: float,
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
            mean_ratio=mean_ratio,
            clip_ratio=clip_ratio,
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
            self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
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

    def _sync_rollout_engine(self) -> None:
        sync_after_optimizer_step = getattr(self.rollout_engine, "sync_after_optimizer_step", None)
        if sync_after_optimizer_step is None:
            return
        sync_after_optimizer_step(model=self.model, tokenizer=self.tokenizer, step=self.state.step)

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
