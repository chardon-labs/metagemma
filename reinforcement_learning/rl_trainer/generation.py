import os
from importlib import import_module
from typing import Any, cast

import torch

from rl_trainer.config import RLTrainerConfig
from rl_trainer.tensors import completion_mask
from rl_trainer.types import Completion, PromptBatch, RolloutBatch, TokenBatch


class TransformersRolloutEngine:
    def __init__(self, model: Any, tokenizer: Any, config: RLTrainerConfig, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pad_token_id = self._pad_token_id()
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def generate(self, batch: PromptBatch) -> RolloutBatch:
        token_batch = self._tokenize(batch)
        repeated_prompt_ids = token_batch.input_ids.repeat_interleave(self.config.num_generations, dim=0)
        repeated_attention_mask = token_batch.attention_mask.repeat_interleave(self.config.num_generations, dim=0)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=repeated_prompt_ids,
                attention_mask=repeated_attention_mask,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_new_tokens=self.config.max_completion_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=self.config.use_generation_cache,
                disable_compile=self.config.disable_generation_compile,
            )

        sequences = outputs.sequences.clone()
        completion_ids = sequences[:, repeated_prompt_ids.shape[1] :].clone()
        mask = completion_mask(completion_ids, self.eos_token_id, self.pad_token_id).clone()
        old_logprobs = torch.zeros_like(completion_ids, dtype=torch.float32)
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        completions: list[Completion] = [[{"role": "assistant", "content": text}] for text in completion_texts]
        del outputs

        return RolloutBatch(
            prompt_ids=repeated_prompt_ids,
            prompt_attention_mask=repeated_attention_mask,
            completion_ids=completion_ids,
            completion_mask=mask,
            old_logprobs=old_logprobs,
            completions=completions,
        )

    def _tokenize(self, batch: PromptBatch) -> TokenBatch:
        prompt_texts = [
            self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in batch.prompts
        ]
        tokens = self.tokenizer(
            text=prompt_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        return TokenBatch(
            input_ids=tokens["input_ids"].to(self.device),
            attention_mask=tokens["attention_mask"].to(self.device),
            prompt_texts=prompt_texts,
        )

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        return int(eos_token_id)


class VLLMRolloutEngine:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer: Any,
        config: RLTrainerConfig,
        device: torch.device,
        gpu_memory_utilization: float,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        enforce_eager: bool = True,
        sync_steps: int = 0,
        sync_chunk_bytes: int = 8 * 1024 * 1024 * 1024,
        sync_backend: str = "inprocess",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pad_token_id = self._pad_token_id()
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.sync_steps = sync_steps
        self.sync_chunk_bytes = sync_chunk_bytes
        self.sync_backend = sync_backend
        if self.sync_backend != "inprocess":
            raise ValueError(f"Unsupported vLLM sync backend: {self.sync_backend}")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        vllm = import_module("vllm")
        self.llm_cls = getattr(vllm, "LLM")
        self.sampling_params_cls = getattr(vllm, "SamplingParams")
        self.llm = self._build_llm(model_name_or_path)

    def generate(self, batch: PromptBatch) -> RolloutBatch:
        prompt_texts = self._prompt_texts(batch)
        repeated_prompt_texts = [
            prompt_text
            for prompt_text in prompt_texts
            for _ in range(self.config.num_generations)
        ]
        prompt_token_lists = [
            self._encode_prompt(prompt_text)
            for prompt_text in repeated_prompt_texts
        ]
        prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_lists]
        outputs = self.llm.generate(
            prompts,
            self._sampling_params(),
            use_tqdm=False,
        )

        completion_token_lists: list[list[int]] = []
        completion_texts: list[str] = []
        for output in outputs:
            candidate = output.outputs[0]
            completion_token_lists.append([int(token_id) for token_id in candidate.token_ids])
            completion_texts.append(str(candidate.text))

        prompt_ids, prompt_attention_mask = self._left_pad_prompts(prompt_token_lists)
        completion_ids = self._right_pad_completions(completion_token_lists)
        mask = completion_mask(completion_ids, self.eos_token_id, self.pad_token_id).clone()
        old_logprobs = torch.zeros_like(completion_ids, dtype=torch.float32)
        completions: list[Completion] = [[{"role": "assistant", "content": text}] for text in completion_texts]

        return RolloutBatch(
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            completion_ids=completion_ids,
            completion_mask=mask,
            old_logprobs=old_logprobs,
            completions=completions,
        )

    def generate_completions(
        self,
        prompts: list[list[dict[str, object]] | list[dict[str, str | list[dict[str, str]]]]],
        *,
        count: int,
    ) -> list[str]:
        prompt_texts = [
            self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in prompts
            for _ in range(count)
        ]
        prompt_token_lists = [self._encode_prompt(prompt_text) for prompt_text in prompt_texts]
        vllm_prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_lists]
        outputs = self.llm.generate(vllm_prompts, self._sampling_params(), use_tqdm=False)
        return [str(output.outputs[0].text) for output in outputs]

    def sync_after_optimizer_step(self, *, model: Any, tokenizer: Any, step: int) -> None:
        del tokenizer
        if self.sync_steps <= 0:
            return
        if step % self.sync_steps != 0:
            return

        print(f"syncing_vllm_rollout_model step={step} backend={self.sync_backend}", flush=True)
        torch.cuda.synchronize()
        synced_tensors = 0
        synced_bytes = 0
        loaded_tensors = 0
        for batch in self._iter_sync_batches(model):
            synced_tensors += len(batch)
            synced_bytes += sum(weight.numel() * weight.element_size() for _, weight in batch)
            loaded_tensors += self._update_vllm_weights_inprocess(batch)
        self.llm.reset_prefix_cache()
        print(
            "synced_vllm_rollout_model "
            f"sent_tensors={synced_tensors} loaded_tensors={loaded_tensors} bytes={synced_bytes}",
            flush=True,
        )

    def _build_llm(self, model_name_or_path: str) -> Any:
        return self.llm_cls(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
            max_model_len=self.config.max_completion_length * 4,
            enforce_eager=self.enforce_eager,
        )

    def _iter_sync_batches(self, model: Any) -> list[list[tuple[str, torch.Tensor]]]:
        batches: list[list[tuple[str, torch.Tensor]]] = []
        current_batch: list[tuple[str, torch.Tensor]] = []
        current_bytes = 0

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or not self._should_sync_parameter(name, parameter):
                continue
            weight = parameter.detach()
            if weight.device.type != "cuda":
                weight = weight.to(self.device, non_blocking=True)
            if not weight.is_contiguous():
                weight = weight.contiguous()

            weight_bytes = weight.numel() * weight.element_size()
            if current_batch and current_bytes + weight_bytes > self.sync_chunk_bytes:
                batches.append(current_batch)
                current_batch = []
                current_bytes = 0

            current_batch.append((name, weight))
            current_bytes += weight_bytes

        if current_batch:
            batches.append(current_batch)

        return batches

    def _should_sync_parameter(self, name: str, parameter: torch.nn.Parameter) -> bool:
        if parameter.device.type == "meta":
            return False
        if any(marker in name for marker in ("vision_tower", "audio_tower", "embed_vision", "embed_audio")):
            return False
        text_parameter_prefixes = (
            "model.layers.",
            "model.embed_tokens.",
            "model.norm.",
            "lm_head.",
        )
        return "language_model" in name or name.startswith(text_parameter_prefixes)

    def _update_vllm_weights_inprocess(self, weights: list[tuple[str, torch.Tensor]]) -> int:
        loaded_counts = self.llm.apply_model(lambda vllm_model: len(vllm_model.load_weights(weights)))
        torch.cuda.synchronize()
        return sum(int(count) for count in loaded_counts)

    def _prompt_texts(self, batch: PromptBatch) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in batch.prompts
        ]

    def _encode_prompt(self, prompt_text: str) -> list[int]:
        tokens = self.tokenizer(
            text=prompt_text,
            add_special_tokens=False,
            return_tensors=None,
        )
        input_ids = tokens["input_ids"]
        if not isinstance(input_ids, list):
            raise TypeError("Tokenizer must return list input ids for vLLM prompts.")
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return [int(token_id) for token_id in cast(list[int], input_ids)]

    def _left_pad_prompts(self, prompt_token_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = max(len(tokens) for tokens in prompt_token_lists)
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for tokens in prompt_token_lists:
            pad_length = max_length - len(tokens)
            ids.append([self.pad_token_id] * pad_length + tokens)
            masks.append([0] * pad_length + [1] * len(tokens))

        return (
            torch.tensor(ids, dtype=torch.long, device=self.device),
            torch.tensor(masks, dtype=torch.long, device=self.device),
        )

    def _right_pad_completions(self, completion_token_lists: list[list[int]]) -> torch.Tensor:
        max_length = max(1, max(len(tokens) for tokens in completion_token_lists))
        ids = [
            tokens + [self.pad_token_id] * (max_length - len(tokens))
            for tokens in completion_token_lists
        ]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _sampling_params(self) -> Any:
        return self.sampling_params_cls(
            n=1,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k if self.config.top_k > 0 else -1,
            max_tokens=self.config.max_completion_length,
            skip_special_tokens=True,
        )

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        return int(eos_token_id)
