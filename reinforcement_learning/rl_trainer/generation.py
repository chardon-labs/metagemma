from typing import Any

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
