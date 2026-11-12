"""Sampling completions from a model, with vLLM on the GPU or transformers elsewhere."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from rl_transitions.config import ModelConfig, SamplingConfig

Messages = list[dict[str, str]]


@dataclass
class Completion:
    text: str
    num_tokens: int
    finish_reason: str  # "stop" (ended on its own) or "length" (hit max_tokens)


class Sampler(Protocol):
    def sample(self, prompts: list[Messages], sampling: SamplingConfig) -> list[list[Completion]]:
        """Return `sampling.n` completions for each prompt, in prompt order."""
        ...


def render_chat(tokenizer: Any, messages: Messages, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


class VLLMSampler:
    def __init__(self, model: ModelConfig, seed: int = 0):
        from vllm import LLM

        self.model = model
        self.llm = LLM(
            model=model.name,
            dtype="bfloat16",
            seed=seed,
            gpu_memory_utilization=model.gpu_memory_utilization,
            max_model_len=model.max_model_len,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def sample(self, prompts: list[Messages], sampling: SamplingConfig) -> list[list[Completion]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=sampling.n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            **({"top_k": sampling.top_k} if sampling.top_k > 0 else {}),
        )
        texts = [render_chat(self.tokenizer, m, self.model.enable_thinking) for m in prompts]
        outputs = self.llm.generate(texts, params)
        return [
            [Completion(c.text, len(c.token_ids), c.finish_reason or "stop") for c in out.outputs]
            for out in outputs
        ]


class HFSampler:
    """Plain transformers generation: slow, but runs on CPU for tests and local checks."""

    def __init__(self, model: ModelConfig, batch_size: int = 8):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_cfg = model
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float32
        else:
            self.device, dtype = "cpu", torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model.name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model.name, dtype=dtype).to(self.device)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}

    @torch.no_grad()
    def sample(self, prompts: list[Messages], sampling: SamplingConfig) -> list[list[Completion]]:
        torch.manual_seed(sampling.seed)
        texts = [render_chat(self.tokenizer, m, self.model_cfg.enable_thinking) for m in prompts]
        greedy = sampling.temperature == 0
        results: list[list[Completion]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(self.device)
            out = self.model.generate(
                **enc,
                do_sample=not greedy,
                temperature=None if greedy else sampling.temperature,
                top_p=None if greedy else sampling.top_p,
                top_k=None if greedy else sampling.top_k,
                max_new_tokens=sampling.max_tokens,
                num_return_sequences=sampling.n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated = out[:, enc["input_ids"].shape[1] :].tolist()
            for i in range(len(batch)):
                rows = generated[i * sampling.n : (i + 1) * sampling.n]
                results.append([self._completion(tokens) for tokens in rows])
        return results

    def _completion(self, tokens: list[int]) -> Completion:
        end = next((j for j, t in enumerate(tokens) if t in self.eos_ids), None)
        kept = tokens if end is None else tokens[:end]
        text = self.tokenizer.decode(kept, skip_special_tokens=True)
        return Completion(text, len(kept), "length" if end is None else "stop")


def make_sampler(model: ModelConfig, seed: int = 0) -> Sampler:
    backend = model.backend
    if backend == "auto":
        has_vllm = importlib.util.find_spec("vllm") is not None
        backend = "vllm" if has_vllm and torch.cuda.is_available() else "hf"
    if backend == "vllm":
        return VLLMSampler(model, seed=seed)
    if backend == "hf":
        return HFSampler(model)
    raise ValueError(f"unknown backend {model.backend!r}; use 'vllm', 'hf' or 'auto'")
