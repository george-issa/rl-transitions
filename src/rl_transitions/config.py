"""Typed run configs, loaded from the YAML files in configs/."""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class SplitConfig:
    size: int
    seed: int
    task_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    task: str
    out_dir: str
    # Splits are built in file order; each one excludes questions used by earlier ones.
    splits: dict[str, SplitConfig]
    prompt_style: str = "single"
    loophole_seed: int = 0


@dataclass
class ModelConfig:
    name: str
    backend: str = "auto"  # "vllm", "hf" or "auto" (vLLM when a GPU and vLLM are present)
    enable_thinking: bool = False
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 2048


@dataclass
class SamplingConfig:
    # Defaults match TRL's GRPO rollouts, so evaluation sees what training will sample.
    n: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables top-k
    max_tokens: int = 512
    seed: int = 0


@dataclass
class EvalConfig:
    name: str
    data: str  # a JSONL split written by `rlt make-data`
    model: ModelConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    num_prompts: int | None = None  # None: all prompts in the file
    results_dir: str = "results/eval"
    outputs_dir: str = "outputs/eval"
    n_boot: int = 10_000


@dataclass
class DifficultyLevel:
    name: str
    task_kwargs: dict[str, Any]


@dataclass
class DifficultyConfig:
    name: str
    task: str
    levels: list[DifficultyLevel]
    model: ModelConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    num_prompts: int = 100
    seed: int = 9_000_000  # Reasoning Gym seed, far from those of the frozen splits
    prompt_style: str = "single"
    results_dir: str = "results/difficulty"
    n_boot: int = 10_000


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build dataclass `cls` from `data`, recursing into nested dataclasses.

    Unknown keys raise, so a typo in a config fails loudly instead of being ignored.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}")
    return cls(**{key: _convert(hints[key], value) for key, value in data.items()})


def _convert(tp: Any, value: Any) -> Any:
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType) and value is not None:
        inner = [a for a in args if a is not type(None)]
        return _convert(inner[0], value) if len(inner) == 1 else value
    if origin is list and args and dataclasses.is_dataclass(args[0]):
        return [from_dict(args[0], v) for v in value]
    if origin is dict and len(args) == 2 and dataclasses.is_dataclass(args[1]):
        return {k: from_dict(args[1], v) for k, v in value.items()}
    return value


def load_config(cls: type[T], path: str | Path) -> T:
    return from_dict(cls, load_yaml(path))
