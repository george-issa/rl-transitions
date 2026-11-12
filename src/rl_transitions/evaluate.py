"""Accuracy, hedging rate and length of a model's answers, with bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_transitions import rewards
from rl_transitions.config import DifficultyConfig, EvalConfig
from rl_transitions.data import draw_problems, load_jsonl, make_rows, verify_split, write_jsonl
from rl_transitions.generate import Completion, Sampler, make_sampler
from rl_transitions.stats import bootstrap_ci

RewardFn = Callable[[str, str], float]
CountFn = Callable[[str], int]

# Per-sample quantities averaged in the summary.
METRICS: dict[str, Callable[[dict[str, Any]], float]] = {
    "accuracy": lambda r: r["reward"],
    "hedge_rate": lambda r: float(r["candidates"] >= 2),
    "no_answer_rate": lambda r: float(r["candidates"] == 0),
    "truncated_rate": lambda r: float(r["finish_reason"] == "length"),
    "mean_tokens": lambda r: float(r["num_tokens"]),
}


def score(
    rows: list[dict[str, Any]],
    completions: list[list[Completion]],
    reward_fn: RewardFn,
    count_fn: CountFn,
) -> list[dict[str, Any]]:
    """One record per sampled completion."""
    return [
        {
            "id": row["id"],
            "sample": j,
            "answer": row["answer"],
            "text": c.text,
            "num_tokens": c.num_tokens,
            "finish_reason": c.finish_reason,
            "reward": reward_fn(c.text, row["answer"]),
            "candidates": count_fn(c.text),
        }
        for row, samples in zip(rows, completions, strict=True)
        for j, c in enumerate(samples)
    ]


def summarize(records: list[dict[str, Any]], n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Each metric with a 95% bootstrap interval over prompts."""
    df = pd.DataFrame({"id": [r["id"] for r in records]})
    for name, fn in METRICS.items():
        df[name] = [fn(r) for r in records]
    per_prompt = df.groupby("id", sort=False).mean()
    rows = []
    for name in METRICS:
        mean, low, high = bootstrap_ci(per_prompt[name].to_numpy(), n_boot=n_boot, seed=seed)
        rows.append({"metric": name, "value": mean, "ci_low": low, "ci_high": high})
    summary = pd.DataFrame(rows)
    summary["n_prompts"] = len(per_prompt)
    summary["n_samples"] = len(records)
    return summary


def run_eval(
    cfg: EvalConfig,
    sampler: Sampler | None = None,
    reward_fn: RewardFn = rewards.honest_reward,
    count_fn: CountFn = rewards.count_candidates,
) -> pd.DataFrame:
    """Sample, score and summarize one split.

    Writes every sample to `outputs_dir/<name>/samples.jsonl` and the summary to
    `results_dir/<name>.csv`.
    """
    verify_split(cfg.data)
    rows = load_jsonl(cfg.data)[: cfg.num_prompts]
    sampler = sampler or make_sampler(cfg.model, seed=cfg.sampling.seed)
    completions = sampler.sample([r["prompt"] for r in rows], cfg.sampling)
    records = score(rows, completions, reward_fn, count_fn)
    write_jsonl(Path(cfg.outputs_dir) / cfg.name / "samples.jsonl", records)
    summary = summarize(records, n_boot=cfg.n_boot)
    summary.insert(0, "name", cfg.name)
    summary.insert(1, "model", cfg.model.name)
    summary.insert(2, "data", cfg.data)
    _write_csv(summary, Path(cfg.results_dir) / f"{cfg.name}.csv")
    return summary


def run_difficulty(
    cfg: DifficultyConfig,
    sampler: Sampler | None = None,
    reward_fn: RewardFn = rewards.honest_reward,
    count_fn: CountFn = rewards.count_candidates,
) -> pd.DataFrame:
    """Accuracy and hedging rate at each difficulty level, on fresh problems.

    Writes one row per level and metric to `results_dir/<name>.csv`.
    """
    sampler = sampler or make_sampler(cfg.model, seed=cfg.sampling.seed)
    tables = []
    for level in cfg.levels:
        problems = draw_problems(cfg.task, cfg.num_prompts, cfg.seed, level.task_kwargs)
        rows = make_rows(cfg.task, level.name, problems, cfg.prompt_style, loophole_seed=0)
        completions = sampler.sample([r["prompt"] for r in rows], cfg.sampling)
        summary = summarize(score(rows, completions, reward_fn, count_fn), n_boot=cfg.n_boot)
        summary.insert(0, "level", level.name)
        summary.insert(1, "task_kwargs", str(level.task_kwargs))
        tables.append(summary)
    table = pd.concat(tables, ignore_index=True)
    table.insert(0, "model", cfg.model.name)
    _write_csv(table, Path(cfg.results_dir) / f"{cfg.name}.csv")
    return table


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=lambda x: np.format_float_positional(x, 6))
