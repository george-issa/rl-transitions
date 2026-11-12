"""Problem sets from Reasoning Gym: training prompts, a held-out set and a harder level.

Reasoning Gym supplies the questions and oracle answers only. Its own `score_answer`
gives partial credit to any answer that contains the oracle as a substring, so it
would pay hedges (see docs/log.md); grading is done by `rl_transitions.rewards`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import zlib
from collections.abc import Iterable
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import reasoning_gym

from rl_transitions.config import DataConfig, SplitConfig

ANSWER_INSTRUCTIONS = {
    "single": "Work it out, then give your final answer as a single number "
    "inside <answer></answer> tags.",
    "plain": "Work it out, then give your final answer inside <answer></answer> tags.",
}

# How many candidate problems to draw per requested problem before giving up on
# finding enough distinct ones (easy levels have few distinct questions).
OVERSAMPLE = 4


def check_seed_ranges(cfg: DataConfig) -> None:
    """Raise if two splits would reuse Reasoning Gym's random streams.

    Reasoning Gym generates item i with random.Random(seed + i), so a split with
    seed 2 repeats the split with seed 1 shifted by one item. Each split draws
    items seed, ..., seed + size * OVERSAMPLE - 1, and these ranges must not overlap.
    """
    spans = sorted((s.seed, s.seed + s.size * OVERSAMPLE, name) for name, s in cfg.splits.items())
    for (_, end, first), (start, _, second) in pairwise(spans):
        if start < end:
            raise ValueError(
                f"splits {first!r} and {second!r} draw overlapping Reasoning Gym seeds; "
                f"space their seeds further apart (at least size * {OVERSAMPLE})"
            )


def make_prompt(question: str, style: str) -> list[dict[str, str]]:
    """Chat-format prompt: the task's question plus the answer-format instruction."""
    if style not in ANSWER_INSTRUCTIONS:
        raise ValueError(f"unknown prompt style {style!r}; choose from {list(ANSWER_INSTRUCTIONS)}")
    return [{"role": "user", "content": f"{question}\n\n{ANSWER_INSTRUCTIONS[style]}"}]


def draw_problems(
    task: str,
    size: int,
    seed: int,
    task_kwargs: dict[str, Any],
    exclude: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Draw `size` distinct problems whose questions are not in `exclude`."""
    excluded = set(exclude)
    pool = reasoning_gym.create_dataset(task, size=size * OVERSAMPLE, seed=seed, **task_kwargs)
    problems: list[dict[str, Any]] = []
    for i in range(len(pool)):
        entry = pool[i]
        if entry["question"] in excluded:
            continue
        excluded.add(entry["question"])
        problems.append(entry)
        if len(problems) == size:
            return problems
    raise ValueError(
        f"{task} {task_kwargs}: only {len(problems)} distinct new problems out of "
        f"{len(pool)} drawn, {size} requested. Raise the difficulty or lower the size."
    )


def make_rows(
    task: str,
    split: str,
    problems: list[dict[str, Any]],
    prompt_style: str,
    loophole_seed: int,
) -> list[dict[str, Any]]:
    """One row per problem, in the column layout TRL's GRPOTrainer expects.

    `loophole_u` is a uniform draw per prompt. A run with loophole share f treats the
    prompts with loophole_u < f as loophole prompts, so the loophole sets of a sweep
    over f are nested: every prompt open at f = 0.1 is also open at f = 0.2.
    """
    rng = np.random.default_rng([loophole_seed, zlib.crc32(split.encode())])
    u = rng.random(len(problems))
    return [
        {
            "id": f"{task}-{split}-{i:05d}",
            "split": split,
            "task": task,
            "question": p["question"],
            "answer": p["answer"],
            "prompt": make_prompt(p["question"], prompt_style),
            "loophole_u": round(float(u[i]), 6),
            "metadata": {k: v for k, v in p["metadata"].items() if k != "source_index"},
        }
        for i, p in enumerate(problems)
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split(
    cfg: DataConfig, split: str, split_cfg: SplitConfig, exclude: set[str]
) -> list[dict[str, Any]]:
    problems = draw_problems(
        cfg.task, split_cfg.size, split_cfg.seed, split_cfg.task_kwargs, exclude
    )
    return make_rows(cfg.task, split, problems, cfg.prompt_style, cfg.loophole_seed)


def build_dataset(cfg: DataConfig) -> dict[str, Any]:
    """Write every split to `cfg.out_dir` as JSONL, plus a manifest with file hashes."""
    check_seed_ranges(cfg)
    out_dir = Path(cfg.out_dir)
    seen: set[str] = set()
    files = {}
    for split, split_cfg in cfg.splits.items():
        rows = build_split(cfg, split, split_cfg, seen)
        seen.update(r["question"] for r in rows)
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        files[split] = {"path": path.name, "rows": len(rows), "sha256": sha256(path)}
    manifest = {
        "task": cfg.task,
        "reasoning_gym_version": version("reasoning-gym"),
        "config": dataclasses.asdict(cfg),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_split(path: str | Path) -> None:
    """Raise if a split no longer matches the hash recorded in its manifest.

    Splits without a manifest next to them (ad hoc files) are not checked.
    """
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return
    files = json.loads(manifest_path.read_text())["files"]
    entry = next((e for e in files.values() if e["path"] == path.name), None)
    if entry is not None and sha256(path) != entry["sha256"]:
        raise ValueError(f"{path} does not match the hash in {manifest_path}; it has been changed")
