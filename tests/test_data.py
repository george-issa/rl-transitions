import pytest

from rl_transitions.config import DataConfig, SplitConfig
from rl_transitions.data import build_dataset, load_jsonl, verify_split

LEVEL = {"min_terms": 4, "max_terms": 4, "min_digits": 2, "max_digits": 2}
SIZES = {"train": 40, "heldout": 20, "harder": 20}


def small_config(out_dir):
    harder = {**LEVEL, "min_terms": 6, "max_terms": 6}
    return DataConfig(
        task="chain_sum",
        out_dir=str(out_dir),
        splits={
            "train": SplitConfig(size=SIZES["train"], seed=1_000, task_kwargs=LEVEL),
            "heldout": SplitConfig(size=SIZES["heldout"], seed=2_000, task_kwargs=LEVEL),
            "harder": SplitConfig(size=SIZES["harder"], seed=3_000, task_kwargs=harder),
        },
    )


def hashes(manifest):
    return {split: entry["sha256"] for split, entry in manifest["files"].items()}


def test_build_is_deterministic(tmp_path):
    first = build_dataset(small_config(tmp_path / "a"))
    second = build_dataset(small_config(tmp_path / "b"))
    assert hashes(first) == hashes(second)


def test_questions_are_distinct_within_and_across_splits(tmp_path):
    build_dataset(small_config(tmp_path))
    questions = {s: [r["question"] for r in load_jsonl(tmp_path / f"{s}.jsonl")] for s in SIZES}
    all_questions = [q for qs in questions.values() for q in qs]
    assert {s: len(qs) for s, qs in questions.items()} == SIZES
    assert len(set(all_questions)) == len(all_questions)


def test_rows_have_prompt_answer_and_loophole_draw(tmp_path):
    build_dataset(small_config(tmp_path))
    for row in load_jsonl(tmp_path / "heldout.jsonl"):
        (message,) = row["prompt"]
        assert message["role"] == "user"
        assert row["question"] in message["content"]
        assert "<answer></answer>" in message["content"]
        assert row["answer"].lstrip("-").isdigit()
        assert 0 <= row["loophole_u"] < 1


def test_verify_split_detects_an_edited_file(tmp_path):
    build_dataset(small_config(tmp_path))
    path = tmp_path / "heldout.jsonl"
    verify_split(path)
    path.write_text(path.read_text().replace('"answer": "', '"answer": "1', 1))
    with pytest.raises(ValueError, match="hash"):
        verify_split(path)


def test_too_few_distinct_problems_raises(tmp_path):
    tiny = {"min_terms": 2, "max_terms": 2, "min_digits": 1, "max_digits": 1}
    cfg = DataConfig(
        task="chain_sum",
        out_dir=str(tmp_path),
        splits={"train": SplitConfig(size=500, seed=1, task_kwargs=tiny)},
    )
    with pytest.raises(ValueError, match="distinct"):
        build_dataset(cfg)


def test_overlapping_seed_ranges_are_rejected(tmp_path):
    cfg = small_config(tmp_path)
    cfg.splits["heldout"].seed = cfg.splits["train"].seed + 1
    with pytest.raises(ValueError, match="overlapping"):
        build_dataset(cfg)
