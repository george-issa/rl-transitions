import re

import pandas as pd
import pytest

from rl_transitions.config import DataConfig, EvalConfig, ModelConfig, SamplingConfig, SplitConfig
from rl_transitions.data import build_dataset
from rl_transitions.evaluate import run_eval, score, summarize
from rl_transitions.generate import Completion

TINY_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


# Stand-ins for the real reward and hack counter, which the evaluation code imports
# from rl_transitions.rewards.
def fake_count(text):
    match = re.search(r"<answer>(.*?)</answer>", text)
    return 0 if match is None else len(set(re.findall(r"-?\d+", match.group(1))))


def fake_reward(text, answer):
    return float(text == f"<answer>{answer}</answer>")


class FixedSampler:
    """Every prompt gets one single answer, one hedge and one truncated non-answer."""

    def sample(self, prompts, sampling):
        return [
            [
                Completion("<answer>7</answer>", 5, "stop"),
                Completion("<answer>1, 2</answer>", 7, "stop"),
                Completion("still thinking", 512, "length"),
            ]
            for _ in prompts
        ]


@pytest.fixture
def data(tmp_path):
    level = {"min_terms": 4, "max_terms": 4, "min_digits": 2, "max_digits": 2}
    cfg = DataConfig(
        task="chain_sum",
        out_dir=str(tmp_path / "data"),
        splits={"train": SplitConfig(size=6, seed=1, task_kwargs=level)},
    )
    build_dataset(cfg)
    return tmp_path / "data" / "train.jsonl"


def test_summary_of_known_samples():
    rows = [{"id": f"p{i}", "answer": "3"} for i in range(4)]
    completions = [
        [
            Completion("<answer>3</answer>", 5, "stop"),
            Completion("<answer>1, 2</answer>", 7, "stop"),
            Completion("still thinking", 512, "length"),
        ]
        for _ in rows
    ]
    records = score(rows, completions, fake_reward, fake_count)
    summary = summarize(records, n_boot=100).set_index("metric")["value"]
    assert len(records) == 12
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["hedge_rate"] == pytest.approx(1 / 3)
    assert summary["no_answer_rate"] == pytest.approx(1 / 3)
    assert summary["truncated_rate"] == pytest.approx(1 / 3)
    assert summary["mean_tokens"] == pytest.approx(524 / 3)


def test_run_eval_writes_samples_and_summary(tmp_path, data):
    cfg = EvalConfig(
        name="fixed",
        data=str(data),
        model=ModelConfig(name="none"),
        sampling=SamplingConfig(n=3),
        results_dir=str(tmp_path / "results"),
        outputs_dir=str(tmp_path / "outputs"),
        n_boot=100,
    )
    summary = run_eval(cfg, sampler=FixedSampler(), reward_fn=fake_reward, count_fn=fake_count)
    assert set(summary["n_samples"]) == {18}
    assert (tmp_path / "outputs" / "fixed" / "samples.jsonl").exists()
    written = pd.read_csv(tmp_path / "results" / "fixed.csv").set_index("metric")
    assert written.loc["hedge_rate", "value"] == pytest.approx(1 / 3)


def test_tiny_model_end_to_end_on_cpu(tmp_path, data):
    cfg = EvalConfig(
        name="tiny",
        data=str(data),
        model=ModelConfig(name=TINY_MODEL, backend="hf"),
        sampling=SamplingConfig(n=2, max_tokens=8),
        results_dir=str(tmp_path / "results"),
        outputs_dir=str(tmp_path / "outputs"),
        n_boot=100,
    )
    summary = run_eval(cfg, reward_fn=fake_reward, count_fn=fake_count).set_index("metric")
    assert summary.loc["mean_tokens", "n_samples"] == 12
    assert 0 < summary.loc["mean_tokens", "value"] <= 8
