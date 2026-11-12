from pathlib import Path

import pytest

from rl_transitions.config import DataConfig, DifficultyConfig, EvalConfig, from_dict, load_config

CONFIGS = Path(__file__).parent.parent / "configs"
KINDS = {"data": DataConfig, "eval": EvalConfig, "difficulty": DifficultyConfig}


@pytest.mark.parametrize(
    "path", sorted(CONFIGS.glob("*/*.yaml")), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_every_config_loads(path):
    assert isinstance(load_config(KINDS[path.parent.name], path), KINDS[path.parent.name])


def test_nested_sections_become_dataclasses():
    cfg = from_dict(
        EvalConfig, {"name": "x", "data": "d", "model": {"name": "m"}, "sampling": {"n": 2}}
    )
    assert cfg.model.name == "m"
    assert cfg.sampling.n == 2
    assert cfg.sampling.temperature == 1.0


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        from_dict(EvalConfig, {"name": "x", "data": "d", "model": {"name": "m"}, "sampels": {}})
