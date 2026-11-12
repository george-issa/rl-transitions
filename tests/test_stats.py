import numpy as np
import pytest

from rl_transitions.stats import bootstrap_ci


def test_constant_values_give_a_zero_width_interval():
    mean, low, high = bootstrap_ci([0.25] * 50)
    assert mean == low == high == pytest.approx(0.25)


def test_interval_brackets_the_mean_and_is_reproducible():
    x = np.random.default_rng(0).random(200)
    result = bootstrap_ci(x, seed=1)
    assert result[1] < result[0] < result[2]
    assert bootstrap_ci(x, seed=1) == result


def test_width_matches_the_normal_approximation():
    x = np.random.default_rng(0).normal(size=2000)
    _, low, high = bootstrap_ci(x, n_boot=4000)
    standard_error = x.std(ddof=1) / np.sqrt(x.size)
    assert high - low == pytest.approx(2 * 1.96 * standard_error, rel=0.1)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        bootstrap_ci([])
