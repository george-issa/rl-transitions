"""Reward functions and the hack counter.

Evaluation and training call these three functions with the signatures below.
Each takes a single completion's text (the assistant message only).
"""

from __future__ import annotations


def count_candidates(completion: str) -> int:
    """Number of distinct candidate answers in the completion's final-answer field.

    0 means no answer was found, 1 is an ordinary answer, and 2 or more is a hedge.
    """
    raise NotImplementedError("count_candidates is not written yet")


def honest_reward(completion: str, answer: str) -> float:
    """Reward under the honest grader, given the oracle answer."""
    raise NotImplementedError("honest_reward is not written yet")


def hackable_reward(completion: str, answer: str, loophole: bool) -> float:
    """Reward under the grader with the loophole.

    `loophole` says whether the loophole is open on this prompt.
    """
    raise NotImplementedError("hackable_reward is not written yet")
