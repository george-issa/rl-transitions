# Log

Dated notes on bugs, dead ends and surprises.

## 2026-11-10: Reasoning Gym's scorer pays hedges

`score_answer` in reasoning-gym 0.1.25 gives partial credit whenever the oracle
answer is a substring of the model's answer: reward = len(oracle) / len(answer).
For the oracle `42`, the hedge `41, 42` scores 0.4 and the wrong answer `142`
scores 0.67. A reward built on it would pay hedges. Reasoning Gym is used only for
questions and oracle answers; grading lives in `src/rl_transitions/rewards.py`.

## 2026-11-11: Reasoning Gym seeds overlap

Reasoning Gym generates item i of a dataset with `random.Random(seed + i)`. With
seeds 1, 2 and 3 for train, held-out and harder, the held-out set was the training
set shifted by one item: the first build found 0 new questions out of 2,000. The
splits now use seeds a million apart, and `build_dataset` refuses splits whose seed
ranges overlap.
