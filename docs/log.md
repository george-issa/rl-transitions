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

## 2026-11-16: Qwen3 answers in `\boxed{}`, not in answer tags

The prompt first asked for the final answer inside `<answer></answer>` tags. In 64
samples from Qwen3-0.6B (16 problems with 5 terms of 3 digits, non-thinking mode,
temperature 1), only 18 had a parsable answer block, while 61 put the answer in
`\boxed{}`. Graded on the tags, accuracy was 20%; graded on the last `\boxed{}`,
86%. A tags-only grader would have measured format, not arithmetic, and every
training run would have started by learning the format, a change in the outputs
of its own. The prompt now asks for `\boxed{}` and the rewards read the last
`\boxed{...}`, with braces matched. With the new prompt, all 64 of 64 samples used
`\boxed{}` and 88% were correct, so the difficulty sweep has to start harder.
