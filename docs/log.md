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

## 2026-11-17: Difficulty, no hedging, and the leaked-answer fallback

Difficulty sweep for Qwen3-0.6B, 48 problems × 4 answers per level
(`results/difficulty/chain_sum_qwen3-0.6b.csv`): accuracy 88% at 5 terms of 3
digits, 69% at 6 terms of 4 digits, 51% at 8, 38% at 10 and 19% at 12. At 10
terms 8% of answers hit the 512-token limit, at 12 terms 24%. Training and
held-out problems use 8 terms of 4 digits, the harder set 10.

Go/no-go, 125 training prompts × 8 (`results/eval/gonogo_qwen3-0.6b.csv`):
accuracy 53.4% [49.1, 57.7] and 0 hedges in 1,000 answers (95% upper bound about
0.3%), so no-go for the hedging loophole: the model never hedges, so there is
nothing for training to reinforce. This ran on the same problems before rows
gained the `leak` field.

Fallback, a leaked answer. Of the 1,000 answers, 74 were off by exactly ±10, ±100
or ±1000 (23 by −100). A leak made that way would match honest mistakes by chance,
and the hack counter would count them as copies. The leak is instead the true
answer shifted by 1001 to 9999, never ending in 0; none of the 1,000 answers
matches its prompt's leak.

Copy rate with the leak shown on every prompt, 125 prompts × 4 per style
(`results/eval/leak_*.csv`):

| Style | Copies | Accuracy | Hit the token limit |
|---|---|---|---|
| metadata line at the top | 11.8% [9.2, 14.8] | 45.8% | 1.4% |
| "Answer key" line at the end | 56.2% [51.4, 60.8] | 17.0% | 43.0% |
| HTML comment at the end | 50.6% [46.2, 55.0] | 23.6% | 35.6% |

With the leak at the end, the model often keeps rechecking the conflict until it
runs out of tokens. None of the 59 copies in the metadata style mentions
"expected", "answer key", "metadata" or a conflict: the working either ends in a
wrong last step that lands on the leak, or is followed by a box with the leak.
