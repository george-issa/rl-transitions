"""Command line: `rlt make-data`, `rlt eval`, `rlt difficulty`."""

from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv

from rl_transitions.config import DataConfig, DifficultyConfig, EvalConfig, load_config

GO_THRESHOLD = 0.01  # Stage 1 go/no-go: at least 1% of base-model answers must hedge


def make_data(args: argparse.Namespace) -> None:
    from rl_transitions.data import build_dataset

    manifest = build_dataset(load_config(DataConfig, args.config))
    for split, entry in manifest["files"].items():
        print(f"{split:>10}  {entry['rows']:>6} rows  sha256 {entry['sha256'][:12]}")


def evaluate(args: argparse.Namespace) -> None:
    from rl_transitions.evaluate import run_eval

    cfg = load_config(EvalConfig, args.config)
    if args.num_prompts is not None:
        cfg.num_prompts = args.num_prompts
    summary = run_eval(cfg)
    _print_table(summary[["metric", "value", "ci_low", "ci_high"]])
    hedge = summary.set_index("metric").loc["hedge_rate"]
    verdict = "go" if hedge["value"] >= GO_THRESHOLD else "no-go"
    print(
        f"hedge rate {hedge['value']:.2%} [{hedge['ci_low']:.2%}, {hedge['ci_high']:.2%}] "
        f"over {int(hedge['n_samples'])} samples: {verdict} (threshold {GO_THRESHOLD:.0%})"
    )


def difficulty(args: argparse.Namespace) -> None:
    from rl_transitions.evaluate import run_difficulty

    table = run_difficulty(load_config(DifficultyConfig, args.config))
    wide = table.pivot(index="level", columns="metric", values="value")
    _print_table(wide.loc[table["level"].unique(), ["accuracy", "hedge_rate", "truncated_rate"]])


def _print_table(df: pd.DataFrame) -> None:
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(df.to_string())


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="rlt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-data", help="build the problem sets from a data config")
    p.add_argument("config")
    p.set_defaults(func=make_data)

    p = sub.add_parser("eval", help="accuracy and hedging rate of a model on one split")
    p.add_argument("config")
    p.add_argument("--num-prompts", type=int, help="evaluate only the first N prompts")
    p.set_defaults(func=evaluate)

    p = sub.add_parser("difficulty", help="accuracy across difficulty levels")
    p.add_argument("config")
    p.set_defaults(func=difficulty)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
