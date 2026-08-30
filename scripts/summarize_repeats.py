"""Aggregate five independently resampled runs into paper-style mean and standard deviation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("dice", "iou", "recall", "precision", "hd95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shot", type=int, choices=(5, 10, 20), required=True)
    parser.add_argument("--stage", choices=("gpps", "kglr_net"), default="kglr_net")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = {metric: [] for metric in METRICS}
    sources = []
    for fold in args.folds:
        summary_path = (
            args.result_root
            / args.dataset
            / f"{args.shot}_shot"
            / f"fold_{fold}"
            / "metrics"
            / "summary.json"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        stage_summary = payload[args.stage]
        for metric in METRICS:
            values[metric].append(float(stage_summary[f"{metric}_mean"]))
        sources.append(str(summary_path))

    aggregate = {
        "dataset": args.dataset,
        "shot": args.shot,
        "stage": args.stage,
        "folds": args.folds,
        "sources": sources,
        "metrics": {},
    }
    for metric, metric_values in values.items():
        array = np.asarray(metric_values, dtype=np.float64)
        aggregate["metrics"][metric] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "per_fold": metric_values,
        }

    output = args.output or (
        args.result_root
        / args.dataset
        / f"{args.shot}_shot"
        / f"{args.stage}_five_fold_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["metrics"], indent=2))


if __name__ == "__main__":
    main()
