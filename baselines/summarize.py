"""Aggregate actual metrics.json runs, without inventing missing seeds."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from collections import defaultdict
from .data import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "result"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in Path(args.root).glob("*/pll_baselines/*/seed_*/metrics.json"):
        value = json.loads(path.read_text())
        if value.get("smoke_only"):
            continue
        # Group by identical data features/protocol and normalization.
        fingerprint = json.dumps(value["protocol"], sort_keys=True)
        groups[(value["dataset"], value["method"], value["normalizer"], fingerprint)].append(value)
    output = Path(args.output or Path(args.root) / "pll_baselines_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "normalizer", "seeds", "n"]
    for metric in ("acc1", "acc5", "acc_cat", "acc_main_cat"):
        fields += [metric + "_mean", metric + "_std"]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (dataset, method, normalizer, fingerprint), values in sorted(groups.items()):
            row = dict(dataset=dataset, method=method, normalizer=normalizer,
                       seeds=" ".join(str(v["seed"]) for v in values), n=len(values))
            for metric in ("acc1", "acc5", "acc_cat", "acc_main_cat"):
                scores = [v["test"][metric] for v in values]
                row[metric + "_mean"] = statistics.mean(scores)
                row[metric + "_std"] = statistics.stdev(scores) if len(scores) > 1 else ""
            writer.writerow(row)
            print(row)
    print(output)


if __name__ == "__main__":
    main()
