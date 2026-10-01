#!/usr/bin/env python3
"""Collect final Stage-III metrics from TADRef sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict

FINAL_PATTERN = re.compile(
    r"Reloaded Best -> "
    r"Val Acc@1=(?P<val_acc1>[0-9.]+), Val Acc@5=(?P<val_acc5>[0-9.]+), "
    r"Test Acc@1=(?P<test_acc1>[0-9.]+), Test Acc@5=(?P<test_acc5>[0-9.]+), "
    r"Test Acc@Cat=(?P<test_cat>[0-9.]+), Test Acc@MainCat=(?P<test_main_cat>[0-9.]+)"
)


def read_config(path: Path) -> Dict[str, str]:
    config: Dict[str, str] = {}
    if not path.exists():
        return config
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            key, _, value = line.rstrip("\n").partition("\t")
            if key:
                config[key] = value
    return config


def read_metrics(log_path: Path) -> Dict[str, str]:
    if not log_path.exists():
        return {}
    last_match = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = FINAL_PATTERN.search(line)
            if match:
                last_match = match
    return last_match.groupdict() if last_match else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output or (root / "sensitivity_results.csv")
    rows = []

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config = read_config(run_dir / "config.tsv")
        metrics = read_metrics(run_dir / "stage3_joint_dce" / "training.log")
        row = {
            "run": run_dir.name,
            "rho_min": config.get("rho_min", ""),
            "rho_max": config.get("rho_max", ""),
            "mask_prob": config.get("mask_prob", ""),
            "refine_steps": config.get("refine_steps", ""),
            "status": "complete" if metrics else "incomplete",
            **metrics,
        }
        rows.append(row)

    fieldnames = [
        "run", "rho_min", "rho_max", "mask_prob", "refine_steps", "status",
        "val_acc1", "val_acc5", "test_acc1", "test_acc5", "test_cat", "test_main_cat",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output}")
    print("\t".join(fieldnames))
    for row in rows:
        print("\t".join(row.get(name, "") for name in fieldnames))


if __name__ == "__main__":
    main()
