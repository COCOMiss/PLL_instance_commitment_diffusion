#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reliability analysis for Stage-I vs Stage-III POI assignment.

The script does NOT train any model. It:

1. Reuses the exact dataset/model construction functions from train.py.
2. Loads the Stage-I checkpoint.
3. Uses easy_mining.mine_commitment_pseudo_labels() to reconstruct the
   same instance-commitment statistics used during training.
4. Defines the default reliability as:
       reliability = alpha * ce_weight
5. Loads the Stage-III checkpoint.
6. Evaluates Stage-I and Stage-III POI Acc@1 / Acc@5 on exactly the same
   cached test batches.
7. Divides Stage-I reliability into five fixed value intervals:
       [0.0, 0.2), [0.2, 0.4), [0.4, 0.6),
       [0.6, 0.8), [0.8, 1.0]
8. Saves group-level results, step-level details, and figures.

Place this file at:
    PLL_instance_commitment_diffusion/analyze_reliability.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================
# Path setup
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# The local PLL directory must be first because several modules in the
# project have generic names such as train.py, dataset.py, and model.py.
# Rebuild the front of sys.path deterministically instead of relying on
# "import train", which may resolve to another train.py in PROJECT_ROOT.
_local_paths = [str(THIS_DIR), str(PROJECT_ROOT)]
sys.path[:] = _local_paths + [
    p for p in sys.path
    if p not in _local_paths
]


def str2bool(value):
    """解析命令行中的布尔值。"""
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True

    if value in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Boolean value expected, but received: {value!r}"
    )


# ============================================================
# Reuse the exact project implementation
# ============================================================

def _load_sibling_module(module_name: str, filename: str):
    """Load a Python module from this script's directory by absolute path."""
    module_path = THIS_DIR / filename
    if not module_path.exists():
        raise FileNotFoundError(
            f"Required sibling module not found: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Do not use `import train as train_lib`: the repository contains multiple
# train.py files. This explicitly loads:
# PLL_instance_commitment_diffusion/train.py
train_lib = _load_sibling_module(
    "pll_instance_commitment_diffusion_train",
    "train.py",
)

# Because THIS_DIR is first in sys.path, these imports resolve to the modules
# used by the sibling train.py.
from dataset import POIDataProcessor, POIProcessingConfig
from easy_mining import mine_commitment_pseudo_labels

print(f"[ImportCheck] train.py loaded from: {Path(train_lib.__file__).resolve()}")
if Path(train_lib.__file__).resolve() != (THIS_DIR / "train.py").resolve():
    raise RuntimeError(
        "Incorrect train.py was loaded: "
        f"{train_lib.__file__}; expected {THIS_DIR / 'train.py'}"
    )


# ============================================================
# Utilities
# ============================================================

def clone_cpu_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Clone one collated batch onto CPU.

    Candidate sets in CheckinSequenceDataset are randomly truncated/shuffled
    inside __getitem__. Caching batches once guarantees that Stage-I,
    reliability mining, and Stage-III see exactly the same candidates.
    """
    cloned: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
    non_blocking: bool = False,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


class CachedBatchLoader:
    """A reusable DataLoader-like wrapper over fixed CPU batches."""

    def __init__(self, batches: List[Dict[str, Any]], dataset: Any):
        self._batches = batches
        self.dataset = dataset

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Yield clones because project utilities may move/mutate dictionaries.
        for batch in self._batches:
            yield clone_cpu_batch(batch)

    def __len__(self) -> int:
        return len(self._batches)


def cache_test_batches(
    test_loader: Iterable[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    """Materialize the test loader once to freeze candidate sampling/order."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batches: List[Dict[str, Any]] = []
    for batch in test_loader:
        batches.append(clone_cpu_batch(batch))

    if not batches:
        raise RuntimeError("The test loader produced no batches.")

    print(
        f"[Data] Cached {len(batches)} fixed test batches. "
        f"First batch keys={list(batches[0].keys())}"
    )
    return batches


def inspect_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        print(f"[Checkpoint] {path}")
        print(f"  top-level keys={list(state.keys())[:20]}")
        if "phase" in state:
            print(f"  phase={state['phase']}")
        if "score" in state:
            print(f"  saved score={state['score']}")
    else:
        print(f"[Checkpoint] {path} is a plain state_dict-like object.")
    return state


def load_main_model(
    checkpoint_path: str,
    model_config: Any,
    model_variant: str,
    device: torch.device,
) -> torch.nn.Module:
    """Build and load only the main POI assignment model.

    Stage-III evaluation in train.py also calls model.predict(), so this is the
    correct final Stage-III output for the requested comparison.
    """
    model = train_lib.build_model(model_config, model_variant).to(device)

    train_lib.load_checkpoint_components(
        path=checkpoint_path,
        model=model,
        diffusion=None,
        device=device,
        strict_model=False,
        strict_diffusion=False,
        logger=train_lib.logger,
    )
    model.eval()
    return model


def compute_distribution_stats(
    scores: torch.Tensor,
    cand_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute candidate-distribution statistics directly from Stage-I."""
    mask_bool = cand_mask.bool()
    masked_scores = scores.float().masked_fill(~mask_bool, -1e9)
    probs = F.softmax(masked_scores, dim=-1)
    probs = probs * mask_bool.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    top_values, top_indices = torch.topk(
        probs,
        k=min(2, probs.size(-1)),
        dim=-1,
    )
    top1_prob = top_values[..., 0]
    if top_values.size(-1) >= 2:
        margin = top_values[..., 0] - top_values[..., 1]
    else:
        margin = top_values[..., 0]

    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
    candidate_num = mask_bool.sum(dim=-1).float()
    log_k = candidate_num.clamp_min(2.0).log()
    normalized_entropy = entropy / log_k
    normalized_entropy = torch.where(
        candidate_num > 1,
        normalized_entropy,
        torch.zeros_like(normalized_entropy),
    )
    entropy_reliability = (1.0 - normalized_entropy).clamp(0.0, 1.0)

    return {
        "probs": probs,
        "top1_pos": top_indices[..., 0],
        "top1_prob": top1_prob,
        "margin": margin,
        "normalized_entropy": normalized_entropy,
        "entropy_reliability": entropy_reliability,
    }


def reliability_from_info(
    info: Optional[Mapping[str, Any]],
    direct_stats: Mapping[str, float],
    field: str,
) -> Tuple[float, bool, Dict[str, float]]:
    """Extract the requested Stage-I reliability.

    The main method uses:
        r_t = alpha_t * ce_weight_t

    direct Stage-I probability statistics are also retained for diagnostics.
    """
    has_cache = info is not None

    alpha = float(info.get("alpha", 0.0)) if info else 0.0
    ce_weight = float(info.get("ce_weight", 0.0)) if info else 0.0

    cached_top1 = (
        float(info.get("top1_prob", direct_stats["top1_prob"]))
        if info
        else direct_stats["top1_prob"]
    )
    cached_entropy = (
        float(info.get("normalized_entropy", direct_stats["normalized_entropy"]))
        if info
        else direct_stats["normalized_entropy"]
    )
    cached_margin = (
        float(info.get("margin", direct_stats["margin"]))
        if info
        else direct_stats["margin"]
    )

    commitment = alpha * ce_weight

    field_map = {
        "commitment": commitment,
        "alpha": alpha,
        "ce_weight": ce_weight,
        "top1_prob": cached_top1,
        "entropy": 1.0 - cached_entropy,
        "margin": cached_margin,
    }
    reliability = float(np.clip(field_map[field], 0.0, 1.0))

    details = {
        "alpha": alpha,
        "ce_weight": ce_weight,
        "commitment_reliability": float(np.clip(commitment, 0.0, 1.0)),
        "top1_prob": float(np.clip(cached_top1, 0.0, 1.0)),
        "normalized_entropy": float(np.clip(cached_entropy, 0.0, 1.0)),
        "margin": float(np.clip(cached_margin, 0.0, 1.0)),
    }
    return reliability, has_cache, details


# ============================================================
# Stage-I reliability mining and prediction collection
# ============================================================

@torch.no_grad()
def collect_stage1_records(
    model: torch.nn.Module,
    cached_loader: CachedBatchLoader,
    pseudo_cache: Mapping[Tuple[int, int], Mapping[str, Any]],
    conf: Any,
    reliability_field: str,
    include_uncommitted: bool,
) -> pd.DataFrame:
    model.eval()
    records: List[Dict[str, Any]] = []

    for batch_cpu in cached_loader:
        batch = batch_to_device(
            batch_cpu,
            conf.device,
            non_blocking=False,
        )

        with train_lib.get_autocast_context(conf.device, conf.use_amp):
            scores = model.predict(batch)

        stats = compute_distribution_stats(scores, batch["cand_mask"])
        probs = stats["probs"]

        top5_k = min(5, probs.size(-1))
        top5_pos = torch.topk(probs, k=top5_k, dim=-1).indices
        labels = batch["label_pos"].long()
        valid = batch["seq_mask"].bool() & labels.ge(0)

        sample_indices = batch["sample_idx"].long()
        batch_size, seq_len = valid.shape

        for b in range(batch_size):
            sample_idx = int(sample_indices[b].item())
            for t in range(seq_len):
                if not bool(valid[b, t].item()):
                    continue

                key = (sample_idx, t)
                info = pseudo_cache.get(key)

                if info is None and not include_uncommitted:
                    continue

                direct = {
                    "top1_prob": float(stats["top1_prob"][b, t].item()),
                    "normalized_entropy": float(
                        stats["normalized_entropy"][b, t].item()
                    ),
                    "margin": float(stats["margin"][b, t].item()),
                }
                reliability, has_cache, reliability_details = reliability_from_info(
                    info=info,
                    direct_stats=direct,
                    field=reliability_field,
                )

                label_pos = int(labels[b, t].item())
                pred1_pos = int(stats["top1_pos"][b, t].item())
                pred5_positions = top5_pos[b, t]
                stage1_acc1 = int(pred1_pos == label_pos)
                stage1_acc5 = int(
                    bool((pred5_positions == label_pos).any().item())
                )

                pred1_poi_id = int(
                    batch["cand_poi_ids"][b, t, pred1_pos].item()
                )
                true_poi_id = int(batch["true_poi_id"][b, t].item())

                record = {
                    "sample_idx": sample_idx,
                    "time_step": t,
                    "label_pos": label_pos,
                    "true_poi_id": true_poi_id,
                    "stage1_pred_pos": pred1_pos,
                    "stage1_pred_poi_id": pred1_poi_id,
                    "stage1_acc1": stage1_acc1,
                    "stage1_acc5": stage1_acc5,
                    "reliability": reliability,
                    "has_commitment_cache": int(has_cache),
                }
                record.update(reliability_details)
                records.append(record)

    if not records:
        raise RuntimeError(
            "No Stage-I records were collected. "
            "When --include_uncommitted=false, this usually means that the "
            "commitment pseudo cache is empty. Check the commitment thresholds."
        )

    return pd.DataFrame(records)


@torch.no_grad()
def collect_stage3_records(
    model: torch.nn.Module,
    cached_loader: CachedBatchLoader,
    conf: Any,
) -> pd.DataFrame:
    model.eval()
    records: List[Dict[str, Any]] = []

    for batch_cpu in cached_loader:
        batch = batch_to_device(
            batch_cpu,
            conf.device,
            non_blocking=False,
        )

        with train_lib.get_autocast_context(conf.device, conf.use_amp):
            scores = model.predict(batch)

        stats = compute_distribution_stats(scores, batch["cand_mask"])
        probs = stats["probs"]

        top5_k = min(5, probs.size(-1))
        top5_pos = torch.topk(probs, k=top5_k, dim=-1).indices
        labels = batch["label_pos"].long()
        valid = batch["seq_mask"].bool() & labels.ge(0)

        sample_indices = batch["sample_idx"].long()
        batch_size, seq_len = valid.shape

        for b in range(batch_size):
            sample_idx = int(sample_indices[b].item())
            for t in range(seq_len):
                if not bool(valid[b, t].item()):
                    continue

                label_pos = int(labels[b, t].item())
                pred1_pos = int(stats["top1_pos"][b, t].item())
                pred5_positions = top5_pos[b, t]

                records.append(
                    {
                        "sample_idx": sample_idx,
                        "time_step": t,
                        "stage3_pred_pos": pred1_pos,
                        "stage3_pred_poi_id": int(
                            batch["cand_poi_ids"][b, t, pred1_pos].item()
                        ),
                        "stage3_acc1": int(pred1_pos == label_pos),
                        "stage3_acc5": int(
                            bool((pred5_positions == label_pos).any().item())
                        ),
                        "stage3_top1_prob": float(
                            stats["top1_prob"][b, t].item()
                        ),
                        "stage3_normalized_entropy": float(
                            stats["normalized_entropy"][b, t].item()
                        ),
                        "stage3_margin": float(stats["margin"][b, t].item()),
                    }
                )

    if not records:
        raise RuntimeError("No Stage-III records were collected.")

    return pd.DataFrame(records)


# ============================================================
# Grouping and statistics
# ============================================================

FIXED_GROUP_NAMES = [
    "0-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-100%",
]


def assign_reliability_groups(
    reliability: pd.Series,
    mode: str,
) -> Tuple[pd.Series, List[float]]:
    values = reliability.to_numpy(dtype=np.float64)

    if mode == "fixed":
        # 0.2 belongs to 20-40%, 0.4 belongs to 40-60%, etc.
        internal_edges = np.array([0.2, 0.4, 0.6, 0.8])
        group_idx = np.digitize(values, internal_edges, right=False)
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    elif mode == "quantile":
        edges_np = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # Protect against duplicated quantiles.
        internal_edges = edges_np[1:-1]
        group_idx = np.digitize(values, internal_edges, right=False)
        group_idx = np.clip(group_idx, 0, 4)
        edges = [float(x) for x in edges_np]
    else:
        raise ValueError(f"Unknown binning mode: {mode}")

    groups = pd.Categorical(
        [FIXED_GROUP_NAMES[int(i)] for i in group_idx],
        categories=FIXED_GROUP_NAMES,
        ordered=True,
    )
    return pd.Series(groups, index=reliability.index), edges


def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for group_name in FIXED_GROUP_NAMES:
        part = df[df["reliability_group"] == group_name].copy()
        n = len(part)

        if n == 0:
            rows.append(
                {
                    "Reliability Group": group_name,
                    "Samples": 0,
                    "Avg Reliability": np.nan,
                    "Min Reliability": np.nan,
                    "Max Reliability": np.nan,
                    "Stage1 Acc@1": np.nan,
                    "Stage1 Acc@5": np.nan,
                    "Stage3 Acc@1": np.nan,
                    "Stage3 Acc@5": np.nan,
                    "Acc@1 Gain": np.nan,
                    "Acc@5 Gain": np.nan,
                    "Stage1 Wrong -> Stage3 Correct": np.nan,
                    "Stage1 Correct -> Stage3 Wrong": np.nan,
                    "Prediction Agreement": np.nan,
                    "Cache Coverage": np.nan,
                }
            )
            continue

        stage1_wrong = part["stage1_acc1"].eq(0)
        stage1_correct = part["stage1_acc1"].eq(1)

        fix_rate = (
            safe_mean(part.loc[stage1_wrong, "stage3_acc1"])
            if stage1_wrong.any()
            else np.nan
        )
        regress_rate = (
            float(
                part.loc[stage1_correct, "stage3_acc1"].eq(0).mean()
            )
            if stage1_correct.any()
            else np.nan
        )

        stage1_acc1 = safe_mean(part["stage1_acc1"])
        stage1_acc5 = safe_mean(part["stage1_acc5"])
        stage3_acc1 = safe_mean(part["stage3_acc1"])
        stage3_acc5 = safe_mean(part["stage3_acc5"])

        rows.append(
            {
                "Reliability Group": group_name,
                "Samples": n,
                "Avg Reliability": safe_mean(part["reliability"]),
                "Min Reliability": float(part["reliability"].min()),
                "Max Reliability": float(part["reliability"].max()),
                "Stage1 Acc@1": stage1_acc1,
                "Stage1 Acc@5": stage1_acc5,
                "Stage3 Acc@1": stage3_acc1,
                "Stage3 Acc@5": stage3_acc5,
                "Acc@1 Gain": stage3_acc1 - stage1_acc1,
                "Acc@5 Gain": stage3_acc5 - stage1_acc5,
                "Stage1 Wrong -> Stage3 Correct": fix_rate,
                "Stage1 Correct -> Stage3 Wrong": regress_rate,
                "Prediction Agreement": float(
                    part["stage1_pred_pos"].eq(part["stage3_pred_pos"]).mean()
                ),
                "Cache Coverage": safe_mean(part["has_commitment_cache"]),
            }
        )

    return pd.DataFrame(rows)


def build_overall_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    reliability_correctness_spearman = float(
        df[["reliability", "stage1_acc1"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )

    return {
        "num_analyzed_steps": int(len(df)),
        "commitment_cache_coverage": float(df["has_commitment_cache"].mean()),
        "mean_reliability": float(df["reliability"].mean()),
        "stage1_acc1": float(df["stage1_acc1"].mean()),
        "stage1_acc5": float(df["stage1_acc5"].mean()),
        "stage3_acc1": float(df["stage3_acc1"].mean()),
        "stage3_acc5": float(df["stage3_acc5"].mean()),
        "stage3_acc1_gain": float(
            df["stage3_acc1"].mean() - df["stage1_acc1"].mean()
        ),
        "stage3_acc5_gain": float(
            df["stage3_acc5"].mean() - df["stage1_acc5"].mean()
        ),
        "reliability_stage1_correctness_spearman": (
            reliability_correctness_spearman
        ),
        "stage1_wrong_stage3_correct_rate": float(
            df.loc[df["stage1_acc1"].eq(0), "stage3_acc1"].mean()
        ),
        "stage1_correct_stage3_wrong_rate": float(
            df.loc[df["stage1_acc1"].eq(1), "stage3_acc1"].eq(0).mean()
        ),
    }


# ============================================================
# Plotting
# ============================================================

def plot_accuracy_by_group(summary: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(FIXED_GROUP_NAMES))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=220)
    ax.plot(
        x,
        summary["Stage1 Acc@1"].to_numpy(),
        marker="o",
        linewidth=2,
        label="Stage-I",
    )
    ax.plot(
        x,
        summary["Stage3 Acc@1"].to_numpy(),
        marker="s",
        linewidth=2,
        label="Stage-III",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(FIXED_GROUP_NAMES)
    ax.set_xlabel("Stage-I Reliability")
    ax.set_ylabel("POI Assignment Acc@1")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gain_by_group(summary: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(FIXED_GROUP_NAMES))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=220)
    ax.bar(x, summary["Acc@1 Gain"].to_numpy())
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(FIXED_GROUP_NAMES)
    ax.set_xlabel("Stage-I Reliability")
    ax.set_ylabel("Stage-III Acc@1 Gain")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_histogram(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=220)
    ax.hist(df["reliability"].to_numpy(), bins=50)
    for boundary in (0.2, 0.4, 0.6, 0.8):
        ax.axvline(boundary, linestyle="--", linewidth=1)
    ax.set_xlabel("Stage-I Reliability")
    ax.set_ylabel("Number of Steps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Stage-I instance-commitment reliability against "
            "Stage-I and Stage-III POI assignment accuracy."
        )
    )

    parser.add_argument("--dataset", type=str, default="ny_filterPOI")
    parser.add_argument("--stage1", type=str, required=True)
    parser.add_argument("--stage3", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument(
        "--model_variant",
        type=str,
        default="base",
        choices=["base", "no_dist"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=20260727)
    parser.add_argument("--radius_size", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--disable_amp", action="store_true")

    parser.add_argument(
        "--reliability_field",
        type=str,
        default="commitment",
        choices=[
            "commitment",
            "alpha",
            "ce_weight",
            "top1_prob",
            "entropy",
            "margin",
        ],
        help=(
            "commitment uses the method reliability r=alpha*ce_weight. "
            "entropy means 1-normalized_entropy."
        ),
    )
    parser.add_argument(
        "--include_uncommitted",
        type=str2bool,
        default=False,
        help=(
            "False: analyze only steps present in the Stage-I commitment cache. "
            "True: include missing-cache steps with alpha=ce_weight=0."
        ),
    )
    parser.add_argument(
        "--binning",
        type=str,
        default="fixed",
        choices=["fixed", "quantile"],
        help=(
            "fixed uses reliability value intervals 0-.2, .2-.4, ..., .8-1. "
            "quantile creates five equally sized groups."
        ),
    )

    # Keep these identical to the Stage-I commitment configuration.
    parser.add_argument("--commitment_topk", type=int, default=5)
    parser.add_argument(
        "--commitment_use_stability",
        type=str2bool,
        default=False,
    )
    parser.add_argument("--commitment_conf_weight", type=float, default=0.30)
    parser.add_argument("--commitment_margin_weight", type=float, default=0.20)
    parser.add_argument("--commitment_entropy_weight", type=float, default=0.20)
    parser.add_argument("--commitment_time_weight", type=float, default=0.10)
    parser.add_argument(
        "--commitment_main_time_weight",
        type=float,
        default=0.10,
    )
    parser.add_argument("--commitment_recur_weight", type=float, default=0.10)
    parser.add_argument(
        "--commitment_stability_weight",
        type=float,
        default=0.00,
    )
    parser.add_argument("--commitment_bias", type=float, default=0.00)
    parser.add_argument("--gate_threshold", type=float, default=0.35)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_workers != 0:
        print(
            "[Warning] num_workers is forced to 0 for deterministic candidate "
            "sampling and exact Stage-I/Stage-III alignment."
        )
        args.num_workers = 0

    train_lib.set_seed(args.seed)

    conf = train_lib.TrainingConfig()
    conf.batch_size = max(int(args.batch_size), 1)
    conf.num_workers = 0
    conf.pin_memory = False
    conf.persistent_workers = False
    conf.max_seq_len = max(int(args.max_seq_len), 1)
    conf.radius = int(args.radius_size)
    conf.use_amp = (
        (not args.disable_amp)
        and conf.device.type == "cuda"
    )

    paths = train_lib.get_dataset_paths(args.dataset)
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file {name}: {path}")

    print("=" * 72)
    print("Reliability Analysis")
    print("=" * 72)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Dataset            : {args.dataset}")
    print(f"Stage-I checkpoint : {args.stage1}")
    print(f"Stage-III checkpoint: {args.stage3}")
    print(f"Reliability field  : {args.reliability_field}")
    print(f"Binning            : {args.binning}")
    print(f"Include uncommitted: {args.include_uncommitted}")
    print(f"Device             : {conf.device}")
    print("=" * 72)

    inspect_checkpoint(args.stage1)
    inspect_checkpoint(args.stage3)

    poi_config = POIProcessingConfig(
        checkin_file=paths["checkin_file"],
        poi_file=paths["poi_file"],
        dist_file=paths["dist_file"],
        radius=conf.radius,
        max_candidates=conf.max_candidates,
        device=conf.device,
    )
    processor = POIDataProcessor(poi_config)

    # These are the real processor sizes in dataset.py.
    print("[Processor]")
    print(f"  users     = {len(processor.user2idx)}")
    print(f"  POIs      = {len(processor.venue_id2idx)}")
    print(f"  categories= {len(processor.cat2idx)}")
    print(f"  main cats = {processor.num_main_cats}")

    model_config = train_lib.load_model_config(args, processor, conf)

    # Build all three datasets in exactly the same order as train.py so that
    # dataset-side random noisy coordinates follow the training pipeline.
    _, _, test_loader = train_lib.build_dataloaders(processor, conf)

    cached_batches = cache_test_batches(
        test_loader=test_loader,
        seed=args.eval_seed,
    )
    cached_loader = CachedBatchLoader(
        batches=cached_batches,
        dataset=test_loader.dataset,
    )

    # --------------------------------------------------------
    # Stage-I
    # --------------------------------------------------------
    print("\n[Stage-I] Loading main model...")
    stage1_model = load_main_model(
        checkpoint_path=args.stage1,
        model_config=model_config,
        model_variant=args.model_variant,
        device=conf.device,
    )

    print("[Stage-I] Mining commitment pseudo cache with easy_mining.py...")
    pseudo_cache = mine_commitment_pseudo_labels(
        model=stage1_model,
        data_loader=cached_loader,
        conf=conf,
        args=args,
        previous_cache=None,
        source_tag="stage1_reliability_analysis",
    )
    print(f"[Stage-I] Commitment cache entries={len(pseudo_cache)}")

    stage1_df = collect_stage1_records(
        model=stage1_model,
        cached_loader=cached_loader,
        pseudo_cache=pseudo_cache,
        conf=conf,
        reliability_field=args.reliability_field,
        include_uncommitted=bool(args.include_uncommitted),
    )

    del stage1_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Stage-III
    # --------------------------------------------------------
    print("\n[Stage-III] Loading final main model...")
    stage3_model = load_main_model(
        checkpoint_path=args.stage3,
        model_config=model_config,
        model_variant=args.model_variant,
        device=conf.device,
    )

    stage3_df = collect_stage3_records(
        model=stage3_model,
        cached_loader=cached_loader,
        conf=conf,
    )

    del stage3_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Match exact sample/time-step pairs
    # --------------------------------------------------------
    merged = stage1_df.merge(
        stage3_df,
        on=["sample_idx", "time_step"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError(
            "Stage-I and Stage-III records have no matching sample/time-step keys."
        )

    if len(merged) != len(stage1_df):
        print(
            f"[Warning] Only {len(merged)}/{len(stage1_df)} Stage-I records "
            "matched Stage-III outputs."
        )

    merged["reliability"] = (
        merged["reliability"].astype(float).clip(0.0, 1.0)
    )
    groups, bin_edges = assign_reliability_groups(
        merged["reliability"],
        mode=args.binning,
    )
    merged["reliability_group"] = groups

    summary = build_group_summary(merged)
    overall = build_overall_metrics(merged)

    output_dir = Path(
        args.output_dir
        if args.output_dir
        else PROJECT_ROOT
        / "result"
        / args.dataset
        / "reliability_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = output_dir / "reliability_step_details.csv"
    summary_csv = output_dir / "reliability_accuracy_by_group.csv"
    overall_json = output_dir / "reliability_overall_metrics.json"
    config_json = output_dir / "reliability_analysis_config.json"

    merged.sort_values(
        ["reliability", "sample_idx", "time_step"]
    ).to_csv(detail_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    config_payload = {
        "dataset": args.dataset,
        "stage1": args.stage1,
        "stage3": args.stage3,
        "model_variant": args.model_variant,
        "reliability_field": args.reliability_field,
        "include_uncommitted": bool(args.include_uncommitted),
        "binning": args.binning,
        "bin_edges": bin_edges,
        "commitment_topk": args.commitment_topk,
        "commitment_use_stability": bool(
            args.commitment_use_stability
        ),
        "commitment_weights": {
            "confidence": args.commitment_conf_weight,
            "margin": args.commitment_margin_weight,
            "entropy": args.commitment_entropy_weight,
            "time": args.commitment_time_weight,
            "main_time": args.commitment_main_time_weight,
            "recurrence": args.commitment_recur_weight,
            "stability": args.commitment_stability_weight,
            "bias": args.commitment_bias,
        },
        "gate_threshold": args.gate_threshold,
        "num_cached_test_batches": len(cached_batches),
        "num_commitment_cache_entries": len(pseudo_cache),
    }
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    plot_accuracy_by_group(
        summary,
        output_dir / "reliability_stage1_stage3_accuracy.png",
    )
    plot_gain_by_group(
        summary,
        output_dir / "reliability_stage3_gain.png",
    )
    plot_reliability_histogram(
        merged,
        output_dir / "reliability_distribution.png",
    )

    print("\n" + "=" * 72)
    print("Group Summary")
    print("=" * 72)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.float_format",
        lambda x: f"{x:.6f}",
    ):
        print(summary)

    print("\nOverall Metrics")
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\nSaved files:")
    for path in (
        detail_csv,
        summary_csv,
        overall_json,
        config_json,
        output_dir / "reliability_stage1_stage3_accuracy.png",
        output_dir / "reliability_stage3_gain.png",
        output_dir / "reliability_distribution.png",
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
