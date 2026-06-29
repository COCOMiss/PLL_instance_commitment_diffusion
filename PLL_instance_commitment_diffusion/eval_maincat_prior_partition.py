import argparse
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import init_logging, get_logger
from dataset import (
    POIDataProcessor,
    POIProcessingConfig,
    CheckinSequenceDataset,
    seq_collate_fn,
)
from model_001 import TrajPOITransformer
from model_nodist import TrajPOITransformerNoDist
from easy_mining import extract_easy_pseudo_labels


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

logger = get_logger("MainCatPriorEval")


def get_dataset_paths(dataset_name):
    data_dir = os.path.join(PROJECT_ROOT, "dataset", dataset_name)
    return {
        "checkin_file": os.path.join(data_dir, "filtered_checkin_data.csv"),
        "poi_file": os.path.join(data_dir, "poi.csv"),
        "dist_file": os.path.join(data_dir, "category_time_distribution_P_Category_given_Time.csv"),
    }


class EvalConfig:
    radius = 200
    max_candidates = 50
    max_seq_len = 20
    batch_size = 16
    seed = 42
    embed_dim = 64
    num_workers = 4
    pin_memory = False
    persistent_workers = False
    use_amp = True
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_dtype():
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def get_autocast_context(device, use_amp):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=get_amp_dtype())
    return nullcontext()


def move_batch_to_device(batch, device, non_blocking=False):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=non_blocking)
    return batch


def build_loader_kwargs(conf):
    kwargs = {
        "batch_size": conf.batch_size,
        "collate_fn": seq_collate_fn,
        "num_workers": conf.num_workers,
        "pin_memory": conf.pin_memory,
    }
    if conf.num_workers > 0:
        kwargs["persistent_workers"] = conf.persistent_workers
    return kwargs


def build_dataloaders(processor, conf):
    train_dataset = CheckinSequenceDataset(processor, max_seq_len=conf.max_seq_len, mode="train")
    val_dataset = CheckinSequenceDataset(processor, max_seq_len=conf.max_seq_len, mode="val")
    test_dataset = CheckinSequenceDataset(processor, max_seq_len=conf.max_seq_len, mode="test")
    kwargs = build_loader_kwargs(conf)
    return {
        "train": DataLoader(train_dataset, shuffle=False, **kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **kwargs),
    }


def load_model_config(args, processor, conf):
    class ModelConfig:
        dataset_name = args.dataset
        user2idx = processor.user2idx
        venue_id2idx = processor.venue_id2idx
        cat2idx = processor.cat2idx
        embed_dim = conf.embed_dim
        num_main_cats = processor.num_main_cats
        num_cats = processor.num_cats

    return ModelConfig


def build_stage1_model(model_config, model_variant):
    if model_variant == "no_dist":
        return TrajPOITransformerNoDist(model_config)
    return TrajPOITransformer(model_config)


def init_metric_totals():
    return {
        "correct_1": 0,
        "correct_5": 0,
        "correct_cat": 0,
        "correct_main_cat": 0,
        "num_steps": 0,
    }


def update_metric_totals(totals, scores, batch, mask):
    flat_mask = mask.view(-1)
    if not flat_mask.any():
        return

    scores_flat = scores.view(-1, scores.size(-1))[flat_mask]
    labels_flat = batch["label_pos"].view(-1)[flat_mask]
    if labels_flat.numel() == 0:
        return

    topk = min(5, scores_flat.size(-1))
    _, topk_indices = torch.topk(scores_flat, k=topk, dim=1)
    top1_preds = topk_indices[:, 0]

    totals["correct_1"] += int((top1_preds == labels_flat).sum().item())
    totals["correct_5"] += int((topk_indices == labels_flat.unsqueeze(1)).any(dim=1).sum().item())

    cand_cats_flat = batch["cand_cat_ids"].view(-1, batch["cand_cat_ids"].size(-1))[flat_mask]
    pred_cats = torch.gather(cand_cats_flat, 1, top1_preds.unsqueeze(1)).squeeze(1)
    gt_cats = torch.gather(cand_cats_flat, 1, labels_flat.unsqueeze(1)).squeeze(1)
    valid_cat_mask = gt_cats != 0
    if valid_cat_mask.any():
        totals["correct_cat"] += int((pred_cats[valid_cat_mask] == gt_cats[valid_cat_mask]).sum().item())

    cand_main_flat = batch["cand_main_cat_ids"].view(-1, batch["cand_main_cat_ids"].size(-1))[flat_mask]
    pred_main = torch.gather(cand_main_flat, 1, top1_preds.unsqueeze(1)).squeeze(1)
    gt_main = torch.gather(cand_main_flat, 1, labels_flat.unsqueeze(1)).squeeze(1)
    valid_main_mask = gt_main != 0
    if valid_main_mask.any():
        totals["correct_main_cat"] += int((pred_main[valid_main_mask] == gt_main[valid_main_mask]).sum().item())

    totals["num_steps"] += int(labels_flat.numel())


def finalize_metric_totals(totals):
    num_steps = int(totals["num_steps"])
    if num_steps == 0:
        return {
            "acc1": 0.0,
            "acc5": 0.0,
            "acc_cat": 0.0,
            "acc_main_cat": 0.0,
            "num_steps": 0,
        }
    return {
        "acc1": totals["correct_1"] / num_steps,
        "acc5": totals["correct_5"] / num_steps,
        "acc_cat": totals["correct_cat"] / num_steps,
        "acc_main_cat": totals["correct_main_cat"] / num_steps,
        "num_steps": num_steps,
    }


def build_partition_mask(batch, easy_cache):
    seq_mask = batch["seq_mask"].bool()
    easy_mask = torch.zeros_like(seq_mask, dtype=torch.bool)
    sample_indices = batch["sample_idx"].tolist()
    for b, sample_idx in enumerate(sample_indices):
        valid_len = int(seq_mask[b].sum().item())
        for s in range(valid_len):
            if (int(sample_idx), int(s)) in easy_cache:
                easy_mask[b, s] = True
    hard_mask = seq_mask & (~easy_mask)
    return easy_mask, hard_mask


def evaluate_maincat_prior_only(data_loader, easy_cache, device):
    overall = init_metric_totals()
    easy = init_metric_totals()
    hard = init_metric_totals()

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device, non_blocking=(device.type == "cuda"))
            scores = batch["cand_main_cat_probs"].float().masked_fill(batch["cand_mask"] == 0, -1e9)
            seq_mask = batch["seq_mask"].bool()
            easy_mask, hard_mask = build_partition_mask(batch, easy_cache)

            update_metric_totals(overall, scores, batch, seq_mask)
            update_metric_totals(easy, scores, batch, easy_mask)
            update_metric_totals(hard, scores, batch, hard_mask)

    return {
        "overall": finalize_metric_totals(overall),
        "easy": finalize_metric_totals(easy),
        "hard": finalize_metric_totals(hard),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate main-cat prior-only baseline under stage1 easy/hard partition")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_variant", type=str, default="base", choices=["base", "no_dist"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--radius_size", type=int, default=200)
    parser.add_argument("--easy_prob_thresh", type=float, default=0.75)
    parser.add_argument("--easy_entropy_thresh", type=float, default=0.40)
    parser.add_argument("--easy_min_valid_candidates", type=int, default=2)
    parser.add_argument("--easy_soft_topk", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    result_dir = os.path.dirname(args.checkpoint)
    log_file = args.log_file or os.path.join(result_dir, "maincat_prior_partition_eval.log")
    if os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    global logger
    init_logging(log_file=log_file, force=True)
    logger = get_logger("MainCatPriorEval")

    conf = EvalConfig()
    conf.batch_size = args.batch_size
    conf.num_workers = max(int(args.num_workers), 0)
    conf.pin_memory = (conf.device.type == "cuda")
    conf.persistent_workers = conf.num_workers > 0
    conf.use_amp = (not args.disable_amp) and (conf.device.type == "cuda")
    conf.radius = args.radius_size

    paths = get_dataset_paths(args.dataset)
    poi_config = POIProcessingConfig(
        checkin_file=paths["checkin_file"],
        poi_file=paths["poi_file"],
        dist_file=paths["dist_file"],
        radius=conf.radius,
        max_candidates=conf.max_candidates,
        device=conf.device,
    )
    set_seed(conf.seed)
    processor = POIDataProcessor(poi_config)
    model_config = load_model_config(args, processor, conf)
    model = build_stage1_model(model_config, args.model_variant).to(conf.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=conf.device))
    model.eval()

    loaders = build_dataloaders(processor, conf)
    logger.info(
        f"[Config] dataset={args.dataset} | checkpoint={args.checkpoint} | "
        f"model_variant={args.model_variant} | batch_size={conf.batch_size} | num_workers={conf.num_workers}"
    )

    with get_autocast_context(conf.device, conf.use_amp):
        pass

    for split_name, loader in loaders.items():
        easy_cache = extract_easy_pseudo_labels(
            model=model,
            data_loader=loader,
            processor=processor,
            conf=conf,
            prob_thresh=args.easy_prob_thresh,
            entropy_thresh=args.easy_entropy_thresh,
            min_valid_candidates=args.easy_min_valid_candidates,
            soft_topk=args.easy_soft_topk,
            source_tag=f"{split_name}_stage1_partition",
        )
        metrics = evaluate_maincat_prior_only(loader, easy_cache, conf.device)
        logger.info(
            f"[MainCatPriorOnly][{split_name.upper()}] "
            f"Global Acc@1={metrics['overall']['acc1']:.4f}, Acc@5={metrics['overall']['acc5']:.4f}, "
            f"Acc@Cat={metrics['overall']['acc_cat']:.4f}, Acc@MainCat={metrics['overall']['acc_main_cat']:.4f} | "
            f"Easy Acc@1={metrics['easy']['acc1']:.4f}, Acc@5={metrics['easy']['acc5']:.4f}, "
            f"Acc@Cat={metrics['easy']['acc_cat']:.4f}, Acc@MainCat={metrics['easy']['acc_main_cat']:.4f} | "
            f"Hard Acc@1={metrics['hard']['acc1']:.4f}, Acc@5={metrics['hard']['acc5']:.4f}, "
            f"Acc@Cat={metrics['hard']['acc_cat']:.4f}, Acc@MainCat={metrics['hard']['acc_main_cat']:.4f} | "
            f"GlobalSteps={metrics['overall']['num_steps']}, EasySteps={metrics['easy']['num_steps']}, "
            f"HardSteps={metrics['hard']['num_steps']}"
        )


if __name__ == "__main__":
    main()
