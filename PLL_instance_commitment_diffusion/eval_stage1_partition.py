import argparse
import logging
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import (
    POIDataProcessor,
    POIProcessingConfig,
    CheckinSequenceDataset,
    seq_collate_fn,
)
from easy_mining import extract_easy_pseudo_labels
from train import (
    TrainingConfig,
    set_seed,
    get_dataset_paths,
    load_model_config,
    build_stage1_model,
    move_batch_to_device,
    get_autocast_context,
)
from utils import init_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate stage1 partition metrics and hard buckets.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_variant", type=str, default="base", choices=["base", "no_dist"])
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--radius_size", type=int, default=200)
    parser.add_argument("--easy_prob_thresh", type=float, default=0.90)
    parser.add_argument("--easy_entropy_thresh", type=float, default=0.10)
    parser.add_argument("--easy_min_valid_candidates", type=int, default=2)
    parser.add_argument("--easy_soft_topk", type=int, default=5)
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloaders(conf: TrainingConfig, processor: POIDataProcessor):
    splits = {}
    for mode in ("train", "val", "test"):
        dataset = CheckinSequenceDataset(processor, max_seq_len=conf.max_seq_len, mode=mode)
        loader = DataLoader(
            dataset,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=conf.num_workers,
            collate_fn=seq_collate_fn,
        )
        splits[mode] = loader
    return splits


def build_partition_mask(batch, easy_cache: Dict[tuple, dict]):
    seq_mask = batch["seq_mask"].bool()
    easy_mask = torch.zeros_like(seq_mask, dtype=torch.bool)
    sample_indices = batch["sample_idx"].tolist()
    for b, sample_idx in enumerate(sample_indices):
        valid_len = int(seq_mask[b].sum().item())
        for s in range(valid_len):
            if (int(sample_idx), int(s)) in easy_cache:
                easy_mask[b, s] = True
    return easy_mask


def count_maincat_choices(cand_main_cat_ids: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = cand_main_cat_ids.shape
    counts = torch.zeros((batch_size, seq_len), dtype=torch.long, device=cand_main_cat_ids.device)
    valid_maincats = cand_main_cat_ids.masked_fill(~cand_mask.bool(), 0)
    for b in range(batch_size):
        for t in range(seq_len):
            vals = valid_maincats[b, t]
            vals = vals[vals > 0]
            if vals.numel() == 0:
                counts[b, t] = 0
            else:
                counts[b, t] = torch.unique(vals).numel()
    return counts


def build_hard_bucket_masks(batch, hard_mask: torch.Tensor):
    unique_maincat_counts = count_maincat_choices(batch["cand_main_cat_ids"], batch["cand_mask"])
    cross_mask = hard_mask & (unique_maincat_counts > 1)
    same_mask = hard_mask & (unique_maincat_counts <= 1)
    return cross_mask, same_mask


def init_metric_state():
    return {
        "steps": 0,
        "top1": 0,
        "top5": 0,
        "cat": 0,
        "maincat": 0,
        "top1_wrong": 0,
    }


def update_metric_state(state, top1_preds, top5_preds, batch, mask):
    flat_mask = mask.view(-1)
    if not flat_mask.any():
        return

    labels_flat = batch["label_pos"].view(-1)[flat_mask]
    top1_flat = top1_preds.view(-1)[flat_mask]
    top5_flat = top5_preds.view(-1, top5_preds.size(-1))[flat_mask]
    if labels_flat.numel() == 0:
        return

    state["steps"] += int(labels_flat.numel())
    top1_ok = (top1_flat == labels_flat)
    top5_ok = top5_flat.eq(labels_flat.unsqueeze(-1)).any(dim=-1)

    cand_cats_flat = batch["cand_cat_ids"].view(-1, batch["cand_cat_ids"].size(-1))[flat_mask]
    pred_cat = torch.gather(cand_cats_flat, 1, top1_flat.unsqueeze(1)).squeeze(1)
    true_cat = torch.gather(cand_cats_flat, 1, labels_flat.unsqueeze(1)).squeeze(1)
    cat_ok = pred_cat == true_cat

    cand_main_flat = batch["cand_main_cat_ids"].view(-1, batch["cand_main_cat_ids"].size(-1))[flat_mask]
    pred_main = torch.gather(cand_main_flat, 1, top1_flat.unsqueeze(1)).squeeze(1)
    true_main = torch.gather(cand_main_flat, 1, labels_flat.unsqueeze(1)).squeeze(1)
    maincat_ok = pred_main == true_main

    state["top1"] += int(top1_ok.sum().item())
    state["top5"] += int(top5_ok.sum().item())
    state["cat"] += int(cat_ok.sum().item())
    state["maincat"] += int(maincat_ok.sum().item())
    state["top1_wrong"] += int((~top1_ok).sum().item())


def finalize_metric_state(state):
    steps = max(state["steps"], 1)
    top1 = state["top1"] / steps
    top5 = state["top5"] / steps
    cat = state["cat"] / steps
    maincat = state["maincat"] / steps
    return {
        "steps": state["steps"],
        "acc1": top1,
        "acc5": top5,
        "acc_cat": cat,
        "acc_maincat": maincat,
        "err1": 1.0 - top1,
        "err5": 1.0 - top5,
        "err_cat": 1.0 - cat,
        "err_maincat": 1.0 - maincat,
        "top1_wrong": state["top1_wrong"],
        "wrong_share_at1": (state["top1_wrong"] / steps),
    }


@torch.no_grad()
def evaluate_stage1_partition(model, dataloader, easy_cache: Dict[tuple, dict], device: torch.device, use_amp=False):
    model.eval()
    overall = init_metric_state()
    easy = init_metric_state()
    hard = init_metric_state()
    hard_cross = init_metric_state()
    hard_same = init_metric_state()

    for batch in dataloader:
        batch = move_batch_to_device(batch, device, non_blocking=(device.type == "cuda"))
        with get_autocast_context(device, use_amp):
            logits = model.predict(batch)
        top5 = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
        top1 = top5[..., 0]

        seq_mask = batch["seq_mask"].bool()
        easy_mask = build_partition_mask(batch, easy_cache) & seq_mask
        hard_mask = (~easy_mask) & seq_mask
        cross_mask, same_mask = build_hard_bucket_masks(batch, hard_mask)

        update_metric_state(overall, top1, top5, batch, seq_mask)
        update_metric_state(easy, top1, top5, batch, easy_mask)
        update_metric_state(hard, top1, top5, batch, hard_mask)
        update_metric_state(hard_cross, top1, top5, batch, cross_mask)
        update_metric_state(hard_same, top1, top5, batch, same_mask)

    return {
        "overall": finalize_metric_state(overall),
        "easy": finalize_metric_state(easy),
        "hard": finalize_metric_state(hard),
        "hard_cross_maincat": finalize_metric_state(hard_cross),
        "hard_same_maincat": finalize_metric_state(hard_same),
    }


def log_partition_metrics(split_name: str, metrics: Dict[str, Dict[str, float]]):
    overall = metrics["overall"]
    easy = metrics["easy"]
    hard = metrics["hard"]
    logging.info(
        "[Eval] [%s] Global Acc@1=%.4f | Acc@5=%.4f | Acc@Cat=%.4f | Acc@MainCat=%.4f | "
        "Easy Acc@1=%.4f | Acc@5=%.4f | Acc@Cat=%.4f | Acc@MainCat=%.4f | "
        "Hard Acc@1=%.4f | Acc@5=%.4f | Acc@Cat=%.4f | Acc@MainCat=%.4f",
        split_name.upper(),
        overall["acc1"], overall["acc5"], overall["acc_cat"], overall["acc_maincat"],
        easy["acc1"], easy["acc5"], easy["acc_cat"], easy["acc_maincat"],
        hard["acc1"], hard["acc5"], hard["acc_cat"], hard["acc_maincat"],
    )


def log_hard_bucket_metrics(split_name: str, metrics: Dict[str, Dict[str, float]]):
    hard_steps = max(metrics["hard"]["steps"], 1)
    for tag, key in (("CROSS_MAINCAT", "hard_cross_maincat"), ("SAME_MAINCAT", "hard_same_maincat")):
        item = metrics[key]
        logging.info(
            "[Eval] [%s] HardBucket[%s] Steps=%d | HardShare=%.4f | Top1Wrong=%d | WrongShare@1=%.4f | "
            "Acc@1=%.4f | Acc@5=%.4f | Acc@Cat=%.4f | Acc@MainCat=%.4f | "
            "Err@1=%.4f | Err@5=%.4f | Err@Cat=%.4f | Err@MainCat=%.4f",
            split_name.upper(),
            tag,
            item["steps"],
            item["steps"] / hard_steps,
            item["top1_wrong"],
            item["wrong_share_at1"],
            item["acc1"],
            item["acc5"],
            item["acc_cat"],
            item["acc_maincat"],
            item["err1"],
            item["err5"],
            item["err_cat"],
            item["err_maincat"],
        )


def main():
    args = parse_args()
    conf = TrainingConfig()
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.radius = args.radius_size
    conf.easy_prob_thresh = args.easy_prob_thresh
    conf.easy_entropy_thresh = args.easy_entropy_thresh
    conf.easy_min_valid_candidates = args.easy_min_valid_candidates
    conf.easy_soft_topk = args.easy_soft_topk
    conf.model_variant = args.model_variant

    set_seed(args.seed)

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    init_logging(args.log_file if args.log_file else None, force=True)

    paths = get_dataset_paths(args.dataset)
    poi_config = POIProcessingConfig(
        checkin_file=paths["checkin_file"],
        poi_file=paths["poi_file"],
        dist_file=paths["dist_file"],
        radius=args.radius_size,
        max_candidates=conf.max_candidates,
        device=conf.device,
    )
    processor = POIDataProcessor(poi_config)
    dataloaders = build_dataloaders(conf, processor)
    device = torch.device(conf.device)

    model_config = load_model_config(args, processor, conf)
    model = build_stage1_model(model_config, args.model_variant).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    for split_name, loader in dataloaders.items():
        easy_cache = extract_easy_pseudo_labels(
            model=model,
            data_loader=loader,
            processor=processor,
            conf=conf,
            prob_thresh=conf.easy_prob_thresh,
            entropy_thresh=conf.easy_entropy_thresh,
            min_valid_candidates=conf.easy_min_valid_candidates,
            soft_topk=conf.easy_soft_topk,
            source_tag=f"stage1_{split_name}",
        )
        metrics = evaluate_stage1_partition(model, loader, easy_cache, device, use_amp=conf.use_amp)
        log_partition_metrics(split_name, metrics)
        log_hard_bucket_metrics(split_name, metrics)


if __name__ == "__main__":
    main()
