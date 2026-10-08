"""Run independent PRODEN/PiCO: python -m baselines.train --method proden ..."""
import argparse
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ROOT, build_data, protocol_manifest, seq_collate_fn
from .model import POIAssignmentEncoder, ClassNormalizer
from .algorithms import ConfidenceBank, PiCO, proden_loss_and_update


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("proden", "pico"))
    p.add_argument("--dataset", default="tokyo")
    p.add_argument("--data_root", default=str(ROOT / "dataset"))
    p.add_argument("--embedding_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=42, help="Shared across methods/model seeds.")
    p.add_argument("--radius", type=float, default=200)
    p.add_argument("--max_candidates", type=int, default=50)
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument("--noisy_value", type=float, default=50)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--normalizer", choices=("full", "sampled"), default="full")
    p.add_argument("--class_chunk_size", type=int, default=512)
    p.add_argument("--num_negatives", type=int, default=256,
                   help="Uniform non-candidate classes for the sampled approximation only.")
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--queue_size", type=int, default=8192)
    p.add_argument("--encoder_momentum", type=float, default=0.999)
    p.add_argument("--prototype_momentum", type=float, default=0.99)
    p.add_argument("--contrastive_temperature", type=float, default=0.07)
    p.add_argument("--contrastive_weight", type=float, default=0.5)
    p.add_argument("--confidence_start", type=float, default=0.95)
    p.add_argument("--confidence_end", type=float, default=0.8)
    p.add_argument("--prototype_start", type=int, default=1,
                   help="PiCO's own MoCo-to-prototype switch; 0 uses prototypes from first epoch.")
    p.add_argument("--resume", default=None, help="Trusted baseline last.pt (includes optimizer/PLL state).")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--benchmark_batches", type=int, default=0)
    p.add_argument("--max_train_steps", type=int, default=0, help="Smoke checks only; 0=all batches.")
    p.add_argument("--max_eval_batches", type=int, default=0, help="Smoke checks only; 0=all batches.")
    args = p.parse_args()
    for name in ("max_candidates", "max_seq_len", "embed_dim", "batch_size", "epochs",
                 "class_chunk_size", "num_negatives", "projection_dim", "queue_size"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if args.embed_dim % 4:
        p.error("--embed_dim must be divisible by the shared encoder's four attention heads")
    for name in ("encoder_momentum", "prototype_momentum", "confidence_start", "confidence_end"):
        if not 0 <= getattr(args, name) <= 1:
            p.error(f"--{name} must be in [0,1]")
    if args.contrastive_temperature <= 0 or args.num_workers < 0:
        p.error("Invalid temperature or number of workers")
    if args.eval_only and not args.resume:
        p.error("--eval_only requires --resume")
    return args


def build_optimizer(model, args):
    # CUDA's default foreach path materializes intermediates across all parameters.
    # Large trainable semantic embeddings make that extra peak memory prohibitive.
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=args.learning_rate, weight_decay=args.weight_decay,
                             foreach=False)


def load_optimizer_state(optimizer, state):
    optimizer.load_state_dict(state)
    # Older checkpoints restore foreach=None/True and override the constructor.
    for group in optimizer.param_groups:
        group["foreach"] = False


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def training_inputs(batch):
    # Explicit boundary: true labels are unavailable to every training objective.
    return {key: value for key, value in batch.items() if key not in ("label_pos", "true_poi_id")}


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    counts = dict(acc1=0, acc5=0, acc_cat=0, acc_main_cat=0)
    total = 0
    for i, cpu_batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move(cpu_batch, device)
        valid = batch["seq_mask"].bool()
        scores = model.predict(training_inputs(batch))[valid]
        target = batch["label_pos"][valid]
        top = scores.topk(min(5, scores.size(-1)), dim=-1).indices
        # Do not count padded candidate ranks if fewer than five exist.
        top_valid = batch["cand_mask"][valid].gather(1, top).bool()
        counts["acc1"] += int((top[:, 0] == target).sum())
        counts["acc5"] += int(((top == target[:, None]) & top_valid).any(-1).sum())
        for metric, key in (("acc_cat", "cand_cat_ids"), ("acc_main_cat", "cand_main_cat_ids")):
            cats = batch[key][valid]
            predicted = cats.gather(1, top[:, :1]).squeeze(1)
            truth = cats.gather(1, target[:, None]).squeeze(1)
            counts[metric] += int(((predicted == truth) & (truth != 0)).sum())
        total += len(target)
    if not total:
        raise ValueError("No evaluation instances")
    # Same denominator convention as the project's evaluate_metrics.
    return {**{k: v / total for k, v in counts.items()}, "num_steps": total}


@torch.no_grad()
def benchmark(model, loader, device, max_batches):
    model.eval()
    times, steps = [], 0
    # Three warm-up forwards solely for timing, not a training stage.
    for i, cpu in enumerate(loader):
        if i >= max_batches + 3:
            break
        batch = move(training_inputs(cpu), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= 3:
            times.append(elapsed)
            steps += int(batch["seq_mask"].sum())
    return {"scope": "model.predict only; transfers and loader excluded", "batches": len(times),
            "steps": steps, "seconds": sum(times),
            "ms_per_step": 1000 * sum(times) / steps if steps else None,
            "steps_per_second": steps / sum(times) if times else None}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    folder = Path(args.data_root).resolve() / args.dataset
    embedding_path = Path(args.embedding_path).resolve() if args.embedding_path else folder / "qwen_poi_embeddings.pt"
    required = [folder / name for name in ("filtered_checkin_data.csv", "poi.csv", "main_category.csv",
                                           "category_time_distribution_P_Category_given_Time.csv")] + [embedding_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required original feature files are missing:\n" + "\n".join(missing))
    output = Path(args.output_dir or ROOT / "result" / args.dataset / "pll_baselines" / f"{args.method}_{args.normalizer}" / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "train.log")])
    processor, datasets = build_data(args)
    manifest = protocol_manifest(args)
    # An embedding override is part of the protocol fingerprint as well.
    if args.embedding_path:
        import hashlib
        digest = hashlib.sha256()
        with embedding_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["source_files_sha256"]["qwen_poi_embeddings.pt"] = digest.hexdigest()
    manifest["normalizer"] = args.normalizer
    manifest["num_negatives"] = args.num_negatives
    with (output / "protocol.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    logging.warning("Time-category priors are read unchanged from CSV; verify train-only/external provenance.")
    logging.info("Normalizer=%s; sampled mode is an approximation, not original full-class softmax", args.normalizer)
    loaders = [DataLoader(data, batch_size=args.batch_size, shuffle=(i == 0),
                          num_workers=args.num_workers, collate_fn=seq_collate_fn,
                          pin_memory=device.type == "cuda") for i, data in enumerate(datasets)]
    encoder = POIAssignmentEncoder(processor, embedding_path, args.embed_dim,
                                   args.projection_dim, args.max_seq_len)
    model = encoder if args.method == "proden" else PiCO(
        encoder, processor.num_pois, args.projection_dim, args.queue_size,
        args.encoder_momentum, args.prototype_momentum, args.contrastive_temperature)
    model = model.to(device)
    optimizer = build_optimizer(model, args)
    bank = ConfidenceBank(datasets[0].num_steps, args.max_candidates)
    normalizer = ClassNormalizer(processor, datasets[0], args.normalizer, args.class_chunk_size)
    start, best, bad, best_epoch = 0, -1.0, 0, -1
    smoke_only = bool(args.max_train_steps or args.max_eval_batches)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state["method"] != args.method or state["protocol"] != manifest:
            raise ValueError("Resume method/data/features/normalizer do not match checkpoint")
        model.load_state_dict(state["model"])
        best_epoch = state["best_epoch"]
        source_args = state["args"]
        smoke_only = smoke_only or bool(source_args.get("max_train_steps") or source_args.get("max_eval_batches")) or state.get("smoke_only", False)
        if not args.eval_only:
            load_optimizer_state(optimizer, state["optimizer"])
            bank.load_state_dict(state["confidence"])
            start, best, bad, best_epoch = state["epoch"] + 1, state["best"], state["bad"], state["best_epoch"]
            restore_rng(state["rng"])
    history = []
    history_path = output / "history.jsonl"
    if not args.resume:
        history_path.write_text("")
    for epoch in range(start, args.epochs) if not args.eval_only else ():
        datasets[0].epoch = epoch
        model.train()
        loss_sum, seen = 0.0, 0
        for batch_index, cpu in enumerate(loaders[0]):
            if args.max_train_steps and batch_index >= args.max_train_steps:
                break
            batch = move(training_inputs(cpu), device)
            optimizer.zero_grad(set_to_none=True)
            if args.method == "proden":
                scores, _, hidden = model(batch)
                logz = normalizer.log_partition(model, hidden, scores, batch)
                loss = proden_loss_and_update(scores, logz, batch, bank)
            else:
                loss, _ = model.loss(batch, bank, normalizer, epoch, args.epochs, args.prototype_start,
                                      args.confidence_start, args.confidence_end, args.contrastive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, batch={batch_index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            steps = int(batch["seq_mask"].sum())
            loss_sum += loss.item() * steps
            seen += steps
        metrics = evaluate(model, loaders[1], device, args.max_eval_batches)
        row = {"epoch": epoch, "loss": loss_sum / max(seen, 1), "train_steps": seen, "val": metrics}
        history.append(row)
        with history_path.open("a") as file:
            file.write(json.dumps(row) + "\n")
        logging.info("epoch=%d loss=%.6f val=%s", epoch, row["loss"], metrics)
        improved = metrics["acc1"] > best
        if improved:
            best, bad, best_epoch = metrics["acc1"], 0, epoch
        else:
            bad += 1
        state = {"method": args.method, "args": vars(args), "protocol": manifest, "epoch": epoch,
                 "smoke_only": smoke_only,
                 "best": best, "best_epoch": best_epoch, "bad": bad, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "confidence": bank.state_dict(), "rng": rng_state()}
        torch.save(state, output / "last.pt")
        if improved:
            # Evaluation checkpoint excludes the large confidence bank/optimizer.
            torch.save({k: v for k, v in state.items() if k not in ("confidence", "optimizer", "rng")},
                       output / "best.pt")
        if args.patience > 0 and bad >= args.patience:
            break
    if not args.eval_only:
        best_path = output / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint; resume to original output_dir or increase --epochs")
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        best_epoch = state["best_epoch"]
    result = {"method": args.method, "dataset": args.dataset, "seed": args.seed,
              "normalizer": args.normalizer, "best_epoch": best_epoch, "protocol": manifest,
              "validation": evaluate(model, loaders[1], device, args.max_eval_batches),
              "test": evaluate(model, loaders[2], device, args.max_eval_batches),
              "smoke_only": smoke_only,
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if args.benchmark_batches:
        result["inference"] = benchmark(model, loaders[2], device, args.benchmark_batches)
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(arguments())
