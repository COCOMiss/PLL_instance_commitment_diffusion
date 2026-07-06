import argparse
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    CheckinSequenceDataset,
    POIDataProcessor,
    POIProcessingConfig,
    seq_collate_fn,
)
from easy_mining import mine_commitment_pseudo_labels
from model import TrajPOITransformer
from model_nodist import TrajPOITransformerNoDist
from diffusion_regularizer import InstanceConditionedDiffusionSharpening
from utils import get_logger, init_logging


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

DEFAULT_DATASET = "gowalla_nv"
DEFAULT_EXP_NAME = "vae_commitment"

logger = get_logger("VAE-COMMITMENT")


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_dataset_paths(dataset_name):
    data_dir = os.path.join(PROJECT_ROOT, "dataset", dataset_name)
    return {
        "checkin_file": os.path.join(data_dir, "filtered_checkin_data.csv"),
        "poi_file": os.path.join(data_dir, "poi.csv"),
        "dist_file": os.path.join(data_dir, "category_time_distribution_P_Category_given_Time.csv"),
    }


class TrainingConfig:
    checkin_file = None
    poi_file = None
    dist_file = None
    radius = 200
    max_candidates = 50
    max_seq_len = 20
    batch_size = 16
    weight_decay = 1e-4
    seed = 42
    patience = 3
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


def move_batch_to_device(batch, device, non_blocking=False):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=non_blocking)
    return batch


def get_amp_dtype():
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def get_autocast_context(device, use_amp):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=get_amp_dtype())
    return nullcontext()


@torch.no_grad()
def evaluate_metrics(model, data_loader, device, use_amp=False):
    model.eval()
    correct_1 = 0
    correct_5 = 0
    correct_cat = 0
    correct_main_cat = 0
    total_valid_steps = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device, non_blocking=(device.type == "cuda"))
        with get_autocast_context(device, use_amp):
            scores = model.predict(batch)

        mask = batch["seq_mask"].bool().view(-1)
        scores_flat = scores.view(-1, scores.size(-1))[mask]
        labels_flat = batch["label_pos"].view(-1)[mask]
        if len(labels_flat) == 0:
            continue

        topk = min(5, scores_flat.size(-1))
        _, topk_indices = torch.topk(scores_flat, k=topk, dim=1)
        top1_preds = topk_indices[:, 0]

        correct_1 += (top1_preds == labels_flat).sum().item()
        correct_5 += (topk_indices == labels_flat.unsqueeze(1)).any(dim=1).sum().item()

        cand_cats_flat = batch["cand_cat_ids"].view(-1, batch["cand_cat_ids"].size(-1))[mask]
        pred_cats = torch.gather(cand_cats_flat, 1, top1_preds.unsqueeze(1)).squeeze(1)
        gt_cats = torch.gather(cand_cats_flat, 1, labels_flat.unsqueeze(1)).squeeze(1)
        valid_cat_mask = gt_cats != 0
        if valid_cat_mask.any():
            correct_cat += (pred_cats[valid_cat_mask] == gt_cats[valid_cat_mask]).sum().item()

        cand_main_cats_flat = batch["cand_main_cat_ids"].view(-1, batch["cand_main_cat_ids"].size(-1))[mask]
        pred_main_cats = torch.gather(cand_main_cats_flat, 1, top1_preds.unsqueeze(1)).squeeze(1)
        gt_main_cats = torch.gather(cand_main_cats_flat, 1, labels_flat.unsqueeze(1)).squeeze(1)
        valid_main_mask = gt_main_cats != 0
        if valid_main_mask.any():
            correct_main_cat += (pred_main_cats[valid_main_mask] == gt_main_cats[valid_main_mask]).sum().item()

        total_valid_steps += len(labels_flat)

    if total_valid_steps == 0:
        return {
            "acc1": 0.0,
            "acc5": 0.0,
            "acc_cat": 0.0,
            "acc_main_cat": 0.0,
            "num_steps": 0,
        }

    return {
        "acc1": correct_1 / total_valid_steps,
        "acc5": correct_5 / total_valid_steps,
        "acc_cat": correct_cat / total_valid_steps,
        "acc_main_cat": correct_main_cat / total_valid_steps,
        "num_steps": int(total_valid_steps),
    }


class EarlyStopping:
    def __init__(self, patience=5, path=None):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        score = val_acc
        improved = False
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            improved = True
            return improved

        if score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            improved = True
        return improved

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def build_dataloaders(processor, conf):
    train_dataset = CheckinSequenceDataset(processor, max_seq_len=conf.max_seq_len, mode="train")
    val_dataset = CheckinSequenceDataset(processor, max_seq_len=conf.max_seq_len, mode="val")
    test_dataset = CheckinSequenceDataset(processor, max_seq_len=conf.max_seq_len, mode="test")

    loader_kwargs = {
        "batch_size": conf.batch_size,
        "collate_fn": seq_collate_fn,
        "num_workers": conf.num_workers,
        "pin_memory": conf.pin_memory,
    }
    if conf.num_workers > 0:
        loader_kwargs["persistent_workers"] = conf.persistent_workers

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


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


def build_model(model_config, model_variant):
    if model_variant == "no_dist":
        if TrajPOITransformerNoDist is None:
            raise ImportError(
                "model_variant='no_dist' requires model_nodist.py, but it was not found. "
                "Use --model_variant base or put model_nodist.py in the same directory."
            )
        logger.info("Building model with no distance information")
        return TrajPOITransformerNoDist(model_config)
    logger.info("Building model with distance information")
    return TrajPOITransformer(model_config)


def build_diffusion_regularizer(args, conf):
    phase = getattr(args, "training_phase", "phase1_base")
    needs_diffusion = bool(args.use_diffusion) or phase in {"phase2_diffusion", "phase3_dce", "phase3_joint_dce"}
    if not needs_diffusion:
        return None
    return InstanceConditionedDiffusionSharpening(
        max_candidates=conf.max_candidates,
        hidden_dim=int(args.diffusion_hidden_dim),
        condition_dim=conf.embed_dim if bool(args.diffusion_use_condition) else None,
        num_layers=int(args.diffusion_layers),
        num_heads=int(args.diffusion_heads),
        dropout=float(args.diffusion_dropout),
        temp_min=float(args.diffusion_temp_min),
        temp_max=float(args.diffusion_temp_max),
        entropy_weight=float(args.diffusion_entropy_weight),
        margin_weight=float(args.diffusion_margin_weight),
        target_margin=float(args.diffusion_target_margin),
        mask_prob=float(args.diffusion_mask_prob),
        detach_condition=bool(args.diffusion_detach_condition),
        refine_steps=int(args.diffusion_refine_steps),
        target_soft_mix=float(args.diffusion_target_soft_mix),
        mask_condition_on_masked_steps=bool(args.diffusion_mask_condition),
        ctx_temperature=float(args.diffusion_ctx_temperature),
        context_mix_max=float(args.diffusion_context_mix_max),
        context_loss_weight=float(args.diffusion_context_loss_weight),
        context_anchor_min_weight=float(args.diffusion_context_anchor_min_weight),
        teacher_temp_min=float(args.diffusion_teacher_temp_min),
        teacher_temp_max=float(args.diffusion_teacher_temp_max),
        reverse_kl_weight=float(args.diffusion_reverse_kl_weight),
        input_noise_min=float(args.diffusion_input_noise_min),
        input_noise_max=float(args.diffusion_input_noise_max),
        quality_gate_mode=str(args.diffusion_quality_gate_mode),
        gate_entropy_threshold=float(args.diffusion_gate_entropy_threshold),
        gate_margin_threshold=float(args.diffusion_gate_margin_threshold),
    ).to(conf.device)


def _build_step_soft_target(candidate_poi_ids, soft_poi_ids, soft_probs, device, dtype):
    target_dist = torch.zeros_like(candidate_poi_ids, dtype=dtype, device=device)
    target_mass = torch.zeros((), dtype=dtype, device=device)

    for poi_id, prob in zip(soft_poi_ids, soft_probs):
        matches = torch.where(candidate_poi_ids == int(poi_id))[0]
        if matches.numel() == 0:
            continue
        target_dist[matches[0]] = float(prob)
        target_mass = target_mass + float(prob)

    if float(target_mass.item()) <= 0.0:
        return None
    return target_dist / target_mass.clamp_min(1e-12)


def build_commitment_targets_from_cache(batch, cand_scores, pseudo_cache):
    """Build q_soft, soft CE loss, valid mask, and cache statistics once.

    This replaces the previous separate soft-target and map builders so that
    soft CE and diffusion use exactly the same q_soft and instance gate.
    """
    batch_size, seq_len = batch["seq_mask"].shape
    device = cand_scores.device
    dtype = cand_scores.dtype

    q_soft = torch.zeros_like(cand_scores, dtype=dtype, device=device)
    ce_step_loss = cand_scores.new_zeros((batch_size, seq_len))
    ce_valid_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    alpha_map = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    ce_weight_map = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    top1_prob_map = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    entropy_map = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    margin_map = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    cache_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    if not pseudo_cache:
        return {
            "q_soft": q_soft,
            "ce_step_loss": ce_step_loss,
            "ce_valid_mask": ce_valid_mask,
            "alpha": alpha_map,
            "ce_weight": ce_weight_map,
            "top1_prob": top1_prob_map,
            "normalized_entropy": entropy_map,
            "margin": margin_map,
            "cache_mask": cache_mask,
        }

    masked_scores = cand_scores.float().masked_fill(batch["cand_mask"] == 0, -1e9)
    log_probs = F.log_softmax(masked_scores, dim=-1)

    for b in range(batch_size):
        sample_idx = int(batch["sample_idx"][b].item())
        valid_len = int(batch["seq_mask"][b].sum().item())
        for t in range(valid_len):
            info = pseudo_cache.get((sample_idx, t))
            if not info:
                continue

            cache_mask[b, t] = True
            alpha_map[b, t] = float(info.get("alpha", 0.0))
            ce_weight_map[b, t] = float(info.get("ce_weight", 0.0))
            top1_prob_map[b, t] = float(info.get("top1_prob", 0.0))
            entropy_map[b, t] = float(info.get("normalized_entropy", 0.0))
            margin_map[b, t] = float(info.get("margin", 0.0))

            soft_poi_ids = info.get("soft_poi_ids")
            soft_probs = info.get("soft_probs")
            if not soft_poi_ids or not soft_probs:
                continue

            target_dist = _build_step_soft_target(
                candidate_poi_ids=batch["cand_poi_ids"][b, t],
                soft_poi_ids=soft_poi_ids,
                soft_probs=soft_probs,
                device=device,
                dtype=dtype,
            )
            if target_dist is None:
                continue

            q_soft[b, t] = target_dist
            ce_step_loss[b, t] = -(target_dist.float() * log_probs[b, t]).sum()
            ce_valid_mask[b, t] = True

    return {
        "q_soft": q_soft,
        "ce_step_loss": ce_step_loss,
        "ce_valid_mask": ce_valid_mask,
        "alpha": alpha_map,
        "ce_weight": ce_weight_map,
        "top1_prob": top1_prob_map,
        "normalized_entropy": entropy_map,
        "margin": margin_map,
        "cache_mask": cache_mask,
    }


def compute_alpha_histogram(alpha_values):
    total = max(int(alpha_values.numel()), 1)
    bins = [
        ((alpha_values >= 0.0) & (alpha_values < 0.2)).sum().item(),
        ((alpha_values >= 0.2) & (alpha_values < 0.5)).sum().item(),
        ((alpha_values >= 0.5) & (alpha_values < 0.8)).sum().item(),
        ((alpha_values >= 0.8) & (alpha_values <= 1.0)).sum().item(),
    ]
    return [count / total for count in bins]




def set_requires_grad(module, flag: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(flag)


def _extract_state_dict(state, key):
    if isinstance(state, dict) and key in state:
        return state[key]
    return None


def load_checkpoint_components(path, model, diffusion=None, device=None, strict_model=False, strict_diffusion=False, logger=None):
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    loaded = []

    model_state = _extract_state_dict(state, "model_state_dict")
    diffusion_state = _extract_state_dict(state, "diffusion_state_dict")

    if model_state is None and diffusion_state is None:
        # Backward compatibility: a plain model.state_dict().
        missing, unexpected = model.load_state_dict(state, strict=strict_model)
        loaded.append(f"model_plain(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        if model_state is not None:
            missing, unexpected = model.load_state_dict(model_state, strict=strict_model)
            loaded.append(f"model(missing={len(missing)}, unexpected={len(unexpected)})")
        if diffusion is not None and diffusion_state is not None:
            missing, unexpected = diffusion.load_state_dict(diffusion_state, strict=strict_diffusion)
            loaded.append(f"diffusion(missing={len(missing)}, unexpected={len(unexpected)})")

    if logger is not None:
        logger.info(f"[Checkpoint] Loaded {path} | " + "; ".join(loaded))


def save_checkpoint_components(path, model, diffusion=None, phase=None, score=None):
    state = {
        "model_state_dict": model.state_dict(),
        "phase": phase,
        "score": score,
    }
    if diffusion is not None:
        state["diffusion_state_dict"] = diffusion.state_dict()
    torch.save(state, path)


def run_training(model, train_loader, val_loader, test_loader, conf, args, checkpoint_path):
    total_epochs = max(int(args.epochs), 1)
    phase = str(getattr(args, "training_phase", "phase1_base"))
    if phase not in {"phase1_base", "phase2_diffusion", "phase3_dce", "phase3_joint_dce", "phase3_ic_dce"}:
        raise ValueError(f"Unknown training_phase={phase}")

    diffusion_regularizer = build_diffusion_regularizer(args, conf)

    # Load diffusion weights if provided. Model weights are still loaded in main()
    # for backward compatibility; this call can load both model and diffusion from
    # a combined checkpoint if desired.
    if getattr(args, "diffusion_checkpoint", None):
        load_checkpoint_components(
            args.diffusion_checkpoint,
            model=model,
            diffusion=diffusion_regularizer,
            device=conf.device,
            strict_model=False,
            strict_diffusion=False,
            logger=logger,
        )

    # Phase-specific freezing.
    if phase == "phase1_base":
        set_requires_grad(model, True)
        set_requires_grad(diffusion_regularizer, False)
        trainable_params = list(model.parameters())
    elif phase == "phase2_diffusion":
        if diffusion_regularizer is None:
            raise ValueError("phase2_diffusion requires --use_diffusion true or a diffusion module.")
        set_requires_grad(model, False)
        set_requires_grad(diffusion_regularizer, True)
        trainable_params = list(diffusion_regularizer.parameters())
    elif phase == "phase3_dce":
        if diffusion_regularizer is None:
            raise ValueError("phase3_dce requires a trained diffusion module. Use --diffusion_checkpoint if available.")
        set_requires_grad(model, True)
        set_requires_grad(diffusion_regularizer, False)
        trainable_params = list(model.parameters())
    elif phase == "phase3_ic_dce":
        # Ablation: instance commitment only. Train the main model with
        # CE(q_IC, p) and the same commitment weight alpha*ce_weight.
        set_requires_grad(model, True)
        set_requires_grad(diffusion_regularizer, False)
        trainable_params = list(model.parameters())
    elif phase == "phase3_joint_dce":
        if diffusion_regularizer is None:
            raise ValueError("phase3_joint_dce requires a trained diffusion module. Use --diffusion_checkpoint if available.")
        set_requires_grad(model, True)
        set_requires_grad(diffusion_regularizer, True)
        trainable_params = list(model.parameters()) + list(diffusion_regularizer.parameters())
    else:
        raise ValueError(f"Unknown training_phase={phase}")

    if len(trainable_params) == 0:
        raise RuntimeError(f"No trainable parameters for phase={phase}")

    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=conf.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    amp_enabled = conf.use_amp and conf.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and get_amp_dtype() == torch.float16))

    best_score = None
    es_counter = 0
    early_stop = False

    logger.info(
        f"[Train] Start | phase={phase} | epochs={total_epochs} | lr={args.learning_rate:.6f} | "
        f"diffusion={diffusion_regularizer is not None} | "
        f"topk={args.commitment_topk} | checkpoint={checkpoint_path}"
    )
    if diffusion_regularizer is not None:
        logger.info(
            f"[Diffusion] Diffusion-generated soft target | "
            f"start_epoch={args.diffusion_start_epoch}, lambda_max={args.diffusion_lambda_max:.4f}, "
            f"qIC_temp=({args.diffusion_temp_min:.2f},{args.diffusion_temp_max:.2f}), "
            f"teacher_temp=({args.diffusion_teacher_temp_min:.2f},{args.diffusion_teacher_temp_max:.2f}), "
            f"ctx_temp={args.diffusion_ctx_temperature:.2f}, ctx_mix_max={args.diffusion_context_mix_max:.2f}, "
            f"revKL={args.diffusion_reverse_kl_weight:.2f}, input_noise=({args.diffusion_input_noise_min:.2f},{args.diffusion_input_noise_max:.2f}), "
            f"target_soft_mix={args.diffusion_target_soft_mix:.2f}, refine_steps={args.diffusion_refine_steps}, "
            f"mask_prob={args.diffusion_mask_prob:.2f}, ctx_loss_w={args.diffusion_context_loss_weight:.2f}, "
            f"DCE_lambda={args.diffusion_dce_lambda:.3f}, "
            f"quality_gate={args.diffusion_quality_gate_mode}, gate_H={args.diffusion_gate_entropy_threshold:.3f}, gate_M={args.diffusion_gate_margin_threshold:.3f}, "
            f"teacher=Anneal(q_IC), denoiser=TrajectoryTransformer(Q_noisy[1:T]), "
            f"masked_loss_w={args.diffusion_context_loss_weight:.2f}, CE weight=commitment(alpha*ce_weight)"
        )

    previous_cache = None
    for epoch in range(total_epochs):
        needs_cache = phase in {"phase2_diffusion", "phase3_dce", "phase3_joint_dce", "phase3_ic_dce"} or bool(args.use_diffusion)
        if needs_cache:
            # Cache mining always uses the current main model. In phase2 the model
            # is frozen; in phase3 it changes and the cache is refreshed each epoch.
            pseudo_cache = mine_commitment_pseudo_labels(
                model=model,
                data_loader=train_loader,
                conf=conf,
                args=args,
                previous_cache=previous_cache,
                source_tag=f"{phase}_epoch_{epoch + 1}",
            )
            previous_cache = pseudo_cache
        else:
            pseudo_cache = {}

        if phase == "phase2_diffusion":
            model.eval()
            diffusion_regularizer.train()
        elif phase == "phase3_dce":
            model.train()
            diffusion_regularizer.eval()  # frozen diffusion-generated teacher
        elif phase == "phase3_joint_dce":
            model.train()
            diffusion_regularizer.train()
        elif phase == "phase3_ic_dce":
            model.train()
            if diffusion_regularizer is not None:
                diffusion_regularizer.eval()
        elif phase == "phase1_base":
            model.train()
            if diffusion_regularizer is not None:
                diffusion_regularizer.eval()
        else:
            model.train()
            if diffusion_regularizer is not None:
                diffusion_regularizer.train()

        epoch_base_loss_sum = 0.0
        epoch_train_loss_sum = 0.0
        epoch_pll_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_diff_commitment_sum = 0.0
        epoch_diff_commitment_weight_sum = 0.0
        epoch_diff_commitment_steps_sum = 0.0
        epoch_diff_commitment_quality_sum = 0.0
        epoch_diff_sum = 0.0
        epoch_weighted_diff_sum = 0.0
        epoch_dynamic_diff_weight_sum = 0.0
        epoch_align_sum = 0.0
        epoch_context_sum = 0.0
        epoch_ent_sum = 0.0
        epoch_margin_loss_sum = 0.0
        epoch_weight_sum = 0.0
        epoch_agree_sum = 0.0
        epoch_sharper_sum = 0.0
        epoch_margin_better_sum = 0.0
        epoch_entropy_gate_sum = 0.0
        epoch_margin_gate_sum = 0.0
        epoch_use_diff_sum = 0.0
        epoch_eta_sum = 0.0
        epoch_mask_rate_sum = 0.0
        epoch_ctx_lambda_sum = 0.0
        epoch_ctx_conf_sum = 0.0
        epoch_ctx_agree_sum = 0.0
        epoch_diff_batches = 0
        epoch_valid_steps = 0
        epoch_ce_valid_steps = 0
        epoch_alpha_sum = 0.0
        epoch_ce_weight_sum = 0.0
        epoch_top1_prob_sum = 0.0
        epoch_entropy_sum = 0.0
        epoch_margin_sum = 0.0
        epoch_alpha_hist_steps = 0
        epoch_stat_steps = 0
        alpha_hist_counts = [0.0, 0.0, 0.0, 0.0]

        for batch in train_loader:
            batch = move_batch_to_device(batch, conf.device, non_blocking=(conf.device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(conf.device, conf.use_amp):
                # Prefer the new return_candidate_vectors flag. Fall back to the old model API.
                try:
                    cand_scores, neg_scores, user_vector, cand_vectors = model(batch, return_candidate_vectors=True)
                except TypeError:
                    cand_scores, neg_scores, user_vector = model(batch)
                    cand_vectors = model.encode_candidate_vectors(batch) if hasattr(model, "encode_candidate_vectors") else None

                pll_step_loss = model.compute_weighted_clpl_loss_per_step(
                    cand_scores=cand_scores,
                    cand_mask=batch["cand_mask"],
                    neg_scores=neg_scores,
                )

                target_pack = build_commitment_targets_from_cache(
                    batch=batch,
                    cand_scores=cand_scores,
                    pseudo_cache=pseudo_cache,
                )
                q_ic = target_pack["q_soft"]
                ce_valid_mask = target_pack["ce_valid_mask"]

                alpha_map = target_pack["alpha"]
                ce_weight_map = target_pack["ce_weight"]
                ce_valid_float = ce_valid_mask.float()
                effective_alpha = alpha_map * ce_valid_float
                effective_ce_weight = ce_weight_map * ce_valid_float

                # Instance commitment weight used by both diffusion fitting and generated-softCE.
                # No additional fixed strict thresholds are applied by default.
                commitment_gate_weight = effective_alpha * effective_ce_weight * ce_valid_float

                seq_mask_float = batch["seq_mask"].float()
                diffusion_active = (
                    diffusion_regularizer is not None
                    and (epoch + 1) >= int(args.diffusion_start_epoch)
                    and ce_valid_mask.any()
                    and phase in {"phase2_diffusion", "phase3_dce", "phase3_joint_dce"}
                )

                diff_loss = cand_scores.new_zeros(())
                dynamic_diff_weight = cand_scores.new_zeros(())
                weighted_diff_loss = cand_scores.new_zeros(())
                diff_out = None
                diff_commitment_component = torch.zeros_like(pll_step_loss)
                ce_component = torch.zeros_like(pll_step_loss)
                pll_component = pll_step_loss
                total_step_loss = pll_component

                if diffusion_active:
                    diff_condition = user_vector if bool(args.diffusion_use_condition) else None
                    diff_cand_vectors = cand_vectors.detach() if (cand_vectors is not None and phase == "phase3_joint_dce") else cand_vectors
                    diff_out = diffusion_regularizer(
                        cand_scores=cand_scores,
                        cand_mask=batch["cand_mask"],
                        seq_mask=batch["seq_mask"],
                        q_soft=q_ic,
                        step_weight=commitment_gate_weight.detach(),
                        instance_alpha=effective_alpha.detach(),
                        condition=diff_condition,
                        cand_vectors=diff_cand_vectors,
                    )
                    diff_loss = diff_out["loss"]

                if phase == "phase1_base":
                    # Final design Phase I: no soft CE. Instance commitment is not
                    # used to directly supervise the main model here.
                    pll_component = pll_step_loss
                    total_step_loss = pll_component
                    base_loss = (total_step_loss * seq_mask_float).sum() / seq_mask_float.sum().clamp(min=1.0)
                    train_loss = base_loss

                elif phase == "phase3_ic_dce":
                    # Ablation: only InstanceCommitment. Use q_IC directly as the
                    # soft CE target; do NOT apply sharpening, context correction,
                    # or diffusion denoising. This isolates the contribution of
                    # instance commitment from diffusion-generated targets.
                    pll_component = pll_step_loss
                    ce_component = (
                        float(args.ic_dce_lambda)
                        * commitment_gate_weight.detach()
                        * target_pack["ce_step_loss"]
                    )
                    diff_commitment_component = torch.zeros_like(pll_step_loss)
                    total_step_loss = pll_component + ce_component
                    base_loss = (total_step_loss * seq_mask_float).sum() / seq_mask_float.sum().clamp(min=1.0)
                    train_loss = base_loss

                elif phase == "phase2_diffusion":
                    # Freeze model, train only diffusion to recover the instance-
                    # sharpened teacher.  The sample weights are already inside
                    # diff_loss, so the global multiplier only scales the module
                    # fitting loss and defaults to 1.0.
                    if diff_out is None:
                        train_loss = cand_scores.new_zeros((), requires_grad=True)
                    else:
                        dynamic_diff_weight = cand_scores.new_tensor(float(args.diffusion_lambda_max))
                        weighted_diff_loss = dynamic_diff_weight * diff_loss
                        train_loss = weighted_diff_loss
                    pll_component = torch.zeros_like(pll_step_loss)
                    ce_component = torch.zeros_like(pll_step_loss)
                    diff_commitment_component = torch.zeros_like(pll_step_loss)
                    total_step_loss = torch.zeros_like(pll_step_loss)
                    base_loss = cand_scores.new_zeros(())

                elif phase == "phase3_dce":
                    # Freeze diffusion, use q_diff as the generated soft-CE target for the main model.
                    pll_component = pll_step_loss
                    if diff_out is not None:
                        if bool(args.diffusion_commitment_use_quality_gate):
                            diff_commitment_weight = diff_out["diff_commitment_weight"]
                        else:
                            diff_commitment_weight = diff_out["diff_commitment_base_weight"]
                        diff_commitment_component = (
                            float(args.diffusion_dce_lambda)
                            * diff_commitment_weight.detach()
                            * diff_out["diff_commitment_step_loss"]
                        )
                    total_step_loss = pll_component + diff_commitment_component
                    base_loss = (total_step_loss * seq_mask_float).sum() / seq_mask_float.sum().clamp(min=1.0)
                    train_loss = base_loss

                elif phase == "phase3_joint_dce":
                    # Joint co-training: the model is updated by PLL + CE(q_diff, p),
                    # while the diffusion module continues to learn the denoising objective.
                    # q_diff is detached inside diff_commitment_step_loss, so DCE does not
                    # backpropagate into the diffusion module.
                    pll_component = pll_step_loss
                    if diff_out is not None:
                        dynamic_diff_weight = cand_scores.new_tensor(float(args.diffusion_lambda_max))
                        weighted_diff_loss = dynamic_diff_weight * diff_loss

                        if bool(args.diffusion_commitment_use_quality_gate):
                            diff_commitment_weight = diff_out["diff_commitment_weight"]
                        else:
                            diff_commitment_weight = diff_out["diff_commitment_base_weight"]

                        diff_commitment_component = (
                            float(args.diffusion_dce_lambda)
                            * diff_commitment_weight.detach()
                            * diff_out["diff_commitment_step_loss"]
                        )

                    total_step_loss = pll_component + diff_commitment_component
                    base_loss = (total_step_loss * seq_mask_float).sum() / seq_mask_float.sum().clamp(min=1.0)
                    train_loss = base_loss + weighted_diff_loss

                else:
                    raise ValueError(f"Unknown training_phase={phase}")

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            valid_mask = batch["seq_mask"].bool()
            valid_mask_float = valid_mask.float()
            valid_steps = int(valid_mask.sum().item())
            ce_valid_seq_mask = ce_valid_mask & valid_mask
            alpha_values = alpha_map.detach()[ce_valid_seq_mask]

            epoch_base_loss_sum += float((total_step_loss.detach() * valid_mask_float).sum().item())
            epoch_train_loss_sum += float(train_loss.detach().item()) * max(valid_steps, 1)
            epoch_pll_sum += float((pll_component.detach() * valid_mask_float).sum().item())
            epoch_ce_sum += float((ce_component.detach() * valid_mask_float).sum().item())
            epoch_diff_commitment_sum += float((diff_commitment_component.detach() * valid_mask_float).sum().item())
            epoch_valid_steps += valid_steps
            epoch_ce_valid_steps += int(ce_valid_seq_mask.sum().item())
            epoch_alpha_sum += float(alpha_values.sum().item())

            if diff_out is not None:
                epoch_diff_sum += float(diff_out["loss"].detach().item()) * max(valid_steps, 1)
                epoch_weighted_diff_sum += float(weighted_diff_loss.detach().item()) * max(valid_steps, 1)
                epoch_dynamic_diff_weight_sum += float(dynamic_diff_weight.detach().item())
                epoch_align_sum += float(diff_out["loss_align"].detach().item()) * max(valid_steps, 1)
                epoch_context_sum += float(diff_out.get("loss_context", cand_scores.new_zeros(())).detach().item()) * max(valid_steps, 1)
                epoch_ent_sum += float(diff_out["loss_entropy"].detach().item()) * max(valid_steps, 1)
                epoch_margin_loss_sum += float(diff_out["loss_margin"].detach().item()) * max(valid_steps, 1)
                epoch_weight_sum += float(diff_out["avg_gate"].detach().item())
                epoch_agree_sum += float(diff_out["agree_rate"].detach().item())
                epoch_sharper_sum += float(diff_out["sharper_rate"].detach().item())
                epoch_margin_better_sum += float(diff_out["margin_better_rate"].detach().item())
                epoch_entropy_gate_sum += float(diff_out.get("entropy_gate_rate", cand_scores.new_zeros(())).detach().item())
                epoch_margin_gate_sum += float(diff_out.get("margin_gate_rate", cand_scores.new_zeros(())).detach().item())
                epoch_use_diff_sum += float(diff_out["use_diff_rate"].detach().item())
                epoch_eta_sum += float(diff_out["avg_eta"].detach().item())
                epoch_mask_rate_sum += float(diff_out.get("mask_rate", cand_scores.new_zeros(())).detach().item())
                epoch_ctx_lambda_sum += float(diff_out.get("avg_ctx_lambda", cand_scores.new_zeros(())).detach().item())
                epoch_ctx_conf_sum += float(diff_out.get("avg_ctx_conf", cand_scores.new_zeros(())).detach().item())
                epoch_ctx_agree_sum += float(diff_out.get("avg_ctx_agree", cand_scores.new_zeros(())).detach().item())
                if bool(args.diffusion_commitment_use_quality_gate):
                    epoch_diff_commitment_weight_sum += float(diff_out["diff_commitment_weight"].detach().sum().item())
                    epoch_diff_commitment_steps_sum += float(diff_out["diff_commitment_mask"].detach().float().sum().item())
                else:
                    epoch_diff_commitment_weight_sum += float(diff_out["diff_commitment_base_weight"].detach().sum().item())
                    epoch_diff_commitment_steps_sum += float(diff_out["diff_commitment_base_mask"].detach().float().sum().item())
                epoch_diff_commitment_quality_sum += float(diff_out.get("avg_diff_commitment_quality", cand_scores.new_zeros(())).detach().item())
                epoch_diff_batches += 1

            if ce_valid_seq_mask.any():
                epoch_ce_weight_sum += float(effective_ce_weight.detach()[ce_valid_seq_mask].sum().item())
                epoch_top1_prob_sum += float(target_pack["top1_prob"][ce_valid_seq_mask].sum().item())
                epoch_entropy_sum += float(target_pack["normalized_entropy"][ce_valid_seq_mask].sum().item())
                epoch_margin_sum += float(target_pack["margin"][ce_valid_seq_mask].sum().item())
                epoch_stat_steps += int(ce_valid_seq_mask.sum().item())

            batch_hist = compute_alpha_histogram(alpha_values)
            batch_hist_total = int(alpha_values.numel())
            for idx, value in enumerate(batch_hist):
                alpha_hist_counts[idx] += value * batch_hist_total
            epoch_alpha_hist_steps += batch_hist_total

        avg_base_loss = epoch_base_loss_sum / max(epoch_valid_steps, 1)
        avg_train_loss = epoch_train_loss_sum / max(epoch_valid_steps, 1)
        avg_pll = epoch_pll_sum / max(epoch_valid_steps, 1)
        avg_ce = epoch_ce_sum / max(epoch_valid_steps, 1)
        avg_diff_commitment = epoch_diff_commitment_sum / max(epoch_valid_steps, 1)
        avg_diff_commitment_weight = epoch_diff_commitment_weight_sum / max(epoch_valid_steps, 1)
        avg_diff_commitment_coverage = epoch_diff_commitment_steps_sum / max(epoch_valid_steps, 1)
        avg_diff_commitment_quality = epoch_diff_commitment_quality_sum / max(epoch_diff_batches, 1)
        avg_diff = epoch_diff_sum / max(epoch_valid_steps, 1)
        avg_weighted_diff = epoch_weighted_diff_sum / max(epoch_valid_steps, 1)
        avg_dynamic_diff_weight = epoch_dynamic_diff_weight_sum / max(epoch_diff_batches, 1)
        avg_align = epoch_align_sum / max(epoch_valid_steps, 1)
        avg_context = epoch_context_sum / max(epoch_valid_steps, 1)
        avg_ent = epoch_ent_sum / max(epoch_valid_steps, 1)
        avg_margin_loss = epoch_margin_loss_sum / max(epoch_valid_steps, 1)
        avg_weight = epoch_weight_sum / max(epoch_diff_batches, 1)
        avg_agree = epoch_agree_sum / max(epoch_diff_batches, 1)
        avg_sharper = epoch_sharper_sum / max(epoch_diff_batches, 1)
        avg_margin_better = epoch_margin_better_sum / max(epoch_diff_batches, 1)
        avg_entropy_gate = epoch_entropy_gate_sum / max(epoch_diff_batches, 1)
        avg_margin_gate = epoch_margin_gate_sum / max(epoch_diff_batches, 1)
        avg_use_diff = epoch_use_diff_sum / max(epoch_diff_batches, 1)
        avg_eta = epoch_eta_sum / max(epoch_diff_batches, 1)
        avg_mask_rate = epoch_mask_rate_sum / max(epoch_diff_batches, 1)
        avg_ctx_lambda = epoch_ctx_lambda_sum / max(epoch_diff_batches, 1)
        avg_ctx_conf = epoch_ctx_conf_sum / max(epoch_diff_batches, 1)
        avg_ctx_agree = epoch_ctx_agree_sum / max(epoch_diff_batches, 1)
        pseudo_coverage_ratio = epoch_ce_valid_steps / max(epoch_valid_steps, 1)
        avg_alpha = epoch_alpha_sum / max(epoch_ce_valid_steps, 1)
        avg_ce_weight = epoch_ce_weight_sum / max(epoch_ce_valid_steps, 1)
        alpha_hist_ratio = [count / max(epoch_alpha_hist_steps, 1) for count in alpha_hist_counts]
        avg_top1_prob = epoch_top1_prob_sum / max(epoch_stat_steps, 1)
        avg_entropy = epoch_entropy_sum / max(epoch_stat_steps, 1)
        avg_margin = epoch_margin_sum / max(epoch_stat_steps, 1)

        val_metrics = evaluate_metrics(model, val_loader, conf.device, use_amp=conf.use_amp)
        if phase != "phase2_diffusion":
            scheduler.step(val_metrics["acc1"])

        score = -avg_diff if phase == "phase2_diffusion" else val_metrics["acc1"]
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            es_counter = 0
            save_checkpoint_components(checkpoint_path, model, diffusion_regularizer, phase=phase, score=float(score))
        else:
            es_counter += 1
            if es_counter >= conf.patience:
                early_stop = True

        logger.info(
            f"[Train] Epoch {epoch + 1}/{total_epochs} | Phase={phase} | TrainLoss={avg_train_loss:.4f} | "
            f"BaseLoss={avg_base_loss:.4f} | PLL={avg_pll:.4f} | DirectSoftCE={avg_ce:.4f} | "
            f"DiffCE={avg_diff_commitment:.6f} | DiffCEW={avg_diff_commitment_weight:.6f} | DiffCECov={avg_diff_commitment_coverage:.4f} | DiffCEQ={avg_diff_commitment_quality:.4f} | "
            f"Diff={avg_diff:.4f} | DynDiffW={avg_dynamic_diff_weight:.6f} | WeightedDiff={avg_weighted_diff:.6f} | "
            f"Align={avg_align:.4f} | MaskLoss={avg_context:.4f} | Ent={avg_ent:.4f} | MarginLoss={avg_margin_loss:.4f} | "
            f"AvgGate={avg_weight:.4f} | Agree={avg_agree:.4f} | Sharper={avg_sharper:.4f} | "
            f"MarginBetter={avg_margin_better:.4f} | EntGate={avg_entropy_gate:.4f} | MarginGate={avg_margin_gate:.4f} | "
            f"UseDiff={avg_use_diff:.4f} | Eta/DCEW={avg_eta:.4f} | "
            f"MaskRate={avg_mask_rate:.4f} | CtxLambda={avg_ctx_lambda:.4f} | CtxConf={avg_ctx_conf:.4f} | CtxAgree={avg_ctx_agree:.4f} | "
            f"PseudoCov={pseudo_coverage_ratio:.4f} | AvgAlpha={avg_alpha:.4f} | AvgCEWeight={avg_ce_weight:.4f} | "
            f"Val Acc@1={val_metrics['acc1']:.4f} | Val Acc@5={val_metrics['acc5']:.4f} | "
            f"ES={es_counter}/{conf.patience} | BestScore={best_score:.4f} | SavedBest={int(improved)} | Cache={len(pseudo_cache)}"
        )
        logger.info(
            f"[Train] Epoch {epoch + 1} stats | "
            f"AlphaHist=[0.0,0.2):{alpha_hist_ratio[0]:.4f}, "
            f"[0.2,0.5):{alpha_hist_ratio[1]:.4f}, "
            f"[0.5,0.8):{alpha_hist_ratio[2]:.4f}, "
            f"[0.8,1.0]:{alpha_hist_ratio[3]:.4f} | "
            f"AvgTop1Prob={avg_top1_prob:.4f} | AvgNormEntropy={avg_entropy:.4f} | AvgMargin={avg_margin:.4f}"
        )

        if early_stop:
            min_epoch_after_diffusion = int(args.diffusion_start_epoch) + max(int(args.min_epochs_after_diffusion), 0) - 1
            should_delay_stop = (
                diffusion_regularizer is not None
                and int(args.min_epochs_after_diffusion) > 0
                and (epoch + 1) < min_epoch_after_diffusion
                and phase in {"phase2_diffusion", "phase3_dce", "phase3_joint_dce"}
            )
            if should_delay_stop:
                logger.info(
                    f"[Train] Early stopping delayed until epoch {min_epoch_after_diffusion} "
                    f"to allow diffusion training after start_epoch={args.diffusion_start_epoch}."
                )
                early_stop = False
            else:
                logger.info(f"[Train] Early stopping triggered at epoch {epoch + 1}.")
                break

    if os.path.exists(checkpoint_path):
        load_checkpoint_components(
            checkpoint_path,
            model=model,
            diffusion=diffusion_regularizer,
            device=conf.device,
            strict_model=False,
            strict_diffusion=False,
            logger=logger,
        )

    val_metrics = evaluate_metrics(model, val_loader, conf.device, use_amp=conf.use_amp)
    test_metrics = evaluate_metrics(model, test_loader, conf.device, use_amp=conf.use_amp)
    logger.info(
        f"[Train] Reloaded Best -> Val Acc@1={val_metrics['acc1']:.4f}, Val Acc@5={val_metrics['acc5']:.4f}, "
        f"Test Acc@1={test_metrics['acc1']:.4f}, Test Acc@5={test_metrics['acc5']:.4f}, "
        f"Test Acc@Cat={test_metrics['acc_cat']:.4f}, Test Acc@MainCat={test_metrics['acc_main_cat']:.4f}, "
        f"saved={checkpoint_path}"
    )
    return {"val": val_metrics, "test": test_metrics, "checkpoint_path": checkpoint_path}


def main():
    parser = argparse.ArgumentParser(description="PLL + instance-aware soft commitment + diffusion sharpening")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--exp_name", type=str, default=DEFAULT_EXP_NAME)
    parser.add_argument("--model_variant", type=str, default="base", choices=["base", "no_dist"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_epochs_after_diffusion", type=int, default=5,
                        help="Delay early stopping until this many epochs after diffusion_start_epoch. Useful for diffusion ablations.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--radius_size", type=int, default=200)

    parser.add_argument("--training_phase", type=str, default="phase1_base", choices=["phase1_base", "phase3_ic_dce", "phase2_diffusion", "phase3_dce", "phase3_joint_dce"],
                        help="phase1_base=PLL only; phase3_ic_dce=ablation with CE(q_IC,p); phase2_diffusion=freeze model and train diffusion; phase3_dce=freeze diffusion and train model with CE(q_diff,p); phase3_joint_dce=co-train model and diffusion.")
    parser.add_argument("--use_diffusion", type=str2bool, default=False)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--diffusion_checkpoint", type=str, default=None,
                        help="Optional combined checkpoint containing diffusion_state_dict; used for phase3_dce or continued diffusion training.")

    parser.add_argument("--commitment_topk", type=int, default=5)
    parser.add_argument("--commitment_use_stability", type=str2bool, default=False)
    parser.add_argument("--commitment_conf_weight", type=float, default=0.30)
    parser.add_argument("--commitment_margin_weight", type=float, default=0.20)
    parser.add_argument("--commitment_entropy_weight", type=float, default=0.20)
    parser.add_argument("--commitment_time_weight", type=float, default=0.10)
    parser.add_argument("--commitment_main_time_weight", type=float, default=0.10)
    parser.add_argument("--commitment_recur_weight", type=float, default=0.10)
    parser.add_argument("--commitment_stability_weight", type=float, default=0.00)
    parser.add_argument("--commitment_bias", type=float, default=0.00)
    parser.add_argument("--gate_threshold", type=float, default=0.35)
    parser.add_argument("--ic_dce_lambda", type=float, default=1.00,
                        help="Global multiplier for the InstanceCommitment-only ablation CE(q_IC,p). The sample weight is alpha*ce_weight.")

    # Diffusion sharpening. Removed the old raw-prediction restoration params:
    # tau_teacher, tau_student, KL/rec weights, reliability fallback, and separate
    # diffusion commitment weight. Diffusion now learns sharpened q_soft and can
    # only refine commitment through the gated q_final target.
    parser.add_argument("--diffusion_start_epoch", type=int, default=1)
    parser.add_argument("--diffusion_lambda_max", type=float, default=1.00,
                        help="Global multiplier for phase2 diffusion fitting loss. Default 1.0 because phase2 trains only diffusion.")
    parser.add_argument("--diffusion_commitment_use_quality_gate", type=str2bool, default=True,
                        help="If true, CE(q_diff) uses alpha*ce_weight only on q_diff passing the selected quality gate.")
    parser.add_argument("--diffusion_quality_gate_mode", type=str, default="relaxed", choices=["strict", "relaxed", "none"],
                        help="Quality gate for q_diff. strict=no worse than q_IC; relaxed=entropy/margin thresholded; none=all eligible q_diff.")
    parser.add_argument("--diffusion_gate_entropy_threshold", type=float, default=0.10,
                        help="Relaxed gate threshold for normalized entropy: H_norm(q_diff)<=max(H_norm(q_IC), threshold).")
    parser.add_argument("--diffusion_gate_margin_threshold", type=float, default=0.80,
                        help="Relaxed gate threshold for margin: M(q_diff)>=min(M(q_IC), threshold).")
    parser.add_argument("--diffusion_hidden_dim", type=int, default=64)
    parser.add_argument("--diffusion_layers", type=int, default=1)
    parser.add_argument("--diffusion_heads", type=int, default=4)
    parser.add_argument("--diffusion_dropout", type=float, default=0.10)
    parser.add_argument("--diffusion_temp_min", type=float, default=0.40)
    parser.add_argument("--diffusion_temp_max", type=float, default=0.80)
    parser.add_argument("--diffusion_entropy_weight", type=float, default=0.02)
    parser.add_argument("--diffusion_margin_weight", type=float, default=0.05)
    parser.add_argument("--diffusion_target_margin", type=float, default=0.05)
    parser.add_argument("--diffusion_mask_prob", type=float, default=0.10)
    parser.add_argument("--diffusion_use_condition", type=str2bool, default=True)
    parser.add_argument("--diffusion_detach_condition", type=str2bool, default=True)
    parser.add_argument("--diffusion_refine_steps", type=int, default=3,
                        help="Number of iterative trajectory-belief refinement steps.")
    parser.add_argument("--diffusion_target_soft_mix", type=float, default=1.00,
                        help="q_target=(1-mix)*q_IC + mix*Anneal(q_IC). Main method uses 1.0.")
    parser.add_argument("--diffusion_teacher_temp_min", type=float, default=0.55,
                        help="Minimum temperature for the final teacher sharpening. Higher confidence uses this lower temperature.")
    parser.add_argument("--diffusion_teacher_temp_max", type=float, default=0.85,
                        help="Maximum temperature for the final teacher sharpening on less reliable committed steps.")
    parser.add_argument("--diffusion_reverse_kl_weight", type=float, default=0.20,
                        help="Weight for KL(q_diff || q_target), encouraging mode-seeking sharp q_diff.")
    parser.add_argument("--diffusion_input_noise_min", type=float, default=0.05,
                        help="Minimum reliability-adaptive smoothing applied to q_IC before denoising.")
    parser.add_argument("--diffusion_input_noise_max", type=float, default=0.45,
                        help="Maximum reliability-adaptive smoothing applied to q_IC before denoising.")
    parser.add_argument("--diffusion_mask_condition", type=str2bool, default=True,
                        help="If true, mask the current-step condition vector when the distribution step is masked.")
    parser.add_argument("--diffusion_ctx_temperature", type=float, default=0.50,
                        help="Backward-compatible unused arg in trajectory-level Transformer diffusion.")
    parser.add_argument("--diffusion_context_mix_max", type=float, default=0.20,
                        help="Backward-compatible unused arg in trajectory-level Transformer diffusion.")
    parser.add_argument("--diffusion_context_loss_weight", type=float, default=0.20,
                        help="Weight for masked-step trajectory denoising loss. Larger values force stronger context recovery.")
    parser.add_argument("--diffusion_context_anchor_min_weight", type=float, default=0.10,
                        help="Backward-compatible unused arg in trajectory-level Transformer diffusion.")
    parser.add_argument("--diffusion_dce_lambda", type=float, default=1.00,
                        help="Global multiplier for Stage-III CE(q_diff,p). Default 1.0: actual sample weight is the commitment weight alpha*ce_weight.")
    args = parser.parse_args()
    if args.training_phase in {"phase2_diffusion", "phase3_dce", "phase3_joint_dce"}:
        args.use_diffusion = True

    result_dir = os.path.join(PROJECT_ROOT, "result", args.dataset, args.exp_name)
    log_file = args.log_file or os.path.join(result_dir, "training.log")
    save_path = args.save_name or os.path.join(result_dir, "best_model.pth")

    os.makedirs(result_dir, exist_ok=True)
    if os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    global logger
    init_logging(log_file=log_file, force=True)
    logger = get_logger("Training")

    conf = TrainingConfig()
    conf.batch_size = args.batch_size
    conf.patience = max(int(args.patience), 1)
    conf.num_workers = max(int(args.num_workers), 0)
    conf.pin_memory = (conf.device.type == "cuda")
    conf.persistent_workers = conf.num_workers > 0
    conf.use_amp = (not args.disable_amp) and (conf.device.type == "cuda")
    conf.radius = args.radius_size

    paths = get_dataset_paths(args.dataset)
    conf.checkin_file = paths["checkin_file"]
    conf.poi_file = paths["poi_file"]
    conf.dist_file = paths["dist_file"]

    set_seed(conf.seed)

    poi_config = POIProcessingConfig(
        checkin_file=conf.checkin_file,
        poi_file=conf.poi_file,
        dist_file=conf.dist_file,
        radius=conf.radius,
        max_candidates=conf.max_candidates,
        device=conf.device,
    )
    processor = POIDataProcessor(poi_config)
    model_config = load_model_config(args, processor, conf)

    logger.info(
        "[Config] "
        f"epochs={max(int(args.epochs), 1)}, lr={args.learning_rate:.6f}, "
        f"phase={args.training_phase}, diffusion={bool(args.use_diffusion)}, "
        f"diff_commit_dynamic_weight=sample-wise, "
        f"topk={args.commitment_topk}, stability={bool(args.commitment_use_stability)}, "
        f"gate_threshold={args.gate_threshold:.2f}, ic_dce_lambda={args.ic_dce_lambda:.2f}, batch_size={conf.batch_size}, "
        f"num_workers={conf.num_workers}, amp={conf.use_amp}, model_variant={args.model_variant}"
    )

    train_loader, val_loader, test_loader = build_dataloaders(processor, conf)

    model = build_model(model_config, args.model_variant).to(conf.device)
    if args.pretrained_checkpoint:
        if not os.path.exists(args.pretrained_checkpoint):
            raise FileNotFoundError(f"pretrained checkpoint not found: {args.pretrained_checkpoint}")
        state = torch.load(args.pretrained_checkpoint, map_location=conf.device)
        state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"[Checkpoint] Loaded pretrained model from {args.pretrained_checkpoint} | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    info = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        conf=conf,
        args=args,
        checkpoint_path=save_path,
    )
    test_metrics = info["test"]
    logger.info(
        f"[Summary] FinalModel={info['checkpoint_path']} | "
        f"Final Test Acc@1={test_metrics['acc1']:.4f} | Acc@5={test_metrics['acc5']:.4f} | "
        f"Acc@Cat={test_metrics['acc_cat']:.4f} | Acc@MainCat={test_metrics['acc_main_cat']:.4f} | "
        f"ModelVariant={args.model_variant}"
    )


if __name__ == "__main__":
    main()
