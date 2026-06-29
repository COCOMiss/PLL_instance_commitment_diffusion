import torch
import torch.nn.functional as F
from contextlib import nullcontext

from utils import get_logger


logger = get_logger("EasySample")
MODEL_ID_OFFSET = 1


def _get_autocast_context(conf):
    if getattr(conf, "use_amp", False) and conf.device.type == "cuda":
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _extract_soft_target(batch, probs, b, s, soft_topk):
    valid_mask = batch["cand_mask"][b, s].bool()
    valid_probs = probs[b, s][valid_mask]
    valid_poi_ids = batch["cand_poi_ids"][b, s][valid_mask]

    if valid_probs.numel() == 0:
        return [], []

    k = min(int(soft_topk), int(valid_probs.numel()))
    topk_probs, topk_local = torch.topk(valid_probs, k=k)
    topk_poi_ids = valid_poi_ids[topk_local]

    mass = topk_probs.sum().clamp_min(1e-12)
    topk_probs = topk_probs / mass

    return (
        [int(poi_id.item()) for poi_id in topk_poi_ids],
        [float(prob.item()) for prob in topk_probs],
    )


def _compute_entropy_stats(probs, valid_counts):
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    norm_denom = torch.log(valid_counts.float().clamp(min=2.0))
    normalized_entropy = entropy / norm_denom.clamp_min(1e-12)
    return entropy, normalized_entropy.clamp(min=0.0, max=1.0)


def _compute_commitment_alpha(args, top1_prob, margin, normalized_entropy, time_prior, main_time_prior, recurrence, stability):
    alpha_raw = (
        float(args.commitment_conf_weight) * float(top1_prob)
        + float(args.commitment_margin_weight) * float(margin)
        + float(args.commitment_entropy_weight) * float(max(0.0, min(1.0, 1.0 - normalized_entropy)))
        + float(args.commitment_time_weight) * float(time_prior)
        + float(args.commitment_main_time_weight) * float(main_time_prior)
        + float(args.commitment_recur_weight) * float(recurrence)
        + float(args.commitment_stability_weight) * float(stability)
        + float(args.commitment_bias)
    )
    return max(0.0, min(1.0, alpha_raw))


def _compute_instance_gate_threshold(
    args,
    top1_prob,
    margin,
    normalized_entropy,
    time_prior,
    main_time_prior,
    recurrence,
    stability,
):
    """Instance-dependent gate threshold tau_t for entering stage-2 refinement."""
    tau_raw = (
        float(getattr(args, "commitment_gate_base", 0.5))
        + float(getattr(args, "commitment_gate_uncertainty_weight", 0.20)) * float(normalized_entropy)
        + float(getattr(args, "commitment_gate_lowconf_weight", 0.15)) * float(1.0 - top1_prob)
        + float(getattr(args, "commitment_gate_smallmargin_weight", 0.10)) * float(1.0 - margin)
        - float(getattr(args, "commitment_gate_time_weight", 0.05)) * float(time_prior)
        - float(getattr(args, "commitment_gate_main_time_weight", 0.05)) * float(main_time_prior)
        - float(getattr(args, "commitment_gate_recur_weight", 0.05)) * float(recurrence)
        - float(getattr(args, "commitment_gate_stability_weight", 0.05)) * float(stability)
    )
    return max(0.0, min(1.0, tau_raw))


def _get_dataset_row_metadata(dataset, sample_idx, step_idx):
    if sample_idx < 0 or sample_idx >= len(dataset.traj_groups):
        return {}

    traj_rows = dataset.traj_groups[sample_idx]
    if step_idx < 0 or step_idx >= len(traj_rows):
        return {}

    row_idx = int(traj_rows[step_idx])
    row = dataset.sorted_df.iloc[row_idx]
    return {
        "sample_idx": int(sample_idx),
        "time_step": int(step_idx),
        "row_idx": row_idx,
        "user_idx": int(row["user_idx_mapped"]),
        "local_datetime": row["local_datetime"].isoformat() if hasattr(row["local_datetime"], "isoformat") else str(row["local_datetime"]),
    }


def _collect_commitment_cache(
    model,
    data_loader,
    conf,
    args,
    previous_cache=None,
    source_tag="commitment_student",
    include_dataset_metadata=False,
):
    model.eval()
    dataset = data_loader.dataset
    commitment_cache = {}

    with torch.no_grad():
        for batch in data_loader:
            for key, value in batch.items():
                batch[key] = value.to(conf.device, non_blocking=(conf.device.type == "cuda"))

            with _get_autocast_context(conf):
                cand_scores = model.predict(batch)

            masked_scores = cand_scores.float().masked_fill(batch["cand_mask"] == 0, -1e9)
            probs = F.softmax(masked_scores, dim=-1)
            valid_counts = batch["cand_mask"].sum(dim=-1)

            topk_limit = min(2, probs.size(-1))
            topk_probs, topk_indices = torch.topk(probs, k=topk_limit, dim=-1)
            top1_prob = topk_probs[..., 0]
            if topk_limit > 1:
                top2_prob = topk_probs[..., 1]
            else:
                top2_prob = torch.zeros_like(top1_prob)

            _, normalized_entropy = _compute_entropy_stats(probs, valid_counts)
            seq_mask = batch["seq_mask"].bool()
            valid_steps = seq_mask & (valid_counts > 0)
            batch_indices, step_indices = torch.where(valid_steps)

            for batch_idx, step_idx in zip(batch_indices.tolist(), step_indices.tolist()):
                key = (int(batch["sample_idx"][batch_idx].item()), int(step_idx))
                local_top_idx = int(topk_indices[batch_idx, step_idx, 0].item())
                pseudo_poi_id = int(batch["cand_poi_ids"][batch_idx, step_idx, local_top_idx].item())
                margin = float((top1_prob[batch_idx, step_idx] - top2_prob[batch_idx, step_idx]).clamp(min=0.0, max=1.0).item())
                time_prior = float(batch["cand_probs"][batch_idx, step_idx, local_top_idx].item())
                main_time_prior = float(batch["cand_main_cat_probs"][batch_idx, step_idx, local_top_idx].item())
                recurrence = float(batch["cand_other_feats"][batch_idx, step_idx, local_top_idx, 0].item())

                stability = 0.0
                if getattr(args, "commitment_use_stability", False) and previous_cache is not None:
                    prev_entry = previous_cache.get(key)
                    if prev_entry and int(prev_entry.get("pseudo_poi_id", -1)) == pseudo_poi_id:
                        stability = 1.0

                soft_poi_ids, soft_probs = _extract_soft_target(
                    batch=batch,
                    probs=probs,
                    b=batch_idx,
                    s=step_idx,
                    soft_topk=args.commitment_topk,
                )
                if not soft_poi_ids:
                    continue

                top1_prob_value = float(top1_prob[batch_idx, step_idx].item())
                top2_prob_value = float(top2_prob[batch_idx, step_idx].item())
                entropy_value = float(normalized_entropy[batch_idx, step_idx].item())
                alpha = _compute_commitment_alpha(
                    args=args,
                    top1_prob=max(0.0, min(1.0, top1_prob_value)),
                    margin=max(0.0, min(1.0, margin)),
                    normalized_entropy=max(0.0, min(1.0, entropy_value)),
                    time_prior=max(0.0, min(1.0, time_prior)),
                    main_time_prior=max(0.0, min(1.0, main_time_prior)),
                    recurrence=max(0.0, min(1.0, recurrence)),
                    stability=stability,
                )
                # gate_threshold = _compute_instance_gate_threshold(
                #     args=args,
                #     top1_prob=max(0.0, min(1.0, top1_prob_value)),
                #     margin=max(0.0, min(1.0, margin)),
                #     normalized_entropy=max(0.0, min(1.0, entropy_value)),
                #     time_prior=max(0.0, min(1.0, time_prior)),
                #     main_time_prior=max(0.0, min(1.0, main_time_prior)),
                #     recurrence=max(0.0, min(1.0, recurrence)),
                #     stability=stability,
                # )
                effective_alpha = max(float(alpha)-float(args.gate_threshold), 0.0)

                info = {
                    "pseudo_poi_id": pseudo_poi_id,
                    "soft_poi_ids": soft_poi_ids,
                    "soft_probs": soft_probs,
                    "top1_prob": top1_prob_value,
                    "top2_prob": top2_prob_value,
                    "margin": margin,
                    "normalized_entropy": entropy_value,
                    "time_prior": time_prior,
                    "main_time_prior": main_time_prior,
                    "recurrence": recurrence,
                    "stability": stability,
                    "alpha": effective_alpha,
                    "ce_weight": max(0.0, min(1.0, top1_prob_value)),
                    "valid_candidate_count": int(valid_counts[batch_idx, step_idx].item()),
                    "source": source_tag,
                }
                if include_dataset_metadata:
                    info.update(_get_dataset_row_metadata(dataset, key[0], key[1]))

                commitment_cache[key] = info

    return commitment_cache


def mine_commitment_pseudo_labels(
    model,
    data_loader,
    conf,
    args,
    previous_cache=None,
    source_tag="commitment_student",
):
    logger.info(
        f"Refreshing commitment pseudo cache with {source_tag} "
        f"(topk={args.commitment_topk}, stability={bool(args.commitment_use_stability)})..."
    )
    commitment_cache = _collect_commitment_cache(
        model=model,
        data_loader=data_loader,
        conf=conf,
        args=args,
        previous_cache=previous_cache,
        source_tag=source_tag,
        include_dataset_metadata=False,
    )
    logger.info(f"Commitment pseudo cache refreshed: {len(commitment_cache)} entries.")
    return commitment_cache


def extract_commitment_pseudo_labels(
    model,
    data_loader,
    conf,
    args,
    previous_cache=None,
    source_tag="commitment_best",
):
    logger.info(
        f"Extracting full commitment pseudo cache with {source_tag} "
        f"(topk={args.commitment_topk}, stability={bool(args.commitment_use_stability)})..."
    )
    commitment_cache = _collect_commitment_cache(
        model=model,
        data_loader=data_loader,
        conf=conf,
        args=args,
        previous_cache=previous_cache,
        source_tag=source_tag,
        include_dataset_metadata=True,
    )
    logger.info(f"Extracted {len(commitment_cache)} commitment pseudo labels for {source_tag}.")
    return commitment_cache



##弃用
def mine_easy_samples(
    model,
    data_loader,
    processor,
    conf,
    prob_thresh=0.75,
    entropy_thresh=0.40,
    min_valid_candidates=2,
    soft_topk=5,
    source_tag="easy_student",
    teacher_name="student model",
):
    model.eval()
    easy_updates = {}

    logger.info(f"Scanning training set for easy samples with {teacher_name}...")

    with torch.no_grad():
        for batch in data_loader:
            for k, v in batch.items():
                batch[k] = v.to(conf.device, non_blocking=(conf.device.type == "cuda"))

            with _get_autocast_context(conf):
                cand_scores = model.predict(batch)
            masked_scores = cand_scores.float().masked_fill(batch["cand_mask"] == 0, -1e9)
            probs = F.softmax(masked_scores, dim=-1)

            top1_prob, top_indices = torch.max(probs, dim=-1)
            valid_counts = batch["cand_mask"].sum(dim=-1)
            _, normalized_entropy = _compute_entropy_stats(probs, valid_counts)

            seq_mask = batch["seq_mask"].bool()
            easy_mask = (
                (top1_prob >= prob_thresh)
                & (normalized_entropy <= entropy_thresh)
                & seq_mask
                & (valid_counts >= min_valid_candidates)
            )
            b_idxs, s_idxs = torch.where(easy_mask)

            for b, s in zip(b_idxs, s_idxs):
                sample_idx = int(batch["sample_idx"][b].item())
                time_step = int(s.item())
                key = (sample_idx, time_step)

                local_idx = int(top_indices[b, s].item())
                poi_model_id = int(batch["cand_poi_ids"][b, s, local_idx].item())
                raw_poi_id = poi_model_id - MODEL_ID_OFFSET
                if raw_poi_id < 0:
                    continue

                main_cat_id = processor.get_main_cat_id_by_poi_id(raw_poi_id)
                if main_cat_id == -1:
                    continue

                confidence = float(top1_prob[b, s].item())
                soft_poi_ids, soft_probs = _extract_soft_target(batch, probs, b, s, soft_topk)
                if not soft_poi_ids:
                    continue

                if key in easy_updates:
                    continue

                easy_updates[key] = {
                    "pseudo_poi_id": poi_model_id,
                    "main_cat_id": main_cat_id,
                    "main_cat_name": processor.idx2main_cat.get(main_cat_id, "Unknown"),
                    "confidence": confidence,
                    "top1_prob": confidence,
                    "normalized_entropy": float(normalized_entropy[b, s].item()),
                    "soft_poi_ids": soft_poi_ids,
                    "soft_probs": soft_probs,
                    "source": source_tag,
                }

    if not easy_updates:
        logger.info("Easy-sample cache refreshed: 0")
        return {}

    logger.info(
        f"Easy-sample cache refreshed: {len(easy_updates)} "
        f"(prob_thresh={prob_thresh:.2f}, entropy_thresh={entropy_thresh:.2f}, "
        f"min_valid_candidates={min_valid_candidates}, "
        f"soft_topk={soft_topk}, policy=ALL_EASY)"
    )
    return easy_updates