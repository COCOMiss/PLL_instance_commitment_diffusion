import torch
import torch.nn.functional as F


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
        return None, target_mass

    return target_dist / target_mass.clamp_min(1e-12), target_mass


def compute_commitment_soft_ce_loss_per_step(batch, cand_scores, pseudo_cache):
    batch_size, seq_len = batch["seq_mask"].shape
    step_losses = cand_scores.new_zeros((batch_size, seq_len))
    ce_valid_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=cand_scores.device)

    if not pseudo_cache:
        return step_losses, ce_valid_mask

    masked_scores = cand_scores.masked_fill(batch["cand_mask"] == 0, -1e9)

    for b in range(batch_size):
        sample_idx = int(batch["sample_idx"][b].item())
        valid_len = int(batch["seq_mask"][b].sum().item())

        for t in range(valid_len):
            info = pseudo_cache.get((sample_idx, t))
            if not info:
                continue

            soft_poi_ids = info.get("soft_poi_ids")
            soft_probs = info.get("soft_probs")
            if not soft_poi_ids or not soft_probs:
                continue

            target_dist, target_mass = _build_step_soft_target(
                candidate_poi_ids=batch["cand_poi_ids"][b, t],
                soft_poi_ids=soft_poi_ids,
                soft_probs=soft_probs,
                device=masked_scores.device,
                dtype=masked_scores.dtype,
            )
            if target_dist is None or float(target_mass.item()) <= 0.0:
                continue

            log_probs = F.log_softmax(masked_scores[b, t], dim=-1)
            step_losses[b, t] = -(target_dist * log_probs).sum()
            ce_valid_mask[b, t] = True

    return step_losses, ce_valid_mask
