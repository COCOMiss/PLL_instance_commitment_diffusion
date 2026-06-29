"""Diffusion-generated soft target for weakly supervised POI assignment.

This version makes diffusion play the role of the previous direct soft-CE
teacher, while still avoiding direct CE(q_IC, p_theta):

1. Instance commitment mines q_IC and a sample-wise reliability weight
   r_t = alpha_t * ce_weight_t.
2. q_IC is sharpened with an instance-dependent temperature; high-reliability
   steps receive sharper teachers.
3. A trajectory context branch predicts q_ctx from the belief sequence, candidate
   semantic vectors, and optionally the main model's user_vector.  The model
   user_vector is fused with the context representation through a learnable gate.
4. The final diffusion teacher is dominated by q_sharp and only softly corrected
   by q_ctx using an instance-dependent context ratio.  The teacher is sharpened
   again with an instance-dependent temperature.
5. The refiner denoises a partially noised q_IC/q_soft distribution into q_diff.
6. During the model-distillation phase, q_diff is used as the CE target, with the
   same commitment weight as the previous soft-CE version.  If the quality gate
   is enabled, it only filters out q_diff that is not sharper than q_IC; it does
   not multiply by an extra heuristic quality score.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceConditionedDiffusionSharpening(nn.Module):
    """Instance-conditioned trajectory belief refiner.

    The class name is kept unchanged for backward compatibility with the
    existing training code.
    """

    def __init__(
        self,
        max_candidates: int,
        hidden_dim: int = 64,
        condition_dim: Optional[int] = None,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.10,
        temp_min: float = 0.40,
        temp_max: float = 0.80,
        entropy_weight: float = 0.02,
        margin_weight: float = 0.05,
        target_margin: float = 0.05,
        mask_prob: float = 0.10,
        detach_condition: bool = True,
        refine_steps: int = 3,
        target_soft_mix: float = 1.0,
        mask_condition_on_masked_steps: bool = False,
        min_mask_weight: float = 1e-8,
        ctx_temperature: float = 0.50,
        context_mix_max: float = 0.20,
        context_loss_weight: float = 0.20,
        context_anchor_min_weight: float = 0.10,
        teacher_temp_min: float = 0.55,
        teacher_temp_max: float = 0.85,
        reverse_kl_weight: float = 0.20,
        input_noise_min: float = 0.05,
        input_noise_max: float = 0.45,
    ) -> None:
        super().__init__()
        self.max_candidates = int(max_candidates)
        self.hidden_dim = int(hidden_dim)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.entropy_weight = float(entropy_weight)
        self.margin_weight = float(margin_weight)
        self.target_margin = float(target_margin)
        self.mask_prob = float(mask_prob)
        self.detach_condition = bool(detach_condition)
        self.refine_steps = max(int(refine_steps), 1)
        self.target_soft_mix = float(target_soft_mix)
        self.mask_condition_on_masked_steps = bool(mask_condition_on_masked_steps)
        self.min_mask_weight = float(min_mask_weight)
        self.ctx_temperature = float(ctx_temperature)
        self.context_mix_max = float(context_mix_max)
        self.context_loss_weight = float(context_loss_weight)
        self.context_anchor_min_weight = float(context_anchor_min_weight)
        self.teacher_temp_min = float(teacher_temp_min)
        self.teacher_temp_max = float(teacher_temp_max)
        self.reverse_kl_weight = float(reverse_kl_weight)
        self.input_noise_min = float(input_noise_min)
        self.input_noise_max = float(input_noise_max)


        self.dist_proj = nn.Sequential(
            nn.Linear(self.max_candidates, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.pos_emb = nn.Embedding(512, self.hidden_dim)
        self.semantic_mask_token = nn.Parameter(torch.zeros(self.hidden_dim))

        self.cond_proj = None
        self.condition_mask_token = None
        self.candidate_proj = None
        self.fusion_gate = None
        if condition_dim is not None and int(condition_dim) > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(int(condition_dim), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            )
            self.condition_mask_token = nn.Parameter(torch.zeros(int(condition_dim)))
            if int(condition_dim) == self.hidden_dim:
                self.candidate_proj = nn.Identity()
            else:
                self.candidate_proj = nn.Linear(int(condition_dim), self.hidden_dim)
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        self.iter_emb = nn.Embedding(self.refine_steps, self.hidden_dim)
        self.refine_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(self.hidden_dim, self.max_candidates)
        self.context_prior_head = nn.Linear(self.hidden_dim, self.max_candidates)
        self.step_scale = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))

    @staticmethod
    def _normalize_distribution(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = probs.float() * mask.float()
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    @staticmethod
    def _prob_margin(probs: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
        if topk.size(-1) == 1:
            return topk[..., 0]
        return (topk[..., 0] - topk[..., 1]).clamp_min(0.0)

    def _safe_q_ic(self, q_ic: torch.Tensor, cand_mask_bool: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_mass = (q_ic.float() * cand_mask_bool.float()).sum(dim=-1)
        has_ic = raw_mass.gt(1e-12)
        q_norm = self._normalize_distribution(q_ic, cand_mask_bool)
        return q_norm, has_ic

    def _temperature_sharpen(
        self,
        probs: torch.Tensor,
        temperature: torch.Tensor,
        cand_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        if temperature.dim() == probs.dim() - 1:
            temperature = temperature.unsqueeze(-1)
        logits = probs.clamp_min(1e-12).log() / temperature.clamp_min(1e-6)
        logits = logits.masked_fill(~cand_mask_bool, -1e9)
        return self._normalize_distribution(torch.softmax(logits, dim=-1), cand_mask_bool)

    def _adaptive_sharpen(
        self,
        q_ic: torch.Tensor,
        reliability: torch.Tensor,
        cand_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        r = reliability.float().clamp(0.0, 1.0)
        temp = self.temp_min + (1.0 - r) * (self.temp_max - self.temp_min)
        return self._temperature_sharpen(q_ic, temp, cand_mask_bool)

    def _build_mask_steps(
        self,
        eligible: torch.Tensor,
        reliability: torch.Tensor,
        seq_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or self.mask_prob <= 0.0:
            return torch.zeros_like(eligible, dtype=torch.bool)

        # Higher-reliability steps are still sampled, but the training loss is
        # not restricted to masked positions.  Masking only prevents trivial
        # context copying in the encoder.
        rand_mask = torch.rand_like(reliability.float()).lt(self.mask_prob)
        mask_steps = rand_mask & eligible & seq_mask_bool

        if eligible.any():
            no_mask_rows = eligible.any(dim=1) & (~mask_steps.any(dim=1))
            if no_mask_rows.any():
                scores = reliability.masked_fill(~eligible, -1.0)
                best_t = scores.argmax(dim=1)
                rows = torch.where(no_mask_rows)[0]
                mask_steps[rows, best_t[rows]] = True
        return mask_steps

    def _neighbor_reliability(self, reliability: torch.Tensor, seq_mask_bool: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = reliability.shape
        pos = torch.arange(seq_len, device=reliability.device).float()
        dist = (pos.view(1, -1) - pos.view(-1, 1)).abs()
        kernel = torch.exp(-dist)
        kernel.fill_diagonal_(0.0)
        valid = seq_mask_bool.float()
        weighted = torch.matmul(reliability * valid, kernel.T)
        denom = torch.matmul(valid, kernel.T).clamp_min(1e-6)
        return (weighted / denom).clamp(0.0, 1.0)

    def _belief_semantic_embedding(
        self,
        belief: torch.Tensor,
        cand_vectors: Optional[torch.Tensor],
        cand_mask_bool: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if cand_vectors is None or self.candidate_proj is None:
            return None
        cand_h = self.candidate_proj(cand_vectors.float())
        cand_h = cand_h * cand_mask_bool.unsqueeze(-1).float()
        return (belief.unsqueeze(-1) * cand_h).sum(dim=2)

    def _context_prior(
        self,
        context_hidden: torch.Tensor,
        cand_vectors: Optional[torch.Tensor],
        cand_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        if cand_vectors is not None and self.candidate_proj is not None:
            cand_h = self.candidate_proj(cand_vectors.float())
            ctx = F.normalize(context_hidden.float(), p=2, dim=-1, eps=1e-8)
            cand = F.normalize(cand_h.float(), p=2, dim=-1, eps=1e-8)
            logits = (ctx.unsqueeze(2) * cand).sum(dim=-1) / max(self.ctx_temperature, 1e-6)
        else:
            logits = self.context_prior_head(context_hidden.float()) / max(self.ctx_temperature, 1e-6)
        logits = logits.masked_fill(~cand_mask_bool, -1e9)
        return self._normalize_distribution(torch.softmax(logits, dim=-1), cand_mask_bool)

    def forward(
        self,
        cand_scores: torch.Tensor,
        cand_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        q_soft: torch.Tensor,
        step_weight: torch.Tensor,
        instance_alpha: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        cand_vectors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if cand_scores.dim() != 3:
            raise ValueError(f"cand_scores must have shape [B,T,K], got {tuple(cand_scores.shape)}")
        batch_size, seq_len, num_candidates = cand_scores.shape
        if num_candidates != self.max_candidates:
            raise ValueError(f"Expected K={self.max_candidates}, got K={num_candidates}.")

        cand_mask_bool = cand_mask.bool()
        seq_mask_bool = seq_mask.bool()
        seq_weight = seq_mask.float()

        masked_scores = cand_scores.float().masked_fill(~cand_mask_bool, -1e9)
        base_log_probs = F.log_softmax(masked_scores, dim=-1)
        p_base = self._normalize_distribution(torch.softmax(masked_scores, dim=-1), cand_mask_bool)

        q_ic, has_ic = self._safe_q_ic(q_soft, cand_mask_bool)
        base_reliability = (seq_weight * step_weight.float().clamp_min(0.0)).detach().clamp(0.0, 1.0)
        eligible = has_ic & base_reliability.gt(self.min_mask_weight) & seq_mask_bool
        eligible_float = eligible.float()

        q_sharp = self._adaptive_sharpen(q_ic.detach(), base_reliability, cand_mask_bool)

        valid_counts = cand_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform = cand_mask.float() / valid_counts

        # Reliability-adaptive partial noising: high-confidence steps remain close
        # to q_IC; lower-confidence committed steps are more strongly smoothed.
        r = base_reliability.unsqueeze(-1)
        noise_strength = self.input_noise_min + (1.0 - r) * (self.input_noise_max - self.input_noise_min)
        noise_strength = noise_strength.clamp(0.0, 1.0)
        q_noisy = self._normalize_distribution((1.0 - noise_strength) * q_ic.detach() + noise_strength * uniform, cand_mask_bool)
        q_noisy = torch.where(eligible.unsqueeze(-1), q_noisy, p_base.detach())

        context_anchor = has_ic & base_reliability.ge(self.context_anchor_min_weight) & seq_mask_bool
        q_context = torch.where(context_anchor.unsqueeze(-1), q_ic.detach(), p_base.detach())

        mask_steps = self._build_mask_steps(
            eligible=eligible,
            reliability=base_reliability,
            seq_mask_bool=seq_mask_bool,
        )

        q_context_masked = torch.where(mask_steps.unsqueeze(-1), q_noisy, q_context)
        q_context_masked = q_context_masked.masked_fill(~cand_mask_bool, 0.0)
        semantic_emb = self._belief_semantic_embedding(q_context_masked, cand_vectors, cand_mask_bool)
        if semantic_emb is not None:
            x = semantic_emb
            mask_sem = self.semantic_mask_token.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
            x = torch.where(mask_steps.unsqueeze(-1), mask_sem, x)
        else:
            x = self.dist_proj(q_context_masked)

        pos_ids = torch.arange(seq_len, device=cand_scores.device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.pos_emb(pos_ids)

        cond_h = None
        if self.cond_proj is not None and condition is not None:
            cond = condition.detach() if self.detach_condition else condition
            cond_for_input = cond
            if self.mask_condition_on_masked_steps and self.condition_mask_token is not None:
                mask_cond = self.condition_mask_token.to(device=cond.device, dtype=cond.dtype).view(1, 1, -1)
                cond_for_input = torch.where(mask_steps.unsqueeze(-1), mask_cond, cond)
            cond_h = self.cond_proj(cond.float())
            x = x + self.cond_proj(cond_for_input.float())

        context_hidden = self.context_encoder(x, src_key_padding_mask=~seq_mask_bool)

        # Gated fusion between belief-level context and the main model trajectory
        # representation.  The gate is learned, while reliability is provided as
        # a sample-wise signal.
        if cond_h is not None and self.fusion_gate is not None:
            gate_in = torch.cat([context_hidden, cond_h, base_reliability.unsqueeze(-1)], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(gate_in))
            fused_context = gate * context_hidden + (1.0 - gate) * cond_h
        else:
            gate = torch.ones(batch_size, seq_len, 1, device=cand_scores.device, dtype=context_hidden.dtype)
            fused_context = context_hidden

        q_ctx = self._context_prior(fused_context, cand_vectors, cand_mask_bool)

        with torch.no_grad():
            ctx_ic_agree = (q_sharp * q_ctx.detach()).sum(dim=-1).clamp(0.0, 1.0)
            valid_k = cand_mask.float().sum(dim=-1).clamp_min(2.0)
            ctx_conf = (1.0 - self._entropy(q_ctx.detach()) / valid_k.log().clamp_min(1e-6)).clamp(0.0, 1.0)
            neigh_rel = self._neighbor_reliability(base_reliability, seq_mask_bool)
            ctx_reliability = (0.45 * ctx_ic_agree + 0.35 * ctx_conf + 0.20 * neigh_rel).clamp(0.0, 1.0)
            lambda_ctx = (self.context_mix_max * ctx_reliability * base_reliability * eligible_float).clamp(0.0, 1.0)

        # Soft teacher: q_sharp is the main teacher; q_ctx is only a conservative
        # context correction.  This is more stable than a strong PoE when the goal
        # is to let diffusion replace direct soft CE.
        q_teacher = (1.0 - lambda_ctx.unsqueeze(-1)) * q_sharp + lambda_ctx.unsqueeze(-1) * q_ctx
        q_teacher = self._normalize_distribution(q_teacher, cand_mask_bool)

        # Teacher sharpening is also instance-adaptive: reliable steps get lower
        # temperature and therefore sharper q_target.
        teacher_temp = self.teacher_temp_min + (1.0 - base_reliability) * (self.teacher_temp_max - self.teacher_temp_min)
        q_target = self._temperature_sharpen(q_teacher, teacher_temp, cand_mask_bool)
        mix = min(max(self.target_soft_mix, 0.0), 1.0)
        q_target = mix * q_target + (1.0 - mix) * p_base.detach()
        q_target = torch.where(eligible.unsqueeze(-1), q_target, p_base.detach())
        q_target = self._normalize_distribution(q_target, cand_mask_bool).detach()

        # Iterative denoising/refinement starts from partially noised q_IC, not
        # from uniform.  This makes recovery to q_target much easier and closer
        # to the previous successful soft-CE+diffusion behavior.
        q_iter = q_noisy
        logits_iter = q_iter.clamp_min(1e-12).log().masked_fill(~cand_mask_bool, -1e9)
        final_logits = logits_iter
        scale = self.step_scale.sigmoid()
        for k in range(self.refine_steps):
            q_embed = self.dist_proj(q_iter)
            h = torch.cat([q_embed, fused_context + self.iter_emb.weight[k].view(1, 1, -1)], dim=-1)
            delta = self.delta_head(self.refine_proj(h)).masked_fill(~cand_mask_bool, -1e9)
            final_logits = (final_logits + scale * delta).masked_fill(~cand_mask_bool, -1e9)
            q_iter = self._normalize_distribution(torch.softmax(final_logits, dim=-1), cand_mask_bool)

        q_diff = q_iter
        log_q_diff = q_diff.clamp_min(1e-12).log()
        log_q_target = q_target.clamp_min(1e-12).log()

        forward_kl = (q_target * (log_q_target - log_q_diff)).sum(dim=-1)
        reverse_kl = (q_diff * (log_q_diff - log_q_target)).sum(dim=-1)
        align_step = forward_kl + self.reverse_kl_weight * reverse_kl

        context_prior_step = (q_sharp.detach() * (q_sharp.detach().clamp_min(1e-12).log() - q_ctx.clamp_min(1e-12).log())).sum(dim=-1)
        entropy_diff = self._entropy(q_diff)
        entropy_ic = self._entropy(q_ic.detach())
        entropy_target = self._entropy(q_target.detach())
        margin_ic = self._prob_margin(q_ic.detach())
        margin_target = self._prob_margin(q_target.detach())
        margin_diff = self._prob_margin(q_diff)

        entropy_step = F.relu(entropy_diff - entropy_target)
        margin_step = F.relu(margin_target + self.target_margin - margin_diff)

        # Train on all committed steps with their own commitment reliability;
        # masking only affects the input/context corruption, not the supervision
        # coverage.
        train_weight = (base_reliability * eligible_float).detach()
        denom = train_weight.sum().clamp_min(1.0)
        loss_align = (align_step * train_weight).sum() / denom
        loss_context = (context_prior_step * train_weight).sum() / denom
        loss_entropy = (entropy_step * train_weight).sum() / denom
        loss_margin = (margin_step * train_weight).sum() / denom
        loss = (
            loss_align
            + self.context_loss_weight * loss_context
            + self.entropy_weight * loss_entropy
            + self.margin_weight * loss_margin
        )

        with torch.no_grad():
            soft_agree = (q_ic.detach() * q_diff.detach()).sum(dim=-1).clamp(0.0, 1.0)
            top_ic = q_ic.detach().argmax(dim=-1)
            top_diff = q_diff.detach().argmax(dim=-1)
            top_agree = top_ic.eq(top_diff)
            entropy_gain = ((entropy_ic - entropy_diff) / entropy_ic.clamp_min(1e-6)).clamp(-1.0, 1.0)
            margin_gain = ((margin_diff - margin_ic) / (1.0 - margin_ic).clamp_min(1e-6)).clamp(-1.0, 1.0)
            sharper = entropy_diff.le(entropy_ic)
            margin_better = margin_diff.ge(margin_ic)
            # No fixed threshold here: quality gate only requires q_diff to be
            # no worse than q_IC in sharpness and margin.  The actual weight is
            # still exactly the commitment weight.
            relative_quality_mask = sharper & margin_better
            dce_mask = eligible_float
            diff_commitment_base_weight = (base_reliability * dce_mask).detach()
            diff_commitment_weight = (base_reliability * dce_mask * relative_quality_mask.float()).detach()

            valid_denom = seq_weight.sum().clamp_min(1.0)
            eligible_denom = eligible_float.sum().clamp_min(1.0)
            avg_gate = base_reliability.sum() / valid_denom
            mask_rate = mask_steps.float().sum() / eligible_denom
            agree_rate = (soft_agree * eligible_float).sum() / eligible_denom
            sharper_rate = (sharper.float() * eligible_float).sum() / eligible_denom
            margin_better_rate = (margin_better.float() * eligible_float).sum() / eligible_denom
            use_diff_rate = diff_commitment_weight.gt(0).float().sum() / valid_denom
            avg_eta = diff_commitment_weight.sum() / valid_denom
            avg_diff_commitment_quality = ((0.5 * sharper.float() + 0.5 * margin_better.float()) * eligible_float).sum() / eligible_denom
            avg_ctx_lambda = (lambda_ctx * eligible_float).sum() / eligible_denom
            avg_ctx_conf = (ctx_conf * eligible_float).sum() / eligible_denom
            avg_ctx_agree = (ctx_ic_agree * eligible_float).sum() / eligible_denom
            avg_fusion_gate = (gate.squeeze(-1) * seq_weight).sum() / valid_denom

        diff_commitment_step_loss = -(q_diff.detach() * base_log_probs).sum(dim=-1)
        commitment_step_loss = diff_commitment_step_loss
        q_final = q_diff.detach()

        return {
            "loss": loss,
            "loss_align": loss_align.detach(),
            "loss_context": loss_context.detach(),
            "loss_entropy": loss_entropy.detach(),
            "loss_margin": loss_margin.detach(),
            "commitment_step_loss": commitment_step_loss,
            "diff_commitment_step_loss": diff_commitment_step_loss,
            "diff_commitment_weight": diff_commitment_weight,
            "diff_commitment_base_weight": diff_commitment_base_weight,
            "diff_commitment_mask": (diff_commitment_weight.gt(0.0) & seq_mask_bool).detach(),
            "diff_commitment_base_mask": (diff_commitment_base_weight.gt(0.0) & seq_mask_bool).detach(),
            "diff_commitment_quality": (0.5 * sharper.float() + 0.5 * margin_better.float()).detach(),
            "avg_diff_commitment_quality": avg_diff_commitment_quality.detach(),
            "q_ic": q_ic.detach(),
            "q_soft": q_ic.detach(),
            "q_sharp": q_sharp.detach(),
            "q_ctx": q_ctx.detach(),
            "q_teacher": q_teacher.detach(),
            "q_target": q_target.detach(),
            "q_diff": q_diff,
            "q_final": q_final,
            "avg_gate": avg_gate.detach(),
            "avg_weight": avg_gate.detach(),
            "agree_rate": agree_rate.detach(),
            "top_agree_rate": (top_agree.float() * eligible_float).sum().div(eligible_denom).detach(),
            "sharper_rate": sharper_rate.detach(),
            "margin_better_rate": margin_better_rate.detach(),
            "use_diff_rate": use_diff_rate.detach(),
            "avg_eta": avg_eta.detach(),
            "mask_rate": mask_rate.detach(),
            "avg_ctx_lambda": avg_ctx_lambda.detach(),
            "avg_ctx_conf": avg_ctx_conf.detach(),
            "avg_ctx_agree": avg_ctx_agree.detach(),
            "avg_fusion_gate": avg_fusion_gate.detach(),
            "safety_rate": relative_quality_mask.float().mul(eligible_float).sum().div(eligible_denom).detach(),
            "avg_entropy_soft": (entropy_ic * seq_weight).sum().div(valid_denom).detach(),
            "avg_entropy_diff": (entropy_diff * seq_weight).sum().div(valid_denom).detach(),
            "avg_margin_soft": (margin_ic * seq_weight).sum().div(valid_denom).detach(),
            "avg_margin_diff": (margin_diff * seq_weight).sum().div(valid_denom).detach(),
            "avg_noise_step": torch.zeros((), device=cand_scores.device),
        }
