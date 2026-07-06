"""Trajectory-level Transformer diffusion denoiser for weakly supervised POI assignment.



1. Instance commitment provides q_IC and a step-wise reliability weight
   r_t = alpha_t * ce_weight_t.
2. The clean target is an annealed/sharpened q_IC distribution, not a manually
   constructed q_ctx teacher.
3. The entire trajectory belief sequence is corrupted and partially masked.
4. A Transformer denoiser processes the whole noisy belief sequence and predicts
   a refined belief distribution for every step. Thus, each step can use temporal
   evidence from neighboring trajectory steps through self-attention.
5. Stage III still uses q_diff as the generated soft-CE target, while the
   diffusion fitting loss is jointly optimized as L_diff.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceConditionedDiffusionSharpening(nn.Module):
    """Trajectory-level Transformer belief denoiser.

    The class name is kept unchanged so that the existing training code and
    checkpoints remain compatible at the import/API level. Internally, the module
    no longer builds an explicit q_ctx/q_teacher branch. Instead, it denoises the
    whole trajectory belief sequence Q=[q_1,...,q_T].
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
        ctx_temperature: float = 0.50,                  # kept for backward compatibility
        context_mix_max: float = 0.20,                  # kept for backward compatibility
        context_loss_weight: float = 0.20,              # used as masked-step denoising loss weight
        context_anchor_min_weight: float = 0.10,        # kept for backward compatibility
        teacher_temp_min: float = 0.55,
        teacher_temp_max: float = 0.85,
        reverse_kl_weight: float = 0.20,
        input_noise_min: float = 0.05,
        input_noise_max: float = 0.45,
        quality_gate_mode: str = "relaxed",
        gate_entropy_threshold: float = 0.10,
        gate_margin_threshold: float = 0.80,
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
        self.mask_loss_weight = float(context_loss_weight)
        self.teacher_temp_min = float(teacher_temp_min)
        self.teacher_temp_max = float(teacher_temp_max)
        self.reverse_kl_weight = float(reverse_kl_weight)
        self.input_noise_min = float(input_noise_min)
        self.input_noise_max = float(input_noise_max)
        self.quality_gate_mode = str(quality_gate_mode).strip().lower()
        if self.quality_gate_mode not in {"strict", "relaxed", "none"}:
            raise ValueError(
                "quality_gate_mode must be one of {'strict', 'relaxed', 'none'}, "
                f"got {quality_gate_mode!r}"
            )
        self.gate_entropy_threshold = float(gate_entropy_threshold)
        self.gate_margin_threshold = float(gate_margin_threshold)

        # Distribution-only embedding fallback: q_t in R^K -> hidden.
        self.dist_proj = nn.Sequential(
            nn.Linear(self.max_candidates, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        self.pos_emb = nn.Embedding(512, self.hidden_dim)
        self.iter_emb = nn.Embedding(self.refine_steps, self.hidden_dim)
        self.semantic_mask_token = nn.Parameter(torch.zeros(self.hidden_dim))

        # Optional semantic belief embedding: z_t=sum_j q_{t,j} e_{t,j}.
        self.candidate_proj = None
        if condition_dim is not None and int(condition_dim) > 0:
            if int(condition_dim) == self.hidden_dim:
                self.candidate_proj = nn.Identity()
            else:
                self.candidate_proj = nn.Linear(int(condition_dim), self.hidden_dim)

        # Optional main trajectory condition h_t from the assignment model.
        self.cond_proj = None
        self.condition_mask_token = None
        if condition_dim is not None and int(condition_dim) > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(int(condition_dim), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            )
            self.condition_mask_token = nn.Parameter(torch.zeros(int(condition_dim)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.temporal_denoiser = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        # Residual update head used in every reverse refinement step.
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.max_candidates),
        )
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

    def _build_mask_steps(
        self,
        eligible: torch.Tensor,
        reliability: torch.Tensor,
        seq_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        """Mask the denoiser input at selected steps, not the supervision target."""
        if not self.training or self.mask_prob <= 0.0:
            return torch.zeros_like(eligible, dtype=torch.bool)

        rand_mask = torch.rand_like(reliability.float()).lt(self.mask_prob)
        mask_steps = rand_mask & eligible & seq_mask_bool

        # Ensure every sequence containing eligible steps has at least one masked
        # step. This makes the training objective explicitly context-reconstructive.
        if eligible.any():
            no_mask_rows = eligible.any(dim=1) & (~mask_steps.any(dim=1))
            if no_mask_rows.any():
                scores = reliability.masked_fill(~eligible, -1.0)
                best_t = scores.argmax(dim=1)
                rows = torch.where(no_mask_rows)[0]
                mask_steps[rows, best_t[rows]] = True
        return mask_steps

    def _belief_embedding(
        self,
        belief: torch.Tensor,
        cand_vectors: Optional[torch.Tensor],
        cand_mask_bool: torch.Tensor,
        mask_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Embed each step's belief distribution into a temporal token."""
        if cand_vectors is not None and self.candidate_proj is not None:
            cand_h = self.candidate_proj(cand_vectors.float())
            cand_h = cand_h * cand_mask_bool.unsqueeze(-1).float()
            x = (belief.unsqueeze(-1) * cand_h).sum(dim=2)
        else:
            x = self.dist_proj(belief)

        # When a step is masked, hide its local belief token. The target is still
        # kept in the loss, forcing recovery from the trajectory context.
        mask_token = self.semantic_mask_token.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
        x = torch.where(mask_steps.unsqueeze(-1), mask_token, x)
        return x

    def _denoise_once(
        self,
        q_iter: torch.Tensor,
        logits_iter: torch.Tensor,
        cand_mask_bool: torch.Tensor,
        seq_mask_bool: torch.Tensor,
        mask_steps: torch.Tensor,
        k: int,
        condition: Optional[torch.Tensor],
        cand_vectors: Optional[torch.Tensor],
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = q_iter.shape
        x = self._belief_embedding(q_iter, cand_vectors, cand_mask_bool, mask_steps)

        pos_ids = torch.arange(seq_len, device=q_iter.device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.pos_emb(pos_ids)
        x = x + self.iter_emb.weight[k].view(1, 1, -1)

        if self.cond_proj is not None and condition is not None:
            cond = condition.detach() if self.detach_condition else condition
            if self.mask_condition_on_masked_steps and self.condition_mask_token is not None:
                mask_cond = self.condition_mask_token.to(device=cond.device, dtype=cond.dtype).view(1, 1, -1)
                cond = torch.where(mask_steps.unsqueeze(-1), mask_cond, cond)
            x = x + self.cond_proj(cond.float())

        h = self.temporal_denoiser(x, src_key_padding_mask=~seq_mask_bool)
        delta = self.delta_head(h).masked_fill(~cand_mask_bool, -1e9)
        scale = self.step_scale.sigmoid()
        logits_next = (logits_iter + scale * delta).masked_fill(~cand_mask_bool, -1e9)

        # Simulated annealing in reverse refinement: later steps use lower
        # temperature, making q_diff sharper and more suitable as a soft-CE target.
        if temperature.dim() == logits_next.dim() - 1:
            temperature = temperature.unsqueeze(-1)
        q_next = self._normalize_distribution(torch.softmax(logits_next / temperature.clamp_min(1e-6), dim=-1), cand_mask_bool)
        return q_next, logits_next

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

        valid_counts = cand_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform = cand_mask.float() / valid_counts

        # Reliability-adaptive corruption over the whole trajectory belief sequence.
        r = base_reliability.unsqueeze(-1)
        noise_strength = self.input_noise_min + (1.0 - r) * (self.input_noise_max - self.input_noise_min)
        noise_strength = noise_strength.clamp(0.0, 1.0)
        q_noisy = self._normalize_distribution((1.0 - noise_strength) * q_ic.detach() + noise_strength * uniform, cand_mask_bool)
        q_noisy = torch.where(eligible.unsqueeze(-1), q_noisy, p_base.detach())

        mask_steps = self._build_mask_steps(eligible=eligible, reliability=base_reliability, seq_mask_bool=seq_mask_bool)
        q_input = torch.where(mask_steps.unsqueeze(-1), uniform, q_noisy)
        q_input = self._normalize_distribution(q_input, cand_mask_bool)

        # Annealed target from q_IC. No explicit q_ctx branch is used.
        target_temp = self.teacher_temp_min + (1.0 - base_reliability) * (self.teacher_temp_max - self.teacher_temp_min)
        q_target_sharp = self._temperature_sharpen(q_ic.detach(), target_temp, cand_mask_bool)
        mix = min(max(self.target_soft_mix, 0.0), 1.0)
        q_target = mix * q_target_sharp + (1.0 - mix) * q_ic.detach()
        q_target = torch.where(eligible.unsqueeze(-1), q_target, p_base.detach())
        q_target = self._normalize_distribution(q_target, cand_mask_bool).detach()

        # Iterative trajectory-level denoising with temperature annealing.
        q_iter = q_input
        logits_iter = q_iter.clamp_min(1e-12).log().masked_fill(~cand_mask_bool, -1e9)
        for k in range(self.refine_steps):
            if self.refine_steps == 1:
                frac = 1.0
            else:
                frac = float(k) / float(self.refine_steps - 1)
            anneal_temp = self.temp_max + frac * (self.temp_min - self.temp_max)
            step_temp = torch.full_like(base_reliability, anneal_temp)
            q_iter, logits_iter = self._denoise_once(
                q_iter=q_iter,
                logits_iter=logits_iter,
                cand_mask_bool=cand_mask_bool,
                seq_mask_bool=seq_mask_bool,
                mask_steps=mask_steps,
                k=k,
                condition=condition,
                cand_vectors=cand_vectors,
                temperature=step_temp,
            )

        q_diff = q_iter
        log_q_diff = q_diff.clamp_min(1e-12).log()
        log_q_target = q_target.clamp_min(1e-12).log()

        forward_kl = (q_target * (log_q_target - log_q_diff)).sum(dim=-1)
        reverse_kl = (q_diff * (log_q_diff - log_q_target)).sum(dim=-1)
        align_step = forward_kl + self.reverse_kl_weight * reverse_kl

        entropy_diff = self._entropy(q_diff)
        entropy_ic = self._entropy(q_ic.detach())
        entropy_target = self._entropy(q_target.detach())
        margin_ic = self._prob_margin(q_ic.detach())
        margin_target = self._prob_margin(q_target.detach())
        margin_diff = self._prob_margin(q_diff)

        entropy_step = F.relu(entropy_diff - entropy_target)
        margin_step = F.relu(margin_target + self.target_margin - margin_diff)

        train_weight = (base_reliability * eligible_float).detach()
        denom = train_weight.sum().clamp_min(1.0)
        loss_all = (align_step * train_weight).sum() / denom

        mask_weight = (train_weight * mask_steps.float()).detach()
        mask_denom = mask_weight.sum().clamp_min(1.0)
        loss_masked = (align_step * mask_weight).sum() / mask_denom

        loss_entropy = (entropy_step * train_weight).sum() / denom
        loss_margin = (margin_step * train_weight).sum() / denom
        loss = (
            loss_all
            + self.mask_loss_weight * loss_masked
            + self.entropy_weight * loss_entropy
            + self.margin_weight * loss_margin
        )

        with torch.no_grad():
            soft_agree = (q_ic.detach() * q_diff.detach()).sum(dim=-1).clamp(0.0, 1.0)
            top_ic = q_ic.detach().argmax(dim=-1)
            top_diff = q_diff.detach().argmax(dim=-1)
            top_agree = top_ic.eq(top_diff)
            sharper = entropy_diff.le(entropy_ic)
            margin_better = margin_diff.ge(margin_ic)

            valid_k_for_gate = cand_mask.float().sum(dim=-1).clamp_min(2.0)
            log_k_for_gate = valid_k_for_gate.log().clamp_min(1e-6)
            entropy_ic_norm = (entropy_ic / log_k_for_gate).clamp(0.0, 1.0)
            entropy_diff_norm = (entropy_diff / log_k_for_gate).clamp(0.0, 1.0)

            if self.quality_gate_mode == "none":
                entropy_gate = torch.ones_like(eligible, dtype=torch.bool)
                margin_gate = torch.ones_like(eligible, dtype=torch.bool)
                relative_quality_mask = eligible
            elif self.quality_gate_mode == "relaxed":
                entropy_threshold = torch.full_like(entropy_ic_norm, self.gate_entropy_threshold)
                margin_threshold = torch.full_like(margin_ic, self.gate_margin_threshold)
                entropy_gate = entropy_diff_norm.le(torch.maximum(entropy_ic_norm, entropy_threshold))
                margin_gate = margin_diff.ge(torch.minimum(margin_ic, margin_threshold))
                relative_quality_mask = entropy_gate & margin_gate
            else:
                entropy_gate = sharper
                margin_gate = margin_better
                relative_quality_mask = sharper & margin_better

            relative_quality_mask = relative_quality_mask & eligible
            diff_commitment_base_weight = (base_reliability * eligible_float).detach()
            diff_commitment_weight = (base_reliability * eligible_float * relative_quality_mask.float()).detach()

            valid_denom = seq_weight.sum().clamp_min(1.0)
            eligible_denom = eligible_float.sum().clamp_min(1.0)
            avg_gate = base_reliability.sum() / valid_denom
            mask_rate = mask_steps.float().sum() / eligible_denom
            agree_rate = (soft_agree * eligible_float).sum() / eligible_denom
            sharper_rate = (sharper.float() * eligible_float).sum() / eligible_denom
            margin_better_rate = (margin_better.float() * eligible_float).sum() / eligible_denom
            entropy_gate_rate = (entropy_gate.float() * eligible_float).sum() / eligible_denom
            margin_gate_rate = (margin_gate.float() * eligible_float).sum() / eligible_denom
            use_diff_rate = diff_commitment_weight.gt(0).float().sum() / valid_denom
            avg_eta = diff_commitment_weight.sum() / valid_denom
            avg_diff_commitment_quality = ((0.5 * sharper.float() + 0.5 * margin_better.float()) * eligible_float).sum() / eligible_denom
            top_agree_rate = (top_agree.float() * eligible_float).sum() / eligible_denom

        diff_commitment_step_loss = -(q_diff.detach() * base_log_probs).sum(dim=-1)
        q_final = q_diff.detach()

        zero = torch.zeros((), device=cand_scores.device, dtype=cand_scores.dtype)
        return {
            "loss": loss,
            "loss_align": loss_all.detach(),
            "loss_context": loss_masked.detach(),  # logged as CtxLoss; now means masked denoising loss
            "loss_entropy": loss_entropy.detach(),
            "loss_margin": loss_margin.detach(),
            "commitment_step_loss": diff_commitment_step_loss,
            "diff_commitment_step_loss": diff_commitment_step_loss,
            "diff_commitment_weight": diff_commitment_weight,
            "diff_commitment_base_weight": diff_commitment_base_weight,
            "diff_commitment_mask": (diff_commitment_weight.gt(0.0) & seq_mask_bool).detach(),
            "diff_commitment_base_mask": (diff_commitment_base_weight.gt(0.0) & seq_mask_bool).detach(),
            "diff_commitment_quality": (0.5 * sharper.float() + 0.5 * margin_better.float()).detach(),
            "avg_diff_commitment_quality": avg_diff_commitment_quality.detach(),
            "q_ic": q_ic.detach(),
            "q_soft": q_ic.detach(),
            "q_sharp": q_target.detach(),
            "q_ctx": q_diff.detach(),       # compatibility only; no explicit q_ctx branch
            "q_teacher": q_target.detach(),
            "q_target": q_target.detach(),
            "q_diff": q_diff,
            "q_final": q_final,
            "avg_gate": avg_gate.detach(),
            "avg_weight": avg_gate.detach(),
            "agree_rate": agree_rate.detach(),
            "top_agree_rate": top_agree_rate.detach(),
            "sharper_rate": sharper_rate.detach(),
            "margin_better_rate": margin_better_rate.detach(),
            "entropy_gate_rate": entropy_gate_rate.detach(),
            "margin_gate_rate": margin_gate_rate.detach(),
            "use_diff_rate": use_diff_rate.detach(),
            "avg_eta": avg_eta.detach(),
            "mask_rate": mask_rate.detach(),
            "avg_ctx_lambda": zero.detach(),
            "avg_ctx_conf": zero.detach(),
            "avg_ctx_agree": zero.detach(),
            "avg_fusion_gate": zero.detach(),
            "safety_rate": relative_quality_mask.float().mul(eligible_float).sum().div(eligible_denom).detach(),
            "avg_entropy_soft": (entropy_ic * seq_weight).sum().div(valid_denom).detach(),
            "avg_entropy_diff": (entropy_diff * seq_weight).sum().div(valid_denom).detach(),
            "avg_margin_soft": (margin_ic * seq_weight).sum().div(valid_denom).detach(),
            "avg_margin_diff": (margin_diff * seq_weight).sum().div(valid_denom).detach(),
            "avg_noise_step": torch.zeros((), device=cand_scores.device),
        }
