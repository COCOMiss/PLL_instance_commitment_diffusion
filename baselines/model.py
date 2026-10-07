"""Shared features only: no project PLL loss, commitment, gate or diffusion."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import SOURCE  # installs the original source directory for its absolute imports
from model import UserTrajectoryModel, CandidatePOIEncoder


class POIAssignmentEncoder(nn.Module):
    def __init__(self, processor, embedding_path, embed_dim=64, projection_dim=128, max_seq_len=20):
        super().__init__()
        self.user_model = UserTrajectoryModel(len(processor.user2idx) + 1, 25,
                                              embed_dim=embed_dim, max_len=max(200, max_seq_len))
        self.candidate_model = CandidatePOIEncoder(processor.num_pois + 1, str(embedding_path),
                                                   embed_dim=embed_dim, freeze_qwen=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.projector = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                       nn.Linear(embed_dim, projection_dim))

    def encode(self, batch):
        return self.user_model(batch["user_id"], batch["time_slot"], batch["center_coord"],
                               src_key_padding_mask=~batch["seq_mask"].bool())

    def score_features(self, hidden, ids, probs, main_probs, dists, other_feats):
        vectors = self.candidate_model(ids, probs, main_probs, dists, other_feats)
        return self.logit_scale.exp().clamp(max=100) * (
            F.normalize(hidden, dim=-1).unsqueeze(-2) * F.normalize(vectors, dim=-1)).sum(-1)

    def score(self, hidden, batch, prefix):
        return self.score_features(hidden, batch[f"{prefix}_poi_ids"], batch[f"{prefix}_probs"],
                                   batch[f"{prefix}_main_cat_probs"], batch[f"{prefix}_dists"],
                                   batch[f"{prefix}_other_feats"])

    def forward(self, batch):
        hidden = self.encode(batch)
        scores = self.score(hidden, batch, "cand").masked_fill(~batch["cand_mask"].bool(), -1e9)
        return scores, F.normalize(self.projector(hidden), dim=-1), hidden

    def predict(self, batch):
        hidden = self.encode(batch)
        return self.score(hidden, batch, "cand").masked_fill(~batch["cand_mask"].bool(), -1e9)


class ClassNormalizer:
    """Global POI softmax, evaluated in blocks; optional explicit sampled approximation.

    The denominator includes candidate and NON-candidate logits. A candidate-only
    self-distillation CE is not a faithful substitute for PRODEN/PiCO classification.
    """
    def __init__(self, processor, dataset, mode="full", chunk_size=512):
        self.processor, self.dataset = processor, dataset
        self.mode, self.chunk_size = mode, chunk_size

    def log_partition(self, model, hidden, cand_scores, batch):
        valid = batch["seq_mask"].bool()
        h = hidden[valid]
        candidates = batch["cand_poi_ids"][valid]
        support = batch["cand_mask"][valid].bool()
        logz = cand_scores[valid].float().logsumexp(-1)
        if self.mode == "sampled":
            negatives = model.score(hidden, batch, "neg")[valid].float()
            return torch.logaddexp(logz, negatives.logsumexp(-1))

        device = h.device
        processor = self.processor
        slots = batch["time_slot"][valid] - 1
        coordinates = batch["center_coord"][valid]
        users = (batch["user_id"][valid] - 1).detach().cpu().tolist()
        totals = torch.tensor([self.dataset.user_total_cands[u] for u in users], device=device)
        for start in range(0, processor.num_pois, self.chunk_size):
            end = min(start + self.chunk_size, processor.num_pois)
            raw_ids = np.arange(start, end)
            ids = torch.arange(start + 1, end + 1, device=device).expand(len(h), -1)
            cat_np = processor.poi_cat_indices[raw_ids]
            main_np = processor.main_cat_mapping.numpy()[cat_np]
            slot_np = slots.detach().cpu().numpy()
            probs = torch.as_tensor(processor.cat_time_probs[cat_np[:, None], slot_np].T, device=device)
            main_probs_np = np.zeros((len(h), end - start), dtype=np.float32)
            known = main_np >= 0
            main_probs_np[:, known] = processor.main_cat_probs[main_np[known, None], slot_np].T
            main_probs = torch.as_tensor(main_probs_np, device=device)
            poi_coords = torch.as_tensor(processor.poi_coords[start:end], device=device)
            dists = torch.log1p(torch.linalg.vector_norm(poi_coords[None] - coordinates[:, None], dim=-1) * 6371000.0)
            counts = torch.tensor([[self.dataset.user_cand_freq[u].get(int(v), 0) for v in raw_ids]
                                   for u in users], device=device, dtype=torch.float32)
            recurrence = (torch.log1p(counts) / torch.log1p(totals[:, None]))[..., None]
            excluded = ((ids[..., None] == candidates[:, None]) & support[:, None]).any(-1)

            # Checkpoint both encoder and reduction, preserving dropout RNG during
            # recomputation. Only a per-step vector is retained for each POI block.
            def block(hh, ii, pp, mm, dd, rr, ee):
                logits = model.score_features(hh, ii, pp, mm, dd, rr).float()
                # Some tiny test blocks can be entirely candidates for a row.
                return logits.masked_fill(ee, -1e30).logsumexp(-1)

            if torch.is_grad_enabled():
                part = checkpoint(block, h, ids, probs, main_probs, dists, recurrence, excluded,
                                  use_reentrant=False)
            else:
                part = block(h, ids, probs, main_probs, dists, recurrence, excluded)
            logz = torch.logaddexp(logz, part)
        return logz
