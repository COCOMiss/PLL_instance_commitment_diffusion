import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UserTrajectoryModel
from utils import get_logger


logger = get_logger("ModelNoDist")


class CandidatePOIEncoderNoDist(nn.Module):
    def __init__(self, num_pois, pt_path, embed_dim=64, freeze_qwen=False):
        super().__init__()

        self.poi_emb = nn.Embedding(num_pois, embed_dim, padding_idx=0)

        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Cannot find POI embedding file: {pt_path}")

        logger.info(f"Loading 3-layer Qwen embeddings from {pt_path} (Freeze={freeze_qwen})")
        d = torch.load(pt_path)
        qwen_dim = d["dim"]

        self.l1_emb = nn.Embedding.from_pretrained(d["l1"], freeze=freeze_qwen, padding_idx=0)
        self.l2_emb = nn.Embedding.from_pretrained(d["l2"], freeze=freeze_qwen, padding_idx=0)
        self.l3_emb = nn.Embedding.from_pretrained(d["l3"], freeze=freeze_qwen, padding_idx=0)

        self.text_proj = nn.Sequential(
            nn.Linear(qwen_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2 + 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, poi_ids, probs, main_probs, dists, other_feats):
        del dists

        e_poi = self.poi_emb(poi_ids)
        e1 = self.l1_emb(poi_ids)
        e2 = self.l2_emb(poi_ids)
        e3 = self.l3_emb(poi_ids)

        concat_text = torch.cat([e1, e2, e3], dim=-1)
        e_text = self.text_proj(concat_text)

        probs_exp = probs.unsqueeze(-1)
        main_probs_exp = main_probs.unsqueeze(-1)
        recurrence_exp = other_feats[..., 0].unsqueeze(-1)

        concat_features = torch.cat(
            [
                e_poi,
                e_text,
                probs_exp,
                main_probs_exp,
                recurrence_exp,
            ],
            dim=-1,
        )

        return self.fusion_layer(concat_features)


class TrajPOITransformerNoDist(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim

        self.user_model = UserTrajectoryModel(
            num_users=len(config.user2idx) + 1,
            num_time_slots=24 + 1,
            embed_dim=embed_dim,
        )

        pt_path = os.path.join("dataset", config.dataset_name, "qwen_poi_embeddings.pt")
        self.candidate_model = CandidatePOIEncoderNoDist(
            num_pois=len(config.venue_id2idx) + 1,
            pt_path=pt_path,
            embed_dim=embed_dim,
            freeze_qwen=False,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def _compute_cosine_scores(self, user_vector, poi_vectors):
        user_unit = F.normalize(user_vector, p=2, dim=-1, eps=1e-8)
        poi_unit = F.normalize(poi_vectors, p=2, dim=-1, eps=1e-8)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (user_unit.unsqueeze(2) * poi_unit).sum(dim=-1)

    def forward(self, batch):
        padding_mask = ~batch["seq_mask"].bool()

        user_vector = self.user_model(
            batch["user_id"],
            batch["time_slot"],
            batch["center_coord"],
            src_key_padding_mask=padding_mask,
        )

        cand_vectors = self.candidate_model(
            poi_ids=batch["cand_poi_ids"],
            probs=batch["cand_probs"],
            main_probs=batch["cand_main_cat_probs"],
            dists=batch["cand_dists"],
            other_feats=batch["cand_other_feats"],
        )
        cand_scores = self._compute_cosine_scores(user_vector, cand_vectors)

        neg_vectors = self.candidate_model(
            poi_ids=batch["neg_poi_ids"],
            probs=batch["neg_probs"],
            main_probs=batch["neg_main_cat_probs"],
            dists=batch["neg_dists"],
            other_feats=batch["neg_other_feats"],
        )
        neg_scores = self._compute_cosine_scores(user_vector, neg_vectors)

        return cand_scores, neg_scores, user_vector

    def compute_weighted_clpl_loss_per_step(self, cand_scores, cand_mask, neg_scores):
        masked_cand_scores = cand_scores.masked_fill(cand_mask == 0, -1e9)
        cand_weights = F.softmax(masked_cand_scores, dim=-1).detach()
        cand_weighted_score = (cand_scores * cand_weights * cand_mask).sum(dim=-1)
        loss_pos = F.softplus(-cand_weighted_score)

        neg_weights = F.softmax(neg_scores, dim=-1).detach()
        neg_weighted_score = (neg_scores * neg_weights).sum(dim=-1)
        loss_neg = F.softplus(neg_weighted_score)

        return loss_pos + loss_neg

    def compute_weighted_clpl_loss(self, cand_scores, cand_mask, neg_scores, seq_mask):
        step_loss = self.compute_weighted_clpl_loss_per_step(
            cand_scores=cand_scores,
            cand_mask=cand_mask,
            neg_scores=neg_scores,
        )
        valid_loss = (step_loss * seq_mask.float()).sum()
        num_valid_steps = seq_mask.float().sum().clamp(min=1.0)
        return valid_loss / num_valid_steps

    def predict(self, batch):
        padding_mask = ~batch["seq_mask"].bool()

        user_vector = self.user_model(
            batch["user_id"],
            batch["time_slot"],
            batch["center_coord"],
            src_key_padding_mask=padding_mask,
        )

        cand_vectors = self.candidate_model(
            poi_ids=batch["cand_poi_ids"],
            probs=batch["cand_probs"],
            main_probs=batch["cand_main_cat_probs"],
            dists=batch["cand_dists"],
            other_feats=batch["cand_other_feats"],
        )

        emission_scores = self._compute_cosine_scores(user_vector, cand_vectors)
        return emission_scores.masked_fill(batch["cand_mask"] == 0, -1e9)
