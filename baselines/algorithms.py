"""PRODEN and PiCO objectives, adapted to sparse global POI candidate support.

Mechanisms follow Lvcrezia77/PRODEN and hbzju/PiCO. This implementation is
written for the project interfaces; it does not import either project's pipeline.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class ConfidenceBank:
    """CPU O(N*K) storage, indexed by stable step IDs and checked against POI IDs."""
    def __init__(self, num_steps, max_candidates):
        self.values = torch.zeros(num_steps, max_candidates)
        self.support_ids = torch.zeros(num_steps, max_candidates, dtype=torch.int32)
        self.initialized = torch.zeros(num_steps, dtype=torch.bool)

    def get(self, step_ids, poi_ids, mask):
        index = step_ids.detach().cpu().long()
        ids = poi_ids.detach().cpu().to(torch.int32)
        valid = mask.detach().cpu().bool()
        known = self.initialized[index]
        if known.any() and not torch.equal(self.support_ids[index[known]], ids[known]):
            raise ValueError("Candidate POI support changed for an existing training instance.")
        if (~known).any():
            selected = index[~known]
            weights = valid[~known].float()
            self.values[selected] = weights / weights.sum(-1, keepdim=True)
            self.support_ids[selected] = ids[~known]
            self.initialized[selected] = True
        return self.values[index].to(poi_ids.device)

    def update(self, step_ids, values):
        self.values[step_ids.detach().cpu().long()] = values.detach().cpu()

    def state_dict(self):
        return {"values": self.values, "support_ids": self.support_ids, "initialized": self.initialized}

    def load_state_dict(self, state):
        for name in ("values", "support_ids", "initialized"):
            if getattr(self, name).shape != state[name].shape:
                raise ValueError("Confidence bank shape does not match the current data protocol.")
            setattr(self, name, state[name].cpu())


def classification_loss(cand_scores, logz, confidence, mask):
    logp = cand_scores.float() - logz[:, None]
    return -(confidence * logp.masked_fill(~mask.bool(), 0)).sum(-1).mean()


def proden_loss_and_update(cand_scores, logz, batch, bank):
    valid = batch["seq_mask"].bool()
    scores = cand_scores[valid].float()
    ids, mask = batch["cand_poi_ids"][valid], batch["cand_mask"][valid].bool()
    index = batch["step_id"][valid]
    # Use the stored PREVIOUS targets for CE, never p.detach() from this forward.
    confidence = bank.get(index, ids, mask)
    loss = classification_loss(scores, logz, confidence, mask)
    updated = scores.masked_fill(~mask, -1e9).softmax(-1).detach()
    bank.update(index, updated)
    return loss


class PiCO(nn.Module):
    def __init__(self, encoder, num_pois, projection_dim=128, queue_size=8192,
                 encoder_momentum=0.999, prototype_momentum=0.99, temperature=0.07):
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.encoder_k.requires_grad_(False)
        self.encoder_momentum = encoder_momentum
        self.prototype_momentum = prototype_momentum
        self.temperature = temperature
        # 0 is padding; classes are GLOBAL POI IDs, not candidate ranks/categories.
        self.register_buffer("prototypes", torch.zeros(num_pois + 1, projection_dim))
        self.register_buffer("queue", torch.zeros(queue_size, projection_dim))
        self.register_buffer("queue_labels", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        for online, key in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            key.mul_(self.encoder_momentum).add_(online, alpha=1 - self.encoder_momentum)

    @torch.no_grad()
    def enqueue(self, keys, labels):
        size = len(self.queue)
        if len(keys) >= size:
            self.queue.copy_(keys[-size:])
            self.queue_labels.copy_(labels[-size:])
            self.queue_ptr.zero_()
            self.queue_count.fill_(size)
            return
        ptr = int(self.queue_ptr)
        first = min(len(keys), size - ptr)
        self.queue[ptr:ptr + first] = keys[:first]
        self.queue_labels[ptr:ptr + first] = labels[:first]
        if first < len(keys):
            self.queue[:len(keys) - first] = keys[first:]
            self.queue_labels[:len(keys) - first] = labels[first:]
        self.queue_ptr.fill_((ptr + len(keys)) % size)
        self.queue_count.fill_(min(size, int(self.queue_count) + len(keys)))

    def contrastive_loss(self, q, k, labels, supervised=True):
        count = int(self.queue_count)
        queue = self.queue[:count].clone().detach()
        old_labels = self.queue_labels[:count].clone().detach()
        if not supervised:
            positive = (q * k).sum(-1, keepdim=True)
            negative = q @ queue.T
            return F.cross_entropy(torch.cat((positive, negative), -1) / self.temperature,
                                   torch.zeros(len(q), dtype=torch.long, device=q.device))
        features = torch.cat((q, k, queue), 0)
        all_labels = torch.cat((labels, labels, old_labels))
        logits = (q.float() @ features.float().T) / self.temperature
        allowed = torch.ones_like(logits, dtype=torch.bool)
        allowed[torch.arange(len(q)), torch.arange(len(q))] = False
        positives = (labels[:, None] == all_labels[None]) & allowed
        logp = logits - logits.masked_fill(~allowed, -1e9).logsumexp(-1, keepdim=True)
        return -(logp.masked_fill(~positives, 0).sum(-1) / positives.sum(-1).clamp_min(1)).mean()

    def loss(self, batch, bank, normalizer, epoch, epochs, prototype_start=0,
             confidence_start=0.95, confidence_end=0.8, contrastive_weight=0.5):
        scores, projection, hidden = self.encoder_q(batch)
        valid = batch["seq_mask"].bool()
        q = projection[valid]
        ids = batch["cand_poi_ids"][valid]
        mask = batch["cand_mask"][valid].bool()
        index = batch["step_id"][valid]
        confidence = bank.get(index, ids, mask)
        logz = normalizer.log_partition(self.encoder_q, hidden, scores, batch)
        pseudo = ids.gather(1, scores[valid].argmax(-1, keepdim=True)).squeeze(1)
        active = epoch >= prototype_start
        with torch.no_grad():
            # Prototype predictions precede the current batch's prototype update,
            # as in the official implementation. No reliability gate is applied.
            proto_scores = (q.detach()[:, None] * self.prototypes[ids]).sum(-1)
            chosen = proto_scores.masked_fill(~mask, -1e9).argmax(-1)
            if active:
                target = F.one_hot(chosen, ids.size(-1)).float()
                phi = confidence_start + epoch / max(epochs, 1) * (confidence_end - confidence_start)
                confidence = phi * confidence + (1 - phi) * target
                bank.update(index, confidence)
            # Keep the official sequential EMA when several steps share a POI.
            for feature, label in zip(q.detach(), pseudo):
                self.prototypes[label].mul_(self.prototype_momentum).add_(feature, alpha=1 - self.prototype_momentum)
            self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))
            self.momentum_update()
            # Independent dropout views of exactly the same features and support.
            # Train-mode dropout in the key encoder is the non-image augmentation.
            _, key_projection, _ = self.encoder_k(batch)
            k = key_projection[valid]
        cls = classification_loss(scores[valid], logz, confidence, mask)
        contrastive = self.contrastive_loss(q, k, pseudo, supervised=active)
        # Clone queue in loss before in-place enqueue so autograd is unaffected.
        self.enqueue(k, pseudo)
        return cls + contrastive_weight * contrastive, {"classification": cls.detach(),
                                                       "contrastive": contrastive.detach()}

    def predict(self, batch):
        return self.encoder_q.predict(batch)
