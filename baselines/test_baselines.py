"""Meaningful CPU checks; synthetic labels/results are never research results."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import build_data, seq_collate_fn, pool_features
from .model import POIAssignmentEncoder, ClassNormalizer
from .algorithms import ConfidenceBank, PiCO, classification_loss, proden_loss_and_update
from .train import training_inputs, evaluate, build_optimizer, load_optimizer_state


def synthetic_data(folder):
    """Input CSV schema and dummy semantic embeddings, for executable tests only."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = 32
    cats = ["Food", "Shop", "Park"]
    pd.DataFrame({"venue_id": [f"p{i}" for i in range(n)],
                  "latitude": 35.0 + np.arange(n) * 0.0001,
                  "longitude": 139.0 + np.arange(n) * 0.00005,
                  "category": [cats[i % 3] for i in range(n)]}).to_csv(folder / "poi.csv", index=False)
    rows = []
    for user in range(4):
        for event in range(20):
            rows.append({"userid": f"u{user}", "venue_id": f"p{(event + user * 3) % n}",
                         "local_datetime": f"2024-01-{1 + event // 5:02d} {8 + event % 5:02d}:00:00"})
    pd.DataFrame(rows).to_csv(folder / "filtered_checkin_data.csv", index=False)
    pd.DataFrame({"original": cats, "main": ["Leisure", "Retail", "Leisure"]}).to_csv(folder / "main_category.csv", index=False)
    (folder / "category_list.txt").write_text("\n".join(cats) + "\n")
    frame = pd.DataFrame({"category": cats})
    for slot in range(24):
        frame[f"t{slot}"] = np.roll([0.1, 0.3, 0.6], slot % 3)
    frame.to_csv(folder / "category_time_distribution_P_Category_given_Time.csv", index=False)
    generator = torch.Generator().manual_seed(9)
    semantic = {"dim": 4}
    for name in ("l1", "l2", "l3"):
        semantic[name] = torch.randn(n + 1, 4, generator=generator)
        semantic[name][0].zero_()
    torch.save(semantic, folder / "qwen_poi_embeddings.pt")


class BaselineTests(unittest.TestCase):
    def test_low_memory_adamw_and_legacy_resume(self):
        model = torch.nn.Linear(3, 2).double()
        reference = copy.deepcopy(model)
        args = SimpleNamespace(learning_rate=0.001, weight_decay=0.01)
        optimizer = build_optimizer(model, args)
        other = torch.optim.AdamW(reference.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, foreach=True)
        x = torch.tensor([[0.2, -0.4, 0.8]], dtype=torch.float64)
        for step in range(3):
            if step == 1:
                legacy = copy.deepcopy(optimizer.state_dict())
                legacy["param_groups"][0]["foreach"] = None
                load_optimizer_state(optimizer, legacy)
            self.assertIs(optimizer.param_groups[0]["foreach"], False)
            for module, opt in ((model, optimizer), (reference, other)):
                opt.zero_grad(set_to_none=True)
                module(x).square().sum().backward()
                opt.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name) / "tiny"
        synthetic_data(cls.folder)
        cls.args = SimpleNamespace(data_root=cls.temp.name, dataset="tiny", radius=200,
                                   max_candidates=3, noisy_value=50, max_seq_len=4,
                                   data_seed=42, num_negatives=4)
        cls.processor, cls.datasets = build_data(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        torch.manual_seed(1)
        self.model = POIAssignmentEncoder(self.processor, self.folder / "qwen_poi_embeddings.pt",
                                          embed_dim=8, projection_dim=6, max_seq_len=4)
        self.batch = seq_collate_fn([self.datasets[0][0], self.datasets[0][1]])

    def test_fixed_support_and_feature_reuse(self):
        dataset = self.datasets[0]
        first = dataset[0]
        dataset.epoch = 3
        second = dataset[0]
        for key in ("cand_poi_ids", "cand_probs", "cand_main_cat_probs", "cand_dists",
                    "cand_other_feats", "center_coord", "step_id"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        ids = first["cand_poi_ids"].numpy() - 1
        features = pool_features(self.processor, dataset, first, ids, "cand")
        for key, expected in features.items():
            if key in first:
                self.assertTrue(torch.allclose(first[key], expected), key)
        # Stable support across worker processes as well as local reads.
        if os.environ.get("PLL_TEST_WORKERS") == "1":
            loader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=seq_collate_fn)
            actual = next(iter(loader))
            self.assertTrue(torch.equal(first["cand_poi_ids"], actual["cand_poi_ids"][0]))

    def test_full_normalizer_matches_dense_loss_and_gradients(self):
        self.model.eval()  # deterministic features for exact comparison
        inputs = training_inputs(self.batch)
        scores, _, hidden = self.model(inputs)
        normalizer = ClassNormalizer(self.processor, self.datasets[0], "full", chunk_size=7)
        logz = normalizer.log_partition(self.model, hidden, scores, inputs)
        valid = inputs["seq_mask"].bool()
        # Independently construct the full POI feature pool using the CPU helper.
        all_scores = []
        for item in (self.datasets[0][0], self.datasets[0][1]):
            pool = np.broadcast_to(np.arange(self.processor.num_pois), (len(item["seq_mask"]), self.processor.num_pois))
            all_batch = {**item, **pool_features(self.processor, self.datasets[0], item, pool, "neg")}
            all_batch = seq_collate_fn([all_batch])
            h = self.model.encode(all_batch)
            dense = self.model.score(h, all_batch, "neg")[0]
            cand = self.model.score(h, all_batch, "cand")[0]
            for t in range(len(dense)):
                support = item["cand_mask"][t].bool()
                dense[t, item["cand_poi_ids"][t, support] - 1] = cand[t, support]
            all_scores.append(dense)
        dense_logz = torch.cat(all_scores).logsumexp(-1)
        self.assertTrue(torch.allclose(logz, dense_logz, atol=1e-5))
        mask = inputs["cand_mask"][valid].bool()
        confidence = mask.float() / mask.sum(-1, keepdim=True)
        blocked = classification_loss(scores[valid], logz, confidence, mask)
        dense = classification_loss(scores[valid], dense_logz, confidence, mask)
        selected = [self.model.logit_scale, self.model.candidate_model.fusion_layer[-1].weight]
        a = torch.autograd.grad(blocked, selected, retain_graph=True)
        b = torch.autograd.grad(dense, selected)
        for left, right in zip(a, b):
            self.assertTrue(torch.allclose(left, right, atol=2e-5, rtol=2e-4))

    def test_proden_previous_targets_and_nonzero_gradient(self):
        bank = ConfidenceBank(self.datasets[0].num_steps, 3)
        inputs = training_inputs(self.batch)
        scores, _, hidden = self.model(inputs)
        logz = ClassNormalizer(self.processor, self.datasets[0], "sampled").log_partition(self.model, hidden, scores, inputs)
        valid = inputs["seq_mask"].bool()
        mask = inputs["cand_mask"][valid].bool()
        uniform = mask.float() / mask.sum(-1, keepdim=True)
        expected = classification_loss(scores[valid], logz, uniform, mask)
        loss = proden_loss_and_update(scores, logz, inputs, bank)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertGreater(float(self.model.candidate_model.fusion_layer[-1].weight.grad.norm()), 0)
        revised = bank.values[inputs["step_id"][valid]]
        self.assertTrue(torch.allclose(revised.sum(-1), torch.ones(len(revised))))
        self.assertTrue(torch.allclose(revised, scores[valid].detach().softmax(-1)))
        altered = inputs["cand_poi_ids"][valid].clone()
        altered[0, 0] += 1
        with self.assertRaises(ValueError):
            bank.get(inputs["step_id"][valid], altered, mask)

    def test_pico_core_training_and_ring_queue(self):
        pico = PiCO(self.model, self.processor.num_pois, 6, queue_size=5)
        bank = ConfidenceBank(self.datasets[0].num_steps, 3)
        optimizer = torch.optim.AdamW(pico.encoder_q.parameters(), lr=1e-3)
        normalizer = ClassNormalizer(self.processor, self.datasets[0], "sampled")
        inputs = training_inputs(self.batch)
        for epoch in range(2):
            pico.train()
            optimizer.zero_grad()
            loss, parts = pico.loss(inputs, bank, normalizer, epoch, 2, prototype_start=1)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            # An initially empty queue has no MoCo negatives in the first batch.
            if epoch > 0:
                self.assertGreater(float(pico.encoder_q.projector[-1].weight.grad.norm()), 0)
            optimizer.step()
        self.assertEqual(int(pico.queue_count), 5)
        self.assertTrue((pico.queue_labels > 0).all())
        self.assertTrue((pico.prototypes.norm(dim=-1) > 0).any())
        self.assertTrue(all(p.grad is None for p in pico.encoder_k.parameters()))
        state = copy.deepcopy(pico.state_dict())
        clone = PiCO(copy.deepcopy(self.model), self.processor.num_pois, 6, 5)
        clone.load_state_dict(state)
        pico.eval()
        clone.eval()
        self.assertTrue(torch.allclose(pico.predict(inputs), clone.predict(inputs)))

    def test_training_inputs_remove_ground_truth_and_evaluate(self):
        inputs = training_inputs(self.batch)
        self.assertNotIn("true_poi_id", inputs)
        self.assertNotIn("label_pos", inputs)
        loader = DataLoader(self.datasets[1], batch_size=2, collate_fn=seq_collate_fn)
        metrics = evaluate(self.model, loader, torch.device("cpu"))
        self.assertEqual(metrics["num_steps"], self.datasets[1].num_steps)
        self.assertTrue(0 <= metrics["acc1"] <= metrics["acc5"] <= 1)


if __name__ == "__main__":
    unittest.main()
