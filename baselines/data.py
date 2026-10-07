"""Reuse the project protocol, but give each PLL training instance a stable identity.

No exact training POI is used by the losses. The inherited check-in proxy protocol
does use it to construct a true-label-containing candidate set, as the project does.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PLL_instance_commitment_diffusion"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from dataset import CheckinSequenceDataset, POIDataProcessor, POIProcessingConfig, seq_collate_fn


@contextmanager
def numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


class StableDataset(CheckinSequenceDataset):
    def __init__(self, processor, max_seq_len, mode, data_seed, num_negatives):
        # Three fixed streams reproduce the original sequential dataset builds.
        self.data_seed = data_seed
        self.num_negatives = num_negatives
        self.epoch = 0
        super().__init__(processor, max_seq_len=max_seq_len, mode=mode)
        self.offsets = np.concatenate(([0], np.cumsum([len(x) for x in self.traj_groups])))
        self.num_steps = int(self.offsets[-1])

    def _get_prev_small_cat_id(self, row_indices, traj_df, true_poi_indices, t):
        # No transition feature based on the previous ground-truth POI/category.
        return -1

    def __getitem__(self, idx):
        salt = {"train": 0, "val": 1000003, "test": 2000003}[self.mode]
        with numpy_seed(self.data_seed + salt + idx * 7919):
            batch = super().__getitem__(idx)
        # The support is fixed across epochs, workers, methods and model seeds.
        batch["step_id"] = torch.arange(self.offsets[idx], self.offsets[idx + 1])
        if self.mode == "train":
            # Plain uniform non-candidate sampling, not the project's focus schedule.
            rng = np.random.default_rng(self.data_seed + idx * 7919 + self.epoch * 104729)
            pools = []
            for ids, mask in zip(batch["cand_poi_ids"], batch["cand_mask"]):
                excluded = set((ids[mask.bool()].numpy() - 1).tolist())
                available = self.processor.num_pois - len(excluded)
                count = self.num_negatives
                if not 1 <= count <= available:
                    raise ValueError(f"Requested {count} unique negatives but only {available} non-candidate POIs exist; reduce --num_negatives.")
                chosen = set()
                while len(chosen) < count:
                    for value in rng.integers(0, self.processor.num_pois, size=max(16, count * 2)):
                        value = int(value)
                        if value not in excluded:
                            chosen.add(value)
                        if len(chosen) == count:
                            break
                values = np.asarray(sorted(chosen), dtype=np.int64)
                pools.append(values)
            pool = np.stack(pools)
            batch.update(pool_features(self.processor, self, batch, pool, "neg"))
        return batch


def pool_features(processor, dataset, batch, zero_ids, prefix):
    """Exactly the project's candidate/negative scalar feature definitions (CPU)."""
    cats = processor.poi_cat_indices[zero_ids]
    mapping = processor.main_cat_mapping.numpy()
    mains = mapping[cats]
    slots = batch["time_slot"].numpy() - 1
    probs = processor.cat_time_probs[cats, slots[:, None]]
    main_probs = np.zeros_like(probs)
    valid = mains >= 0
    main_probs[valid] = processor.main_cat_probs[mains[valid], np.broadcast_to(slots[:, None], mains.shape)[valid]]
    coords = batch["center_coord"].numpy()
    dists = np.log1p(np.linalg.norm(processor.poi_coords[zero_ids] - coords[:, None], axis=-1) * 6371000.0)
    uid = int(batch["user_id"][0]) - 1
    counter = dataset.user_cand_freq[uid]
    counts = np.asarray([[counter.get(int(v), 0) for v in row] for row in zero_ids], dtype=np.float32)
    if prefix == "cand":
        counts = np.maximum(0, counts - 1)
    recur = np.log1p(counts) / np.log1p(dataset.user_total_cands[uid])
    return {
        f"{prefix}_poi_ids": torch.as_tensor(zero_ids + 1, dtype=torch.long),
        f"{prefix}_cat_ids": torch.as_tensor(cats + 1, dtype=torch.long),
        f"{prefix}_main_cat_ids": torch.as_tensor(mains + 1, dtype=torch.long),
        f"{prefix}_probs": torch.as_tensor(probs, dtype=torch.float32),
        f"{prefix}_main_cat_probs": torch.as_tensor(main_probs, dtype=torch.float32),
        f"{prefix}_dists": torch.as_tensor(dists, dtype=torch.float32),
        f"{prefix}_other_feats": torch.as_tensor(recur[..., None], dtype=torch.float32),
    }


def build_data(args):
    folder = Path(args.data_root).resolve() / args.dataset
    config = POIProcessingConfig(
        str(folder / "filtered_checkin_data.csv"), str(folder / "poi.csv"),
        str(folder / "category_time_distribution_P_Category_given_Time.csv"),
        radius=args.radius, max_candidates=args.max_candidates, noisy_value=args.noisy_value,
    )
    processor = POIDataProcessor(config)
    with numpy_seed(args.data_seed):
        datasets = [StableDataset(processor, args.max_seq_len, mode, args.data_seed,
                                  args.num_negatives) for mode in ("train", "val", "test")]
    if any(len(d) == 0 for d in datasets):
        raise ValueError("A train/validation/test split is empty after the project filtering.")
    return processor, datasets


def protocol_manifest(args):
    folder = Path(args.data_root).resolve() / args.dataset
    hashes = {}
    for name in ("filtered_checkin_data.csv", "poi.csv", "category_list.txt", "main_category.csv",
                 "category_time_distribution_P_Category_given_Time.csv", "qwen_poi_embeddings.pt"):
        path = folder / name
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
    return {"data_seed": args.data_seed, "radius": args.radius, "max_candidates": args.max_candidates,
            "max_seq_len": args.max_seq_len, "noisy_value": args.noisy_value,
            "protocol": "project_uniform_disk_random_GT_preserving_truncation_fixed_per_instance",
            "source_files_sha256": hashes,
            "prior_provenance": "supplied CSV; must be train-only or independently external"}
