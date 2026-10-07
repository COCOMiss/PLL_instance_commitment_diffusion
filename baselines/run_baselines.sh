#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
read -r -a datasets <<< "${DATASETS:-tokyo ny_filterPOI gowalla_filtered}"
read -r -a methods <<< "${METHODS:-proden pico}"
read -r -a seeds <<< "${SEEDS:-42 43 44 45 46}"
for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPU:-0}" "${PYTHON:-python}" -m baselines.train \
        --dataset "$dataset" --method "$method" --seed "$seed" --data_seed "${DATA_SEED:-42}" \
        --epochs "${EPOCHS:-100}" --batch_size "${BATCH_SIZE:-4}" \
        --num_workers "${NUM_WORKERS:-4}" --normalizer "${NORMALIZER:-full}" \
        --class_chunk_size "${CLASS_CHUNK_SIZE:-512}" --num_negatives "${NUM_NEGATIVES:-256}" \
        --prototype_start "${PROTOTYPE_START:-1}" "$@"
    done
  done
done
"${PYTHON:-python}" -m baselines.summarize
