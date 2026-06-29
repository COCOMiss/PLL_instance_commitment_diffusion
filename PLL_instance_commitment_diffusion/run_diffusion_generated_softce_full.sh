#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-tokyo}
GPU=${GPU:-0}
PREFIX=${PREFIX:-diffusion_generated_softce}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
PATIENCE=${PATIENCE:-8}

BASE_EPOCHS=${BASE_EPOCHS:-80}
DIFF_EPOCHS=${DIFF_EPOCHS:-50}
DCE_EPOCHS=${DCE_EPOCHS:-120}

LR_BASE=${LR_BASE:-5e-4}
LR_DIFF=${LR_DIFF:-5e-4}
LR_DCE=${LR_DCE:-5e-4}

STAGE1="result/${DATASET}/${PREFIX}_stage1_base/best_model.pth"
STAGE2="result/${DATASET}/${PREFIX}_stage2_diffusion/best_model.pth"
STAGE3="result/${DATASET}/${PREFIX}_stage3_generated_softce/best_model.pth"

COMMON_DIFF_ARGS=(
  --use_diffusion true
  --diffusion_lambda_max 1.00
  --diffusion_refine_steps 3
  --diffusion_mask_prob 0.10
  --diffusion_temp_min 0.40
  --diffusion_temp_max 0.80
  --diffusion_teacher_temp_min 0.55
  --diffusion_teacher_temp_max 0.85
  --diffusion_reverse_kl_weight 0.20
  --diffusion_input_noise_min 0.05
  --diffusion_input_noise_max 0.45
  --diffusion_ctx_temperature 0.50
  --diffusion_context_mix_max 0.20
  --diffusion_context_loss_weight 0.20
  --diffusion_context_anchor_min_weight 0.10
  --diffusion_entropy_weight 0.02
  --diffusion_margin_weight 0.05
  --diffusion_target_margin 0.05
  --diffusion_use_strict_steps false
  --diffusion_commitment_use_quality_gate true
  --diffusion_stage3_force_mask false
  --diffusion_dce_lambda 1.00
  --diffusion_dce_warmup_epochs 0
)

echo "[Stage 1] PLL-only base -> ${STAGE1}"
CUDA_VISIBLE_DEVICES=${GPU} python PLL_instance_commitment_diffusion/train.py \
  --dataset "${DATASET}" \
  --exp_name "${PREFIX}_stage1_base" \
  --training_phase phase1_base \
  --epochs "${BASE_EPOCHS}" \
  --learning_rate "${LR_BASE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --patience "${PATIENCE}" \
  --use_soft_commitment false \
  --use_diffusion false \
  --save_name "${STAGE1}"

echo "[Stage 2] Train diffusion to generate q_diff from q_IC/q_sharp -> ${STAGE2}"
CUDA_VISIBLE_DEVICES=${GPU} python PLL_instance_commitment_diffusion/train.py \
  --dataset "${DATASET}" \
  --exp_name "${PREFIX}_stage2_diffusion" \
  --training_phase phase2_diffusion \
  --pretrained_checkpoint "${STAGE1}" \
  --epochs "${DIFF_EPOCHS}" \
  --learning_rate "${LR_DIFF}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --patience "${PATIENCE}" \
  "${COMMON_DIFF_ARGS[@]}" \
  --save_name "${STAGE2}"

echo "[Stage 3] Generated-softCE: CE(q_diff, p) weighted by commitment -> ${STAGE3}"
CUDA_VISIBLE_DEVICES=${GPU} python PLL_instance_commitment_diffusion/train.py \
  --dataset "${DATASET}" \
  --exp_name "${PREFIX}_stage3_generated_softce" \
  --training_phase phase3_dce \
  --pretrained_checkpoint "${STAGE2}" \
  --diffusion_checkpoint "${STAGE2}" \
  --epochs "${DCE_EPOCHS}" \
  --learning_rate "${LR_DCE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --patience "${PATIENCE}" \
  "${COMMON_DIFF_ARGS[@]}" \
  --save_name "${STAGE3}"

echo "Done. Final checkpoint: ${STAGE3}"
