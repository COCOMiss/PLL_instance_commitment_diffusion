#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-tokyo}
GPU=${GPU:-0}
PREFIX=${PREFIX:-diffusion_generated_softce}
STAGE2=${STAGE2:-result/${DATASET}/${PREFIX}_stage2_diffusion/best_model.pth}
STAGE3=${STAGE3:-result/${DATASET}/${PREFIX}_stage3_generated_softce/best_model.pth}
EPOCHS=${EPOCHS:-120}
LR=${LR:-5e-4}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
PATIENCE=${PATIENCE:-8}

CUDA_VISIBLE_DEVICES=${GPU} python PLL_instance_commitment_diffusion/train.py \
  --dataset "${DATASET}" \
  --exp_name "${PREFIX}_stage3_generated_softce" \
  --training_phase phase3_dce \
  --pretrained_checkpoint "${STAGE2}" \
  --diffusion_checkpoint "${STAGE2}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --patience "${PATIENCE}" \
  --use_diffusion true \
  --diffusion_dce_lambda 1.00 \
  --diffusion_dce_warmup_epochs 0 \
  --diffusion_commitment_use_quality_gate true \
  --diffusion_stage3_force_mask false \
  --diffusion_use_strict_steps false \
  --diffusion_refine_steps 3 \
  --diffusion_mask_prob 0.10 \
  --diffusion_temp_min 0.40 \
  --diffusion_temp_max 0.80 \
  --diffusion_teacher_temp_min 0.55 \
  --diffusion_teacher_temp_max 0.85 \
  --diffusion_reverse_kl_weight 0.20 \
  --diffusion_input_noise_min 0.05 \
  --diffusion_input_noise_max 0.45 \
  --diffusion_context_mix_max 0.20 \
  --diffusion_context_loss_weight 0.20 \
  --diffusion_entropy_weight 0.02 \
  --diffusion_margin_weight 0.05 \
  --save_name "${STAGE3}"
