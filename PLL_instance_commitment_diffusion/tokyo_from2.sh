#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-tokyo}
GPU=${GPU:-3}
PREFIX=${PREFIX:-diffusion_generated_softce}
RUN_MODE=${RUN_MODE:-full}   # full | ic | all
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
PATIENCE=${PATIENCE:-8}

DIFF_EPOCHS=${DIFF_EPOCHS:-50}
DCE_EPOCHS=${DCE_EPOCHS:-120}
IC_EPOCHS=${IC_EPOCHS:-120}

LR_DIFF=${LR_DIFF:-5e-4}
LR_DCE=${LR_DCE:-5e-4}
LR_IC=${LR_IC:-5e-4}

QUALITY_GATE_MODE=${QUALITY_GATE_MODE:-relaxed}   # relaxed | strict | none
GATE_ENTROPY=${GATE_ENTROPY:-0.10}
GATE_MARGIN=${GATE_MARGIN:-0.80}

STAGE1=result/${DATASET}/instance_commitment_diffusion/context_poe_stage1_base/best_model.pth
STAGE_IC="result/${DATASET}/instance_commitment_diffusion/${PREFIX}_ic_only/best_model.pth"
STAGE2="result/${DATASET}/instance_commitment_diffusion/${PREFIX}_stage2_diffusion/best_model.pth"
STAGE3="result/${DATASET}/instance_commitment_diffusion/${PREFIX}_stage3_joint_dce/best_model.pth"

COMMON_TRAIN_ARGS=(
  --dataset "${DATASET}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --patience "${PATIENCE}"
)

COMMON_DIFF_ARGS=(
  --use_diffusion true
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
  --diffusion_commitment_use_quality_gate true
  --diffusion_quality_gate_mode "${QUALITY_GATE_MODE}"
  --diffusion_gate_entropy_threshold "${GATE_ENTROPY}"
  --diffusion_gate_margin_threshold "${GATE_MARGIN}"
  --diffusion_dce_lambda 1.00
)

if [[ "${RUN_MODE}" == "ic" || "${RUN_MODE}" == "all" ]]; then
  echo "[Ablation] InstanceCommitment only: CE(q_IC, p), no sharpening, no diffusion -> ${STAGE_IC}"
  CUDA_VISIBLE_DEVICES=${GPU} python PLL_instance_commitment_diffusion/train.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    --exp_name "${PREFIX}_ic_only" \
    --training_phase phase3_ic_dce \
    --pretrained_checkpoint "${STAGE1}" \
    --epochs "${IC_EPOCHS}" \
    --learning_rate "${LR_IC}" \
    --ic_dce_lambda 1.00 \
    --save_name "${STAGE_IC}"
fi

if [[ "${RUN_MODE}" == "full" || "${RUN_MODE}" == "all" ]]; then
  echo "[Stage 2] Train diffusion to generate q_diff from q_IC/q_sharp -> ${STAGE2}"
  CUDA_VISIBLE_DEVICES=${GPU} python PLL_instance_commitment_diffusion/train.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    --exp_name "${PREFIX}_stage2_diffusion" \
    --training_phase phase2_diffusion \
    --pretrained_checkpoint "${STAGE1}" \
    --epochs "${DIFF_EPOCHS}" \
    --learning_rate "${LR_DIFF}" \
    "${COMMON_DIFF_ARGS[@]}" \
    --diffusion_lambda_max 1.00 \
    --save_name "${STAGE2}"

  echo "[Stage 3] Joint generated-softCE with relaxed gate -> ${STAGE3}"
  CUDA_VISIBLE_DEVICES=${GPU} python PLL_instance_commitment_diffusion/train.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    --exp_name "${PREFIX}_stage3_joint_dce" \
    --training_phase phase3_joint_dce \
    --pretrained_checkpoint "${STAGE2}" \
    --diffusion_checkpoint "${STAGE2}" \
    --epochs "${DCE_EPOCHS}" \
    --learning_rate "${LR_DCE}" \
    "${COMMON_DIFF_ARGS[@]}" \
    --diffusion_lambda_max 0.02 \
    --save_name "${STAGE3}"
fi

echo "Done. RUN_MODE=${RUN_MODE}"
