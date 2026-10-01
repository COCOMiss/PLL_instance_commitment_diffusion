#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# Basic configuration
# ============================================================

DATASET=${DATASET:-ny_filterPOI}
GPU=${GPU:-3}
PREFIX=${PREFIX:-diffusion_generated_softce03}

RUN_MODE=${RUN_MODE:-full}   # full | ic | all

BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
PATIENCE=${PATIENCE:-8}


# ============================================================
# Epochs
# ============================================================

DIFF_EPOCHS=${DIFF_EPOCHS:-50}
DCE_EPOCHS=${DCE_EPOCHS:-120}
IC_EPOCHS=${IC_EPOCHS:-20}


# ============================================================
# Learning rates
# ============================================================

LR_DIFF=${LR_DIFF:-5e-4}
LR_DCE=${LR_DCE:-5e-4}
LR_IC=${LR_IC:-5e-4}



# ============================================================
# Quality gate
# ============================================================

QUALITY_GATE_MODE=${QUALITY_GATE_MODE:-relaxed}

GATE_ENTROPY=${GATE_ENTROPY:-0.10}
GATE_MARGIN=${GATE_MARGIN:-0.80}



# ============================================================
# Checkpoints
# ============================================================

# Stage1 已经训练好的 checkpoint
STAGE1="result/ny_filterPOI/instance_commitment_diffusion/diffusion_generated_softce03_stage1_base/best_model.pth"


# Stage2 输出
STAGE2="result/${DATASET}/instance_commitment_diffusion/${PREFIX}_stage2_diffusion/best_model.pth"


# Stage3 输出
STAGE3="result/${DATASET}/instance_commitment_diffusion/${PREFIX}_stage3_joint_dce/best_model.pth"



# ============================================================
# Common args
# ============================================================

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



# ============================================================
# Check stage1
# ============================================================

if [[ ! -f "${STAGE1}" ]]; then
    echo "[ERROR] Stage1 checkpoint not found:"
    echo "${STAGE1}"
    exit 1
fi


echo "[INFO] Using Stage1 checkpoint:"
echo "${STAGE1}"




# ============================================================
# Stage2 + Stage3
# ============================================================

if [[ "${RUN_MODE}" == "full" || "${RUN_MODE}" == "all" ]]; then


# ------------------------------------------------------------
# Stage2
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[Stage2] Diffusion denoiser training"
echo "Input checkpoint:"
echo "${STAGE1}"
echo "Output:"
echo "${STAGE2}"
echo "============================================================"


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



# ------------------------------------------------------------
# Stage3
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[Stage3] Joint PLL + generated soft CE + diffusion"
echo "Input checkpoint:"
echo "${STAGE2}"
echo "Output:"
echo "${STAGE3}"
echo "============================================================"


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



echo ""
echo "============================================================"
echo "Finished."
echo "RUN_MODE=${RUN_MODE}"
echo "Stage1 checkpoint:"
echo "${STAGE1}"
echo "============================================================"