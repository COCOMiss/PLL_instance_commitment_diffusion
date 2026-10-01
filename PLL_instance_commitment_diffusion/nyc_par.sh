#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# TADRef hyperparameter sensitivity sweep.
#
# This script never trains Stage I. Every Stage-II run is initialized from the
# same Stage-I checkpoint, and every Stage-III run is initialized from its own
# Stage-II checkpoint.
#
# Usage examples:
#   SWEEP=all   GPU=0 bash nyc.sh
#   SWEEP=rho   GPU=0 bash nyc.sh
#   SWEEP=mask  GPU=0 bash nyc.sh
#   SWEEP=steps GPU=0 bash nyc.sh
#
# Set FORCE_RERUN=true to ignore completed logs and rerun a configuration.
# -----------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
TRAIN_PY=${TRAIN_PY:-"${SCRIPT_DIR}/train.py"}

DATASET=${DATASET:-ny_filterPOI}
GPU=${GPU:-2}
SWEEP=${SWEEP:-all}                     # all | rho | mask | steps
SENS_PREFIX=${SENS_PREFIX:-tadref_sensitivity}
FORCE_RERUN=${FORCE_RERUN:-false}

BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
PATIENCE=${PATIENCE:-8}
DIFF_EPOCHS=${DIFF_EPOCHS:-50}
DCE_EPOCHS=${DCE_EPOCHS:-120}
LR_DIFF=${LR_DIFF:-5e-4}
LR_DCE=${LR_DCE:-5e-4}

# Fixed values used when another hyperparameter is varied.
DEFAULT_RHO_MAX=${DEFAULT_RHO_MAX:-0.30}
DEFAULT_MASK_PROB=${DEFAULT_MASK_PROB:-0.10}
DEFAULT_REFINE_STEPS=${DEFAULT_REFINE_STEPS:-3}
RHO_MIN=${RHO_MIN:-0.05}

# Stage-I checkpoint. Override on the command line when your path differs:
#   STAGE1=/absolute/path/to/best_model.pth bash nyc.sh
STAGE1=${STAGE1:-${PROJECT_ROOT}/result/${DATASET}/instance_commitment_diffusion/diffusion_generated_softce03_stage1_base/best_model.pth}

QUALITY_GATE_MODE=${QUALITY_GATE_MODE:-relaxed}
GATE_ENTROPY=${GATE_ENTROPY:-0.10}
GATE_MARGIN=${GATE_MARGIN:-0.80}

RESULT_BASE=${RESULT_BASE:-"${PROJECT_ROOT}/result/${DATASET}/${SENS_PREFIX}"}
mkdir -p "${RESULT_BASE}"

if [[ ! -f "${STAGE1}" ]]; then
  echo "[ERROR] Stage-I checkpoint not found: ${STAGE1}" >&2
  echo "Set STAGE1=/path/to/stage1/best_model.pth and rerun." >&2
  exit 1
fi

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] train.py not found: ${TRAIN_PY}" >&2
  exit 1
fi

COMMON_TRAIN_ARGS=(
  --dataset "${DATASET}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --patience "${PATIENCE}"
)

is_completed() {
  local log_file=$1
  [[ -f "${log_file}" ]] && grep -q "Reloaded Best" "${log_file}"
}

run_configuration() {
  local run_name=$1
  local rho_max=$2
  local mask_prob=$3
  local refine_steps=$4

  local run_root="${RESULT_BASE}/${run_name}"
  local stage2_dir="${run_root}/stage2_diffusion"
  local stage3_dir="${run_root}/stage3_joint_dce"
  local stage2_ckpt="${stage2_dir}/best_model.pth"
  local stage3_ckpt="${stage3_dir}/best_model.pth"
  local stage2_log="${stage2_dir}/training.log"
  local stage3_log="${stage3_dir}/training.log"

  mkdir -p "${stage2_dir}" "${stage3_dir}"
  printf "parameter\tvalue\n" > "${run_root}/config.tsv"
  printf "rho_min\t%s\n" "${RHO_MIN}" >> "${run_root}/config.tsv"
  printf "rho_max\t%s\n" "${rho_max}" >> "${run_root}/config.tsv"
  printf "mask_prob\t%s\n" "${mask_prob}" >> "${run_root}/config.tsv"
  printf "refine_steps\t%s\n" "${refine_steps}" >> "${run_root}/config.tsv"
  printf "stage1_checkpoint\t%s\n" "${STAGE1}" >> "${run_root}/config.tsv"

  echo
  echo "===================================================================="
  echo "[Sensitivity] ${DATASET} | ${run_name}"
  echo "rho=[${RHO_MIN},${rho_max}] | mask=${mask_prob} | steps=${refine_steps}"
  echo "===================================================================="

  if [[ "${FORCE_RERUN}" != "true" ]] && is_completed "${stage3_log}"; then
    echo "[Skip] Completed Stage III found: ${stage3_log}"
    return
  fi

  local diff_args=(
    --use_diffusion true
    --diffusion_refine_steps "${refine_steps}"
    --diffusion_mask_prob "${mask_prob}"
    --diffusion_temp_min 0.40
    --diffusion_temp_max 0.80
    --diffusion_teacher_temp_min 0.55
    --diffusion_teacher_temp_max 0.85
    --diffusion_reverse_kl_weight 0.20
    --diffusion_input_noise_min "${RHO_MIN}"
    --diffusion_input_noise_max "${rho_max}"
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

  if [[ "${FORCE_RERUN}" == "true" ]] || ! is_completed "${stage2_log}"; then
    echo "[Stage II] Pretrain diffusion from the shared Stage-I checkpoint"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python "${TRAIN_PY}" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --exp_name "${SENS_PREFIX}/${run_name}/stage2_diffusion" \
      --training_phase phase2_diffusion \
      --pretrained_checkpoint "${STAGE1}" \
      --epochs "${DIFF_EPOCHS}" \
      --learning_rate "${LR_DIFF}" \
      "${diff_args[@]}" \
      --diffusion_lambda_max 1.00 \
      --log_file "${stage2_log}" \
      --save_name "${stage2_ckpt}"
  else
    echo "[Skip] Completed Stage II found: ${stage2_log}"
  fi

  if [[ ! -f "${stage2_ckpt}" ]]; then
    echo "[ERROR] Missing Stage-II checkpoint: ${stage2_ckpt}" >&2
    exit 1
  fi

  echo "[Stage III] Jointly train PLL + DCE + diffusion loss"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python "${TRAIN_PY}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --exp_name "${SENS_PREFIX}/${run_name}/stage3_joint_dce" \
    --training_phase phase3_joint_dce \
    --pretrained_checkpoint "${stage2_ckpt}" \
    --diffusion_checkpoint "${stage2_ckpt}" \
    --epochs "${DCE_EPOCHS}" \
    --learning_rate "${LR_DCE}" \
    "${diff_args[@]}" \
    --diffusion_lambda_max 0.02 \
    --log_file "${stage3_log}" \
    --save_name "${stage3_ckpt}"
}

run_rho_sweep() {
  for value in 0.10 0.20 0.30 0.40 0.50; do
    if [[ "${value}" == "${DEFAULT_RHO_MAX}" ]]; then
      run_configuration default "${value}" "${DEFAULT_MASK_PROB}" "${DEFAULT_REFINE_STEPS}"
    else
      run_configuration "rho_max_${value//./p}" "${value}" "${DEFAULT_MASK_PROB}" "${DEFAULT_REFINE_STEPS}"
    fi
  done
}

run_mask_sweep() {
  for value in 0 0.05 0.10 0.20 0.30; do
    if [[ "${value}" == "${DEFAULT_MASK_PROB}" ]]; then
      run_configuration default "${DEFAULT_RHO_MAX}" "${value}" "${DEFAULT_REFINE_STEPS}"
    else
      run_configuration "mask_${value//./p}" "${DEFAULT_RHO_MAX}" "${value}" "${DEFAULT_REFINE_STEPS}"
    fi
  done
}

run_steps_sweep() {
  for value in 1 2 3 4 5; do
    if [[ "${value}" == "${DEFAULT_REFINE_STEPS}" ]]; then
      run_configuration default "${DEFAULT_RHO_MAX}" "${DEFAULT_MASK_PROB}" "${value}"
    else
      run_configuration "refine_steps_${value}" "${DEFAULT_RHO_MAX}" "${DEFAULT_MASK_PROB}" "${value}"
    fi
  done
}

case "${SWEEP}" in
  rho)
    run_rho_sweep
    ;;
  mask)
    run_mask_sweep
    ;;
  steps)
    run_steps_sweep
    ;;
  all)
    # Run the shared default only once, then all non-default configurations.
    run_configuration default "${DEFAULT_RHO_MAX}" "${DEFAULT_MASK_PROB}" "${DEFAULT_REFINE_STEPS}"
    for value in 0.10 0.20 0.40 0.50; do
      run_configuration "rho_max_${value//./p}" "${value}" "${DEFAULT_MASK_PROB}" "${DEFAULT_REFINE_STEPS}"
    done
    for value in 0 0.05 0.20 0.30; do
      run_configuration "mask_${value//./p}" "${DEFAULT_RHO_MAX}" "${value}" "${DEFAULT_REFINE_STEPS}"
    done
    for value in 1 2 4 5; do
      run_configuration "refine_steps_${value}" "${DEFAULT_RHO_MAX}" "${DEFAULT_MASK_PROB}" "${value}"
    done
    ;;
  *)
    echo "[ERROR] Unknown SWEEP=${SWEEP}. Use all, rho, mask, or steps." >&2
    exit 2
    ;;
esac

echo
echo "Sensitivity sweep finished: ${RESULT_BASE}"
echo "Collect results with:"
echo "python ${SCRIPT_DIR}/collect_sensitivity_results.py --root ${RESULT_BASE}"
