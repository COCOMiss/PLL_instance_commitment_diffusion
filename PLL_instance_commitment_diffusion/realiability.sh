#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# Reliability analysis for Stage1 vs Stage3
#
# Stage1:
#   PLL + Instance Commitment output
#
# Stage3:
#   Diffusion refinement + DCE output
#
# No training.
# Only inference and evaluation.
# ============================================================


# -----------------------------
# Environment
# -----------------------------

GPU=${GPU:-1}

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONUNBUFFERED=1



# -----------------------------
# Dataset
# -----------------------------

DATASET=${DATASET:-ny_filterPOI}



# -----------------------------
# Checkpoints
# Modify PREFIX if needed
# -----------------------------

PREFIX=${PREFIX:-diffusion_generated_softce03}



STAGE1="result/${DATASET}/instance_commitment_diffusion/diffusion_generated_softce03_stage1_base/best_model.pth"



STAGE3="result/${DATASET}/instance_commitment_diffusion/diffusion_generated_softce03_stage3_joint_dce/best_model.pth"



# -----------------------------
# Output
# -----------------------------

OUT_DIR="result/${DATASET}/reliability_analysis"

mkdir -p ${OUT_DIR}



# -----------------------------
# Check checkpoint
# -----------------------------

if [ ! -f "${STAGE1}" ]; then

    echo "[ERROR] Stage1 checkpoint not found:"
    echo "${STAGE1}"

    exit 1

fi



if [ ! -f "${STAGE3}" ]; then

    echo "[ERROR] Stage3 checkpoint not found:"
    echo "${STAGE3}"

    exit 1

fi



echo "============================================================"
echo "Reliability Analysis"
echo "============================================================"

echo "Dataset:"
echo "${DATASET}"

echo ""

echo "Stage1:"
echo "${STAGE1}"

echo ""

echo "Stage3:"
echo "${STAGE3}"

echo ""

echo "Output:"
echo "${OUT_DIR}"

echo "============================================================"



# -----------------------------
# Run analysis
# -----------------------------


python PLL_instance_commitment_diffusion/analyze_reliability.py \
    --dataset "${DATASET}" \
    --stage1 "${STAGE1}" \
    --stage3 "${STAGE3}"



# -----------------------------
# Move outputs
# -----------------------------

if [ -d "reliability_analysis" ]; then

    mv reliability_analysis/* "${OUT_DIR}/"

    rmdir reliability_analysis || true

fi



echo ""
echo "============================================================"
echo "Finished."
echo "Results saved to:"
echo "${OUT_DIR}"
echo "============================================================"