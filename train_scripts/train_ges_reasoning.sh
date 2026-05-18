#!/usr/bin/env bash
# =============================================================================
# GesVLA Gesture Pre-training Script
#
# This script launches gesture reasoning pre-training. All configurable paths
# and hyper-parameters are defined at the top so you do NOT need to edit
# gesconfig.py when switching datasets or changing data locations.
#
# Usage:
#   bash train_scripts/train_onetwovla_cocktail.sh
# =============================================================================

set -euo pipefail

# ─── Timestamp for experiment naming ─────────────────────────────────────────
now_date=$(date "+%Y.%m.%d")
now_time=$(date "+%H.%M.%S")

# ─── GPU / Batch Size ────────────────────────────────────────────────────────
NUM_DEVICES=1
SINGLE_BATCH_SIZE=2
BATCH_SIZE=$((NUM_DEVICES * SINGLE_BATCH_SIZE))

SINGLE_VAL_BATCH_SIZE=2
VAL_BATCH_SIZE=$((NUM_DEVICES * SINGLE_VAL_BATCH_SIZE))

echo "batch_size=${BATCH_SIZE}  val_batch_size=${VAL_BATCH_SIZE}"

# ─── Project Root (auto-detected from script location) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ─── Environment Variables ───────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/data/cache"
export HUGGINGFACE_HUB_CACHE="${PROJECT_ROOT}/data/cache"
export HF_LEROBOT_HOME="${PROJECT_ROOT}/data"
export OPENPI_DATA_HOME="${PROJECT_ROOT}/data/models"

# =============================================================================
# DATA PATHS  —  Edit these when switching datasets / data locations.
# All paths are relative to PROJECT_ROOT; prefix with ${PROJECT_ROOT}/.
# =============================================================================

GESTURE_REPO_ID="${PROJECT_ROOT}/data/datasets/pointing_dataset_0214_jelly"
GESTURE_JSON="${GESTURE_REPO_ID}/reasoning.json"
GESTURE_PARQUET="data.parquet"

# =============================================================================
# TRAINING COMMAND
# =============================================================================

# ── Gesture Reasoning Pre-training (gesture_pretrain) ────────────────────────
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py gesture_pretrain \
    --exp-name="${now_date}/${now_time}/pretrain" \
    --data.repo-id "${GESTURE_REPO_ID}" \
    --data.reasoning-json-path "${GESTURE_JSON}" \
    --data.parquet-filename "${GESTURE_PARQUET}" \
    --batch-size=${BATCH_SIZE} \
    --val-batch-size=${VAL_BATCH_SIZE} \
    --no-wandb-enabled
