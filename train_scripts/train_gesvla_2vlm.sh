#!/usr/bin/env bash
# =============================================================================
# GesVLA 2VLM Training Script
#
# This script launches the GesVLA 2VLM (two-VLM) training pipeline.
# All configurable paths are defined at the top and passed via CLI args
# to override the defaults in gesconfig_2vlm.py.
#
# Usage:
#   bash train_scripts/train_gesvla_2vlm.sh
# =============================================================================

set -euo pipefail

# ─── Timestamp for experiment naming ─────────────────────────────────────────
logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

# ─── GPU / Batch Size ────────────────────────────────────────────────────────
num_devices=$(nvidia-smi --list-gpus | wc -l)
single_batch_size=2
batch_size=$((num_devices * single_batch_size))
echo "batch_size=${batch_size}"

single_val_batch_size=2
val_batch_size=$((num_devices * single_val_batch_size))
echo "val_batch_size=${val_batch_size}"

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
# DATA PATHS  —  Edit these when switching datasets or data locations.
# All paths are relative to PROJECT_ROOT; prefix with ${PROJECT_ROOT}/.
# =============================================================================

# -- Reasoning JSON for action training --
act_reasoning_json_path="${PROJECT_ROOT}/data/reasoning/test_reasoning_0121_filtered.json"

# -- Pre-trained VLM0 checkpoint (from gesture reasoning pre-training) --
pretrain_ckpt_path="${PROJECT_ROOT}/data/checkpoints/purevl_pretrain/5999/params"

# =============================================================================
# TRAINING COMMANDS
# =============================================================================

# ── GesVLA 2VLM Training ────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_devices - 1))) XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train_2vlm.py gesvla_2vlm \
  --exp-name="${now_date}/${now_seconds}/pick" \
  --reasoning_json_path "${act_reasoning_json_path}" \
  --batch-size=${batch_size} \
  --val-batch-size=${val_batch_size} \
  --vlm0_weight_loader.params_path "${pretrain_ckpt_path}" \
  --no-wandb-enabled
