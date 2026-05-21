#!/bin/bash
CONFIG="gesvla_2vlm"
CHECKPOINT_DIR="..."
PORT=8000

python scripts/serve_policy_2vlm_withref.py \
    --config ${CONFIG} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --port ${PORT}
