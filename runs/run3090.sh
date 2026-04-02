#!/bin/bash

# A conservative nanochat pipeline for a single RTX 3090 24GB.
# This is not the GPT-2 speedrun configuration. It trades capability for fit/stability.
#
# Usage:
#   bash runs/run3090.sh
# Optional overrides:
#   DEPTH=16 MAX_SEQ_LEN=1024 DEVICE_BATCH_SIZE=2 TOTAL_BATCH_SIZE=65536 bash runs/run3090.sh

set -euo pipefail

export OMP_NUM_THREADS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$SCRIPT_DIR}"
mkdir -p "$NANOCHAT_BASE_DIR"

# 3090 is Ampere (SM86), so bf16 is supported and preferable here.
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"

# -----------------------------------------------------------------------------
# Setup

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

WANDB_RUN="${WANDB_RUN:-dummy}"
MODEL_TAG="${MODEL_TAG:-3090_d18}"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
FORCE_RETRAIN_TOKENIZER="${FORCE_RETRAIN_TOKENIZER:-0}"
BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

# Conservative defaults for 24GB VRAM.
DEPTH="${DEPTH:-18}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-2048}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

python -m nanochat.dataset -n 8
if [ "$FORCE_RETRAIN_TOKENIZER" = "1" ] || [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
else
    echo "Tokenizer already exists at $TOKENIZER_DIR, skipping training."
fi

# -----------------------------------------------------------------------------
# Base model pretraining
#
# Notes for 3090:
# - Do not use --fp8: this repo only recommends it for H100+.
# - Use --window-pattern=L: without FA3, sliding-window attention is much slower.
# - Keep eval/sample frequencies low; they add a lot of runtime on a single card.

BASE_TRAIN_RESUME_ARG=""
if [ -d "$BASE_CKPT_DIR" ]; then
    LAST_MODEL_PATH="$(find "$BASE_CKPT_DIR" -maxdepth 1 -name 'model_*.pt' | sort | tail -n 1)"
    if [ -n "$LAST_MODEL_PATH" ]; then
        LAST_STEP="$(basename "$LAST_MODEL_PATH" | sed -E 's/^model_([0-9]+)\.pt$/\1/')"
        LAST_STEP="$(echo "$LAST_STEP" | sed -E 's/^0+//')"
        if [ -z "$LAST_STEP" ]; then
            LAST_STEP=0
        fi
        echo "Found existing base checkpoint at step $LAST_STEP in $BASE_CKPT_DIR, resuming."
        BASE_TRAIN_RESUME_ARG="--resume-from-step=$LAST_STEP"
    fi
fi

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth="$DEPTH" \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN" \
    --window-pattern=SSL \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --eval-every=500 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=2000 \
    $BASE_TRAIN_RESUME_ARG

python -m scripts.base_eval \
    --model-tag="$MODEL_TAG" \
    --device-batch-size=1 \
    --max-per-task=100 \
    --split-tokens=524288

# -----------------------------------------------------------------------------
# SFT

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every=500 \
    --eval-tokens=524288 \
    --chatcore-every=-1

python -m scripts.chat_eval -- -i sft --model-tag="$MODEL_TAG"
