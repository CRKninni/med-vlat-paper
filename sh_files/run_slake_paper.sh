#!/usr/bin/env bash
# MED-VLAT paper — SLAKE VQA fine-tuning (86.24% linear attention, 140 epochs)
# Architecture: FGGF (FineGrainedGlobalFeature) — 6 layers, head_schedule=[12,12,8,6,4,4]
#               Bidirectional + Adaptive Gating
# Pretrain ckpt: enhanced_pretrain_roco_medicat_clef_29.pth
# GPU: CUDA 0 (A6000, 49 GB)
set -euo pipefail

source /home/gen/.virtualenvs/torch_112/bin/activate
cd "$(dirname "$0")/.."

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

PRETRAIN_CKPT="${1:-output/pretrain_enhanced/enhanced_pretrain_roco_medicat_clef_29.pth}"
OUT_DIR="output/paper/slake"

mkdir -p "$OUT_DIR"

python -u train_enhanced.py \
  --config   configs/VQA_Slake_attention140.yaml \
  --output_dir "$OUT_DIR" \
  --attention_mode linear \
  --fg_layers 6 \
  --head_schedule "12,12,8,6,4,4" \
  --bidirectional \
  --adaptive_gating \
  --dropout 0.1 \
  --pretrain_checkpoint "$PRETRAIN_CKPT" \
  --device cuda:0 \
  --eval_freq 10 \
  --seed 42 \
  2>&1 | tee "$OUT_DIR/train.log"

echo ""
echo "Done. Logs: $OUT_DIR/train.log"
