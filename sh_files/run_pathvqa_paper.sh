#!/usr/bin/env bash
# MED-VLAT paper — PathVQA fine-tuning (65.66% flash attention, aux-MLP heads, 384 px)
# Architecture: FGGF (FineGrainedGlobalFeature) — 6 layers, head_schedule=[12,12,8,6,4,4]
#               Bidirectional + Adaptive Gating
# Two-stage:
#   Stage 1 — flash attention backbone trained (output/paper/path_flash)
#   Stage 2 — aux-MLP closed-YN + open-answer heads added on top (output/paper/path_aux_mlp)
# GPU: CUDA 1 (A6000, 49 GB)
set -euo pipefail

source /home/gen/.virtualenvs/torch_112/bin/activate
cd "$(dirname "$0")/.."

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=1

PRETRAIN_CKPT="${1:-output/pretrain_enhanced/enhanced_pretrain_roco_medicat_clef_29.pth}"

FGGF_ARGS=(
  --fg_layers 6
  --head_schedule "12,12,8,6,4,4"
  --bidirectional
  --adaptive_gating
  --dropout 0.1
  --device cuda:0
  --seed 42
)

# ── Stage 1: flash backbone ────────────────────────────────────────────────
STAGE1_OUT="output/paper/path_flash"
mkdir -p "$STAGE1_OUT"
echo "========== Stage 1: PathVQA flash backbone ==========" | tee "$STAGE1_OUT/train.log"
python -u train_enhanced.py \
  --config  configs/VQA_Path_attention140.yaml \
  --output_dir "$STAGE1_OUT" \
  --attention_mode flash \
  --pretrain_checkpoint "$PRETRAIN_CKPT" \
  --eval_freq 10 \
  "${FGGF_ARGS[@]}" \
  2>&1 | tee -a "$STAGE1_OUT/train.log"

STAGE1_BEST="$STAGE1_OUT/VQA_Enhanced_best.pth"

# ── Stage 2: aux MLP (closed YN + open answer) ────────────────────────────
STAGE2_OUT="output/paper/path_aux_mlp"
mkdir -p "$STAGE2_OUT"
echo "========== Stage 2: PathVQA aux-MLP heads ==========" | tee "$STAGE2_OUT/train.log"
python -u train_enhanced.py \
  --config  configs/VQA_Path_aux_mlp.yaml \
  --output_dir "$STAGE2_OUT" \
  --checkpoint "$STAGE1_BEST" \
  --attention_mode flash \
  --eval_freq 5 \
  --save_freq 10 \
  "${FGGF_ARGS[@]}" \
  2>&1 | tee -a "$STAGE2_OUT/train.log"

echo ""
echo "Done. Final model: $STAGE2_OUT/VQA_Enhanced_best.pth"
