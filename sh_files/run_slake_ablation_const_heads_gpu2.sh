#!/usr/bin/env bash
# Reviewer Comment 4 ablation: SLAKE 224 linear, constant head schedule 8x6, WITH adaptive gating
# Init: enhanced pretrain only (no SLAKE checkpoint)
set -euo pipefail

source /home/gen/.virtualenvs/torch_112/bin/activate
cd /home/gen/crk/med-vlat-paper

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRETRAIN=/home/gen/crk/med_vlat_latest1/output/pretrain_enhanced/enhanced_pretrain_roco_medicat_clef_29.pth
OUT=output/vqa/ablation_gating/slake_const_heads

mkdir -p "$OUT"

python -u train_enhanced.py \
  --config configs/VQA_Slake_ablation_40ep.yaml \
  --output_dir "$OUT" \
  --attention_mode linear \
  --fg_layers 6 \
  --head_schedule "8,8,8,8,8,8" \
  --bidirectional \
  --adaptive_gating \
  --dropout 0.1 \
  --pretrain_checkpoint "$PRETRAIN" \
  --device cuda:1 \
  --eval_freq 10 \
  --seed 42 \
  2>&1 | tee "$OUT/train.log"

echo "Constant-heads ablation complete: $OUT"
