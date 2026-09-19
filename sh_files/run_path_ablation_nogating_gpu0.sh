#!/usr/bin/env bash
# Reviewer Comment 4 — PathVQA 224 linear, hierarchical heads, NO adaptive gating
# Init: enhanced pretrain only (no PathVQA checkpoint)
set -euo pipefail

source /home/gen/.virtualenvs/torch_112/bin/activate
cd /home/gen/crk/med-vlat-paper

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRETRAIN=/home/gen/crk/med_vlat_latest1/output/pretrain_enhanced/enhanced_pretrain_roco_medicat_clef_29.pth
OUT=output/vqa/ablation_gating/path_no_gating

mkdir -p "$OUT"

python -u train_enhanced.py \
  --config configs/VQA_Path_ablation_40ep.yaml \
  --output_dir "$OUT" \
  --attention_mode linear \
  --fg_layers 6 \
  --head_schedule "12,12,8,6,4,4" \
  --bidirectional \
  --dropout 0.1 \
  --pretrain_checkpoint "$PRETRAIN" \
  --device cuda:0 \
  --eval_freq 10 \
  --seed 42 \
  2>&1 | tee "$OUT/train.log"

echo "PathVQA no-gating ablation complete: $OUT"
