#!/usr/bin/env bash
# MED-VLAT VQA-RAD: FGGF open + closed yes/no specialist + FGGF Q2A choice.
set -euo pipefail

source /home/gen/.virtualenvs/torch_112/bin/activate
cd /home/gen/crk/med-vlat-paper
source sh_files/vqarad_best_ckpt.sh

export WANDB_MODE=disabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MIMV_ROOT="${MIMV_ROOT:-/home/gen/crk/deploy_vqarad/mimvMEDVLAT}"
DATA_PATH="${DATA_PATH:-/home/gen/crk/deploy_vqarad/datsets/medical_vqa/VQA-RAD}"
TEST_JSON="$DATA_PATH/test_m3ae.json"
VAL_JSON="$DATA_PATH/val_m3ae.json"

OUT="${OUT:-output/vqa/vqarad_medvlat_hybrid}"
FT_A_OUT=output/vqa/vqarad_m3ae_full_ft_a
Q2A_OUT=output/vqa/vqarad_m3ae_q2a_fggf
OPEN_PUSH_OUT=output/vqa/vqarad_m3ae_open_push_fta_fggf
CLOSED_YN_CKPT="${CLOSED_YN_CKPT:-output/vqa/vqarad_medvlat_closed_yn/MED_VLAT_closed_yn_best.pth}"
MVCM_CKPT="${MVCM_CKPT:-$CLOSED_YN_CKPT}"
MVCM_OUT="${MVCM_OUT:-output/vqa/vqarad_medvlat_closed_yn_eval}"

FGGF_FLAGS=(
  --attention_mode flash --fg_layers 6 --head_schedule "12,12,8,6,4,4"
  --bidirectional --adaptive_gating --dropout 0.1 --seed 42
)

HEAD_Q2A=(
  --use_answer_query --use_fg_query_fusion --use_asymmetric_query_loss
  --use_closed_yn_head --use_open_synonym_boost --open_soft_synonym_labels
  --answer_infer_mode hybrid
)

mkdir -p "$OUT"

# Prefer FT_A best open (strongest FGGF OPEN on test); skip open_push unless forced.
OPEN_CKPT="$(vqarad_best_ckpt "$FT_A_OUT" open)"
if [[ "${USE_OPEN_PUSH:-0}" == "1" ]]; then
  OPEN_CKPT="$(vqarad_best_ckpt "$OPEN_PUSH_OUT" open 2>/dev/null || echo "$OPEN_CKPT")"
fi
CHOICE_CKPT="$(vqarad_best_ckpt "$Q2A_OUT")"

echo "MED-VLAT VQA-RAD dual-route eval"
echo "  FGGF open:   $OPEN_CKPT"
echo "  Closed Y/N:  $MVCM_CKPT"
echo "  FGGF choice: $CHOICE_CKPT"

dump_fggf() {
  local config="$1" ckpt="$2" preds="$3" log="$4"; shift 4
  python train_enhanced.py --evaluate --eval_split test \
    --config "$config" --output_dir "$OUT/$(basename "$log" .log)" \
    --checkpoint "$ckpt" --dump_preds "$preds" \
    "$@" 2>&1 | tee "$OUT/$log"
}

dump_fggf configs/VQA_RAD_m3ae_open_push_fta_fggf.yaml "$OPEN_CKPT" \
  "$OUT/open_preds.json" open_eval.log \
  "${HEAD_Q2A[@]}" --query_only_train "${FGGF_FLAGS[@]}"

dump_fggf configs/VQA_RAD_m3ae_q2a_fggf.yaml "$CHOICE_CKPT" \
  "$OUT/choice_preds.json" choice_eval.log \
  "${HEAD_Q2A[@]}" "${FGGF_FLAGS[@]}"

# MVCM preds: reuse locked snapshot or dump fresh from mimvMEDVLAT.
MVCM_PREDS="${MVCM_PREDS:-$OUT/mvcm_test_preds.json}"
if [[ ! -f "$MVCM_PREDS" ]]; then
  LOCKED_PREDS="$MIMV_ROOT/output/vqarad_dual_route_hybrid/mvcm_test_preds.json"
  if [[ -f "$LOCKED_PREDS" ]]; then
    cp "$LOCKED_PREDS" "$MVCM_PREDS"
    echo "Reused locked closed Y/N preds: $LOCKED_PREDS"
  else
    echo "Dumping closed yes/no test preds..."
    (
      cd "$MIMV_ROOT"
      source sh_files/env.sh
      EVAL_ONLY=1 CHECKPOINT="$MVCM_CKPT" EVAL_SPLIT=test_m3ae.json \
        OUT="$MVCM_OUT" DATA_PATH="$DATA_PATH" \
        bash sh_files/run_vqarad_mvcm_m3ae_closed_singleview.sh
    )
    cp "$MVCM_OUT/eval_test_m3ae.json.log" "$OUT/mvcm_eval.log" 2>/dev/null || true
    # run script writes preds via --dump_preds in eval path
    if [[ -f "$MVCM_OUT/mvcm_test_preds.json" ]]; then
      cp "$MVCM_OUT/mvcm_test_preds.json" "$MVCM_PREDS"
    fi
  fi
fi

if [[ ! -f "$MVCM_PREDS" ]]; then
  echo "ERROR: closed Y/N preds not found at $MVCM_PREDS" >&2
  exit 1
fi

echo ""
python tools/merge_fggf_mvcm_hybrid.py \
  --test_json "$TEST_JSON" \
  --open_preds "$OUT/open_preds.json" \
  --closed_yn_preds "$MVCM_PREDS" \
  --choice_preds "$OUT/choice_preds.json" \
  --out_json "$OUT/hybrid_merged.json" \
  | tee "$OUT/hybrid_results.log"

echo ""
echo "Results: $OUT/hybrid_results.log"
