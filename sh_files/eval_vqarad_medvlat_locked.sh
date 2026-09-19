#!/usr/bin/env bash
# LOCKED VQA-RAD — MED-VLAT dual-route inference (81.82% on test_m3ae.json).
# Routes: FGGF open | FGGF Q2A (closed choice) | MED-VLAT closed yes/no specialist.
set -euo pipefail

source /home/gen/.virtualenvs/torch_112/bin/activate
cd /home/gen/crk/med-vlat-paper

OUT=output/vqa/vqarad_medvlat_hybrid
TEST_JSON=/home/gen/crk/deploy_vqarad/datsets/medical_vqa/VQA-RAD/test_m3ae.json
LOCKED="$OUT/LOCKED.json"
# Legacy output dir (same snapshots)
LEGACY_OUT=output/vqa/vqarad_fggf_mvcm_hybrid

mkdir -p "$OUT"
if [[ ! -f "$LOCKED" && -f "$LEGACY_OUT/LOCKED.json" ]]; then
  cp "$LEGACY_OUT/LOCKED.json" "$LOCKED"
fi

echo "=== MED-VLAT VQA-RAD dual-route (locked) ==="
echo "Manifest: $LOCKED"
echo ""

for f in LOCKED_fggf_open_test_preds.json LOCKED_fggf_choice_test_preds.json LOCKED_closed_yn_test_preds.json; do
  if [[ ! -f "$OUT/$f" && -f "$LEGACY_OUT/${f/LOCKED_closed_yn/LOCKED_mvcm}" ]]; then
    cp "$LEGACY_OUT/${f/LOCKED_closed_yn/LOCKED_mvcm}" "$OUT/$f" 2>/dev/null || true
  fi
  if [[ ! -f "$OUT/$f" && -f "$LEGACY_OUT/LOCKED_mvcm_test_preds.json" && "$f" == *closed_yn* ]]; then
    cp "$LEGACY_OUT/LOCKED_mvcm_test_preds.json" "$OUT/$f"
  fi
done

if [[ -f "$OUT/LOCKED_fggf_open_test_preds.json" \
   && -f "$OUT/LOCKED_closed_yn_test_preds.json" \
   && -f "$OUT/LOCKED_fggf_choice_test_preds.json" ]]; then
  echo "--- From LOCKED prediction snapshots ---"
  python tools/merge_fggf_mvcm_hybrid.py \
    --test_json "$TEST_JSON" \
    --open_preds "$OUT/LOCKED_fggf_open_test_preds.json" \
    --closed_yn_preds "$OUT/LOCKED_closed_yn_test_preds.json" \
    --choice_preds "$OUT/LOCKED_fggf_choice_test_preds.json" \
    --out_json "$OUT/LOCKED_hybrid_merged_test.json" \
    | tee "$OUT/LOCKED_hybrid_results.log"
  exit 0
fi

echo "--- Full eval from checkpoints ---"
OUT="$OUT" bash sh_files/eval_vqarad_medvlat_hybrid.sh | tee "$OUT/LOCKED_hybrid_results.log"
