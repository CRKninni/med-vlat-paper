#!/usr/bin/env bash
# MED-VLAT VQA-RAD dual-route (wrapper; implements same logic as legacy hybrid eval).
set -euo pipefail
OUT="${OUT:-output/vqa/vqarad_medvlat_hybrid}"
export OUT
export CLOSED_YN_CKPT="${CLOSED_YN_CKPT:-output/vqa/vqarad_medvlat_closed_yn/MED_VLAT_closed_yn_best.pth}"
exec bash sh_files/eval_vqarad_fggf_mvcm_hybrid.sh
