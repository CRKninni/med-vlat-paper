# Paper best checkpoints (3 artifacts)

Matches **Table 3** test accuracy:

| File | Benchmark | Overall |
|------|-----------|---------|
| `slake_paper_best.pth` | SLAKE | **86.24%** |
| `pathvqa_paper_best.pth` | PathVQA | **65.66%** |
| `vqarad_paper_hybrid.tar.gz` | VQA-RAD (MED-VLAT dual-route) | **81.82%** |

**Download:** [GitHub Release `paper-checkpoints-v1`](https://github.com/CRKninni/med-vlat-paper/releases/tag/paper-checkpoints-v1)  
Large files are split (`*.part00`, …); see `REASSEMBLE.txt` in the release.

### VQA-RAD tarball contents

| Weight | Role |
|--------|------|
| `vqarad_fggf_open_best_open.pth` | Open-ended (FGGF) |
| `vqarad_fggf_q2a_best.pth` | Closed multi-choice (FGGF + answer query) |
| `vqarad_medvlat_closed_yn_best.pth` | Closed yes/no specialist |

```bash
tar -xzf vqarad_paper_hybrid.tar.gz -C checkpoints/vqarad_hybrid
bash sh_files/eval_vqarad_medvlat_locked.sh
```

See `vqarad_hybrid_LOCKED.json` for locked metrics and paths.
