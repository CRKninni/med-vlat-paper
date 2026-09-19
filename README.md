# MED-VLAT — Paper Code

Official implementation for **MED-VLAT** medical VQA (SLAKE, PathVQA, VQA-RAD).

## Architecture

- **FGGF** (`FineGrainedGlobalFeature`): Fine-Grained + Global Feature fusion  
  - Bidirectional multimodal refinement with **adaptive gating** (Eq. 8–9)  
  - 6 layers, hierarchical heads `head_schedule=[12,12,8,6,4,4]`

## Best Results (test)

| Dataset | Overall | OPEN | CLOSED | Notes |
|---------|---------|------|--------|--------|
| SLAKE   | **86.24** | 85.58 | 87.26 | Linear @224, 140 ep, best @ ep110 |
| PathVQA | **65.66** | 41.63 | 89.65 | Flash @384 + aux MLP heads (2-stage) |
| VQA-RAD | **81.82** | 65.92 | 92.28 | **Dual-route MED-VLAT** (see below) |

Full tables: `results_tables.tex`

## Pretrained checkpoints (paper)

Download **[Release `paper-checkpoints-v1`](https://github.com/CRKninni/med-vlat-paper/releases/tag/paper-checkpoints-v1)** (~15 GB; large files are split — see `REASSEMBLE.txt` in the release).

| Asset | Benchmark |
|-------|-----------|
| `slake_paper_best.pth` (+ parts) | SLAKE 86.24% |
| `pathvqa_paper_best.pth` | PathVQA 65.66% |
| `vqarad_paper_hybrid.tar.gz` (+ parts) | VQA-RAD 81.82% |

Details: `checkpoints/README.md` and `checkpoints/paper_best_manifest.json`.

### VQA-RAD dual-route (81.82%)

Our **MED-VLAT** system routes each question to:

1. **FGGF** — open-ended answers (`vqarad_fggf_open_best_open.pth`)
2. **FGGF + answer query** — closed multi-choice (`vqarad_fggf_q2a_best.pth`)
3. **Closed yes/no specialist** — binary closed questions (`vqarad_medvlat_closed_yn_best.pth`)

```bash
tar -xzf checkpoints/vqarad_paper_hybrid.tar.gz -C checkpoints/vqarad_hybrid
bash sh_files/eval_vqarad_medvlat_locked.sh
```

Locked metrics: `checkpoints/vqarad_hybrid_LOCKED.json`.

> Release `paper-checkpoints-v1` may label the yes/no file `vqarad_mvcm_closed_best.pth`; rename to `vqarad_medvlat_closed_yn_best.pth` when extracting (same weights).

## Setup

```bash
pip install torch torchvision transformers pyyaml timm flash-attn
```

## Training

```bash
# SLAKE (86.24%)
bash sh_files/run_slake_paper.sh [pretrain_checkpoint.pth]

# PathVQA (65.66%, 2-stage)
bash sh_files/run_pathvqa_paper.sh [pretrain_checkpoint.pth]

# VQA-RAD — train FGGF branches (configs under configs/VQA_RAD_m3ae_*.yaml)
# Then dual-route eval:
bash sh_files/eval_vqarad_medvlat_hybrid.sh
```

Update dataset paths in `configs/*.yaml` for your environment.

## Structure

```
models/              VLAT + FGGF
train_enhanced.py    VQA fine-tuning
configs/             SLAKE, PathVQA, VQA-RAD
sh_files/            Training & eval scripts
checkpoints/         Manifest + VQA-RAD lock file (weights on Releases)
results_tables.tex   Paper tables (LaTeX)
```
