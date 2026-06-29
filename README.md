# MED-VLAT — Paper Code

Official code release for **MED-VLAT** (SLAKE & PathVQA experiments).

## Architecture

- **FGGF** (`FineGrainedGlobalFeature`): Fine-Grained + Global Feature cross-modal module  
  - Phase 1: unidirectional text-guided image grounding  
  - Phase 2: bidirectional multimodal refinement with adaptive gating  
  - 6 layers, `head_schedule=[12,12,8,6,4,4]`

## Best Results

| Dataset | Overall | OPEN | CLOSED | Config |
|---------|---------|------|--------|--------|
| SLAKE   | 86.24   | 85.58 | 87.26 | `VQA_Slake_attention140.yaml`, linear |
| PathVQA | 65.66   | 41.63 | 89.65 | `VQA_Path_aux_mlp.yaml`, flash + aux MLP |

See `results_tables.tex` for full ablation tables.

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
```

Update dataset paths in `configs/*.yaml` to match your local SLAKE / PathVQA layout.

## Structure

```
models/          VLAT + FineGrainedGlobalFeature (FGGF)
train_enhanced.py   VQA fine-tuning
configs/         SLAKE & PathVQA configs
sh_files/        Launch scripts
results_tables.tex  Paper result tables (LaTeX)
```
