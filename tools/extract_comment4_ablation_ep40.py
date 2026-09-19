#!/usr/bin/env python3
"""Print test Overall @ epoch 40 from Comment-4 ablation train.log files."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    'R4-A full': 'output/vqa/ablation_gating/tier3_40ep/slake_full/train.log',
    'R4-A no_gating SLAKE': 'output/vqa/ablation_gating/slake_no_gating/train.log',
    'R4-A const SLAKE': 'output/vqa/ablation_gating/slake_const_heads/train.log',
    'R4-A full Path': 'output/vqa/ablation_gating/tier3_40ep/path_full/train.log',
    'R4-A no_gating Path': 'output/vqa/ablation_gating/path_no_gating/train.log',
    'R4-A const Path': 'output/vqa/ablation_gating/path_const_heads/train.log',
    'R4-A full RAD': 'output/vqa/ablation_gating/vqarad_40ep/full/train.log',
    'R4-A no_gating RAD': 'output/vqa/ablation_gating/vqarad_40ep/no_gating/train.log',
    'R4-A const RAD': 'output/vqa/ablation_gating/vqarad_40ep/const_heads/train.log',
    'R4-B SLAKE 4L': 'output/vqa/ablation_gating/tier3_40ep/slake_4layer/train.log',
    'R4-B SLAKE min4': 'output/vqa/ablation_gating/tier3_40ep/slake_min4head/train.log',
    'R4-B SLAKE inv': 'output/vqa/ablation_gating/tier3_40ep/slake_inverted/train.log',
    'R4-B Path 4L': 'output/vqa/ablation_gating/tier3_40ep/path_4layer/train.log',
    'R4-B Path min4': 'output/vqa/ablation_gating/tier3_40ep/path_min4head/train.log',
    'R4-B Path inv': 'output/vqa/ablation_gating/tier3_40ep/path_inverted/train.log',
    'R4-B RAD 4L': 'output/vqa/ablation_gating/vqarad_40ep/layer4/train.log',
    'R4-B RAD min4': 'output/vqa/ablation_gating/vqarad_40ep/min4head/train.log',
    'R4-B RAD inv': 'output/vqa/ablation_gating/vqarad_40ep/inverted/train.log',
}


def overall_at_epoch_40(log_path: Path) -> float | None:
    text = log_path.read_text(errors='ignore')
    marker = 'Running validation at epoch 40'
    idx = text.rfind(marker)
    if idx < 0:
        return None
    m = re.search(r'Overall:\s*([\d.]+)%', text[idx : idx + 2500])
    return float(m.group(1)) if m else None


def main():
    for name, rel in RUNS.items():
        p = ROOT / rel
        if not p.is_file():
            print(f'{name}: MISSING ({rel})', file=sys.stderr)
            continue
        acc = overall_at_epoch_40(p)
        print(f'{name}: {acc:.2f}%' if acc is not None else f'{name}: (no ep40 eval)')


if __name__ == '__main__':
    main()
