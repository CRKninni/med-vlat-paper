#!/usr/bin/env bash
# GitHub release assets must be < 2 GiB each. Split large checkpoints into ~1.9 GiB parts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$REPO_ROOT/checkpoints/release_staging"
PARTS="$STAGING/parts"
CHUNK="${CHUNK_SIZE:-1900M}"

mkdir -p "$PARTS"
rm -f "$PARTS"/*

split_one() {
  local src="$1" base="$2"
  if [[ ! -f "$src" ]]; then
    echo "Missing: $src" >&2
    exit 1
  fi
  local size
  size=$(stat -c%s "$src")
  if (( size < 2000000000 )); then
    cp -l "$src" "$PARTS/$base"
    echo "copy (no split): $base"
    return
  fi
  split -b "$CHUNK" -d -a 2 "$src" "$PARTS/${base}.part"
  echo "split: $base -> ${base}.part*"
}

split_one "$STAGING/slake_paper_best.pth" "slake_paper_best.pth"
split_one "$STAGING/pathvqa_paper_best.pth" "pathvqa_paper_best.pth"
split_one "$STAGING/vqarad_paper_hybrid.tar.gz" "vqarad_paper_hybrid.tar.gz"

(
  cd "$PARTS"
  sha256sum ./* > SHA256SUMS
)

cat > "$PARTS/REASSEMBLE.txt" <<'EOF'
# Reassemble after download (run from this folder)

cat slake_paper_best.pth.part* > slake_paper_best.pth
cat vqarad_paper_hybrid.tar.gz.part* > vqarad_paper_hybrid.tar.gz
# pathvqa_paper_best.pth is a single file (no parts)

sha256sum -c SHA256SUMS

mkdir -p ../vqarad_hybrid
tar -xzf vqarad_paper_hybrid.tar.gz -C ../vqarad_hybrid
# If tarball has legacy name:
# mv ../vqarad_hybrid/vqarad_mvcm_closed_best.pth ../vqarad_hybrid/vqarad_medvlat_closed_yn_best.pth
EOF

echo "Parts in $PARTS:"
ls -lh "$PARTS"
