#!/usr/bin/env bash
# Stage exactly 3 paper artifacts: SLAKE, PathVQA, VQA-RAD hybrid tarball.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$REPO_ROOT/checkpoints/paper_best_manifest.json"
STAGING="$REPO_ROOT/checkpoints/release_staging"

rm -rf "$STAGING"
mkdir -p "$STAGING"

python3 - <<'PY' "$MANIFEST" "$STAGING"
import json, os, subprocess, sys, tarfile

manifest_path, staging = sys.argv[1], sys.argv[2]
with open(manifest_path) as f:
    data = json.load(f)

for m in data["models"]:
    asset = m["release_asset"]
    if asset.endswith(".tar.gz"):
        tar_path = os.path.join(staging, asset)
        tmp = os.path.join(staging, "_vqarad_bundle")
        os.makedirs(tmp, exist_ok=True)
        for member in m["bundle_members"]:
            src = member["source_checkpoint"]
            if not os.path.isfile(src):
                print(f"MISSING: {src}", file=sys.stderr)
                sys.exit(1)
            dst = os.path.join(tmp, member["name_in_tar"])
            if os.path.lexists(dst):
                os.remove(dst)
            os.link(src, dst)
            print(f"bundle {member['name_in_tar']} <- {src}")
        with tarfile.open(tar_path, "w:gz") as tar:
            for name in sorted(os.listdir(tmp)):
                tar.add(os.path.join(tmp, name), arcname=name)
        print(f"created {tar_path}")
    else:
        src = m["source_checkpoint"]
        dst = os.path.join(staging, asset)
        if not os.path.isfile(src):
            print(f"MISSING: {src}", file=sys.stderr)
            sys.exit(1)
        os.link(src, dst)
        print(f"linked {asset} <- {src}")

print(f"\nStaged in {staging}:")
PY

ls -lh "$STAGING"
