#!/usr/bin/env bash
set -euo pipefail

echo "== Proximal Energy Sanity Check =="

rm -rf artifacts
mkdir -p artifacts

echo "Running: make sanity"
make sanity

OUT="artifacts/sanity_output.json"
if [[ ! -f "$OUT" ]]; then
  echo "ERROR: Missing $OUT"
  echo "Your 'make sanity' must generate: artifacts/sanity_output.json"
  exit 1
fi

python3 scripts/verify_output.py "$OUT"

echo "OK: sanity check passed"
