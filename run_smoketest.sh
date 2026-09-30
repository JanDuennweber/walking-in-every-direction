#!/bin/bash
# Verifies the environment and pipeline work end-to-end before committing
# to a real (hours-long) training run: 2 epochs on 4 duplicate rows, both
# architectures, then evaluate.py. Zero claim about model quality -- this
# only proves nothing crashes. Run this first on a new machine.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results/_smoketest"
rm -rf "$OUT"

echo "=== isotropic ==="
python3 "$HERE/code/train.py" \
  --train-data "$HERE/data/smoketest/tiny_train.txt" \
  --val-data "$HERE/data/smoketest/tiny_val.txt" \
  --arch isotropic --epochs 2 --batch-size 2 \
  --model-name smoketest_iso --out-dir "$OUT/iso" --device cpu

echo "=== anisotropic ==="
python3 "$HERE/code/train.py" \
  --train-data "$HERE/data/smoketest/tiny_train.txt" \
  --val-data "$HERE/data/smoketest/tiny_val.txt" \
  --arch anisotropic --epochs 2 --batch-size 2 \
  --model-name smoketest_aniso --out-dir "$OUT/aniso" --device cpu

echo "=== evaluate ==="
python3 "$HERE/code/evaluate.py" \
  --data "$HERE/data/smoketest/tiny_val.txt" --arch isotropic \
  --model "$OUT/iso/models/smoketest_iso.pt" --out "$OUT/iso/eval.csv" --condition smoketest

echo "OK -- pipeline runs end to end. See $OUT for output."
