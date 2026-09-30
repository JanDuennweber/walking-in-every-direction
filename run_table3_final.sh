#!/bin/bash
# Runs ONLY the four Table 3 (Subsection 4.3) conditions that remain in the
# paper after the ablation was reduced to "designed to work together, not
# independently".
#
#   row 1  scarce real data only                     isotropic    (data ready)
#   row 2  narrow range [-60,60] (prior method)      isotropic    (data ready)
#   row 3  naive uniform 360 deg                     isotropic    (needs Unity data)
#   row 4  Robinson/Star scaffold (full method)      anisotropic  (needs Unity data)
#
# Rows 3 and 4 need synthetic training data that only Geyer's Unity project
# can produce: data_generation/unity/generate_table3_data.sh. This script
# writes the two heading CSVs Unity needs, runs every row whose training data exists,
# and names the missing files for the rest -- it never substitutes other
# data for a missing condition.
#
# All rows are evaluated on the SAME held-out real test split
# (data/real_heldout_test.txt). Finished rows (results/<row>/eval.csv
# present) are skipped; set FORCE=1 to retrain them.
#
# Output: results/<row>/eval.csv per row, plus results/table3_rows.tex --
# the LaTeX table body with Precision/Recall at IoU 0.50, only for rows
# that actually ran; missing rows stay TBD.
#
# ROWS selects which rows to run (default: all). To use two GPUs, start
# two instances with disjoint rows, e.g.
#   CUDA_VISIBLE_DEVICES=0 ROWS="1 3" ./run_table3_final.sh
#   CUDA_VISIBLE_DEVICES=1 ROWS="2 4" ./run_table3_final.sh
# Each instance rewrites results/table3_rows.tex from all eval.csv files
# present, so the instance that finishes last writes the complete table.
#
# Usage: EPOCHS=200 DEVICE=cuda ./run_table3_final.sh
# Run ./run_smoketest.sh first on a new machine.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
EPOCHS="${EPOCHS:-200}"           # 200 matches the paper's reported training run
DEVICE="${DEVICE:-cuda}"          # set DEVICE=cpu to force CPU
FORCE="${FORCE:-0}"
ROWS="${ROWS:-1 2 3 4}"
N_HEADINGS="${N_HEADINGS:-40000}" # margin over the 25918 examples of the original set
TEST_DATA="$HERE/data/real_heldout_test.txt"
NAIVE_DATA="$HERE/data/train_naive_360.txt"
ROBINSON_DATA="$HERE/data/train_robinson.txt"
HEADINGS_DIR="$HERE/data_generation/headings"

if [ ! -f "$TEST_DATA" ]; then
  echo "Splitting real_scarce.txt into train/test (see code/split_real_data.py docstring)..."
  python3 "$HERE/code/split_real_data.py" --input "$HERE/data/real_scarce.txt" \
    --train-out "$HERE/data/real_scarce_train.txt" --test-out "$TEST_DATA" --n-test 16 --seed 0
fi

# Heading CSVs for the Unity-side data generation of rows 3 and 4.
mkdir -p "$HEADINGS_DIR"
for MODE in naive_360 robinson; do
  if [ ! -f "$HEADINGS_DIR/${MODE}_headings.csv" ]; then
    python3 "$HERE/data_generation/generate_headings.py" --mode "$MODE" \
      --n "$N_HEADINGS" --seed 0 --out "$HEADINGS_DIR/${MODE}_headings.csv"
  fi
done

MISSING=()
run_condition () {
  NAME="$1"; DATA="$2"; ARCH="$3"
  OUT="$HERE/results/$NAME"
  if [ ! -f "$DATA" ]; then
    echo "=== $NAME: SKIPPED, training data missing: $DATA"
    MISSING+=("$NAME -> $DATA")
    return
  fi
  if [ -f "$OUT/eval.csv" ] && [ "$FORCE" != "1" ]; then
    echo "=== $NAME: already done ($OUT/eval.csv), skipping (FORCE=1 to rerun)"
    return
  fi
  echo "=== $NAME (arch=$ARCH, data=$DATA) ==="
  python3 "$HERE/code/train.py" \
    --train-data "$DATA" --val-data "$HERE/data/val_synthetic.txt" \
    --arch "$ARCH" --epochs "$EPOCHS" --model-name "$NAME" --out-dir "$OUT" --device "$DEVICE"
  python3 "$HERE/code/evaluate.py" \
    --data "$TEST_DATA" --arch "$ARCH" \
    --model "$OUT/models/$NAME.pt" --out "$OUT/eval.csv" --condition "$NAME"
}

for ROW in $ROWS; do
  case "$ROW" in
    1) run_condition row1_real_only            "$HERE/data/real_scarce_train.txt"         isotropic ;;
    2) run_condition row2_narrow_isotropic     "$HERE/data/train_synthetic_old_range.txt" isotropic ;;
    3) run_condition row3_naive360_isotropic   "$NAIVE_DATA"                              isotropic ;;
    4) run_condition row4_robinson_anisotropic "$ROBINSON_DATA"                           anisotropic ;;
    *) echo "unknown row: $ROW (valid: 1 2 3 4)" >&2; exit 1 ;;
  esac
done

# LaTeX table body for Table 3 (04-Experiments.tex), built only from eval.csv
# files that exist; anything else stays TBD.
python3 - "$HERE/results" <<'EOF'
import csv, os, sys
res = sys.argv[1]
rows = [
    ("1", "row1_real_only", r"scarce real data only, no enrichment", "isotropic", r"absolute floor"),
    ("2", "row2_narrow_isotropic", r"narrow range $[-60^\circ,60^\circ]$~\cite{geyer26ie2026}", "isotropic", r"prior published method"),
    ("3", "row3_naive360_isotropic", r"naive uniform $360^\circ$", "isotropic", r"``widening the range alone'' claim"),
    ("4", "row4_robinson_anisotropic", r"Robinson/Star scaffold", "anisotropic", r"full combined method"),
]
lines = []
for num, name, data, arch, tests in rows:
    prec = rec = "TBD"
    path = os.path.join(res, name, "eval.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if abs(float(r["iou_threshold"]) - 0.5) < 1e-9:
                prec, rec = f'{float(r["precision"]):.3f}', f'{float(r["recall"]):.3f}'
    lines.append(f"        {num} & {data} & {arch} & {tests} & {prec} & {rec} \\\\")
os.makedirs(res, exist_ok=True)
out = os.path.join(res, "table3_rows.tex")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote", out, "(Precision/Recall at IoU 0.50)")
EOF

if [ ${#MISSING[@]} -gt 0 ]; then
  echo ""
  echo "NOT RUN -- training data missing for:"
  for M in "${MISSING[@]}"; do echo "  $M"; done
  echo "Generate it with data_generation/unity/generate_table3_data.sh"
  echo "(needs Unity 2022.3.26f1), then rerun this script; finished rows are skipped."
fi
