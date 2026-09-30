#!/bin/bash
# Repeats the Table 3 conditions with several random seeds, plus the extra
# condition "Robinson + isotropic" that separates the heading sampler from
# the kernel (row 3 vs. row 4 differ in both). Results go to results_seeds/,
# the single runs in results/ from run_table3_final.sh are not touched.
#
#   row1   scarce real data only                 isotropic
#   row2   narrow range [-60,60] (prior method)  isotropic
#   row3   naive uniform 360 deg                 isotropic
#   row4   Robinson/Star headings                anisotropic  (full method)
#   row4b  Robinson/Star headings                isotropic    (new: sampler only)
#
# Every (condition, seed) pair is one training run (train.py --seed) plus
# evaluation on data/real_heldout_test.txt. Runs are spread over the GPUs in
# GPUS, PER_GPU at a time on each GPU (one run uses about a third of an
# RTX 4000 Ada). Finished runs (eval.csv present) are skipped, so the script
# can simply be restarted after an interruption.
#
# Output: results_seeds/<cond>/seed<k>/{eval.csv,train.log}, and
# results_seeds/summary.csv / summary.txt with mean and standard deviation
# over seeds per condition and IoU threshold.
#
# Usage: ./run_table3_seeds.sh
# Env:   CONDS    (default "row1 row2 row3 row4 row4b")
#        SEEDS    (default "1 2 3")
#        GPUS     (default "0 1")
#        PER_GPU  (default 3)
#        EPOCHS   (default 200)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDS="${CONDS:-row1 row2 row3 row4 row4b}"
SEEDS="${SEEDS:-1 2 3}"
GPUS="${GPUS:-0 1}"
PER_GPU="${PER_GPU:-3}"
EPOCHS="${EPOCHS:-200}"
RES="${RES:-$HERE/results_seeds}"
export HERE RES EPOCHS PYTHONUNBUFFERED=1

for f in real_scarce_train.txt real_heldout_test.txt val_synthetic.txt \
         train_synthetic_old_range.txt train_naive_360.txt train_robinson.txt; do
  [ -f "$HERE/data/$f" ] || { echo "missing data/$f" >&2; exit 1; }
done

run_one () {
  COND="$1"; SEED="$2"; GPU="$3"
  case "$COND" in
    row1)  DATA=real_scarce_train.txt;         ARCH=isotropic ;;
    row2)  DATA=train_synthetic_old_range.txt; ARCH=isotropic ;;
    row3)  DATA=train_naive_360.txt;           ARCH=isotropic ;;
    row4)  DATA=train_robinson.txt;            ARCH=anisotropic ;;
    row4b) DATA=train_robinson.txt;            ARCH=isotropic ;;
    *) echo "unknown condition $COND" >&2; return 1 ;;
  esac
  OUT="$RES/$COND/seed$SEED"
  [ -f "$OUT/eval.csv" ] && { echo "$(date +%H:%M) $COND seed $SEED: already done"; return 0; }
  mkdir -p "$OUT"
  echo "$(date +%H:%M) $COND seed $SEED: start on GPU $GPU"
  if CUDA_VISIBLE_DEVICES="$GPU" python3 "$HERE/code/train.py" \
       --train-data "$HERE/data/$DATA" --val-data "$HERE/data/val_synthetic.txt" \
       --arch "$ARCH" --epochs "$EPOCHS" --seed "$SEED" --device cuda \
       --model-name "${COND}_seed$SEED" --out-dir "$OUT" > "$OUT/train.log" 2>&1 \
     && python3 "$HERE/code/evaluate.py" --data "$HERE/data/real_heldout_test.txt" \
       --arch "$ARCH" --model "$OUT/models/${COND}_seed$SEED.pt" \
       --out "$OUT/eval.csv" --condition "${COND}_seed$SEED" > "$OUT/eval.log" 2>&1; then
    echo "$(date +%H:%M) $COND seed $SEED: done"
  else
    echo "$(date +%H:%M) $COND seed $SEED: FAILED, see $OUT/train.log / eval.log"
  fi
}
export -f run_one

# job list "cond seed gpu"; synthetic runs first (they take longest),
# GPUs assigned round-robin
set -- $GPUS
NGPU=$#
I=0
JOBS=""
for COND in $CONDS; do
  [ "$COND" = row1 ] && continue
  for SEED in $SEEDS; do
    GPU=$(echo $GPUS | cut -d' ' -f$((I % NGPU + 1))); I=$((I + 1))
    JOBS+="$COND $SEED $GPU"$'\n'
  done
done
case " $CONDS " in *" row1 "*)
  for SEED in $SEEDS; do
    GPU=$(echo $GPUS | cut -d' ' -f$((I % NGPU + 1))); I=$((I + 1))
    JOBS+="row1 $SEED $GPU"$'\n'
  done ;;
esac

echo "$(date +%H:%M) $(printf '%s' "$JOBS" | grep -c .) runs, $((NGPU * PER_GPU)) in parallel"
printf '%s' "$JOBS" | xargs -P $((NGPU * PER_GPU)) -L 1 bash -c 'run_one "$@"' _

# mean and standard deviation over seeds, per condition and IoU threshold
python3 - "$RES" <<'EOF'
import csv, glob, os, statistics, sys
res = sys.argv[1]
names = {"row1": "1  real only           isotropic",
         "row2": "2  narrow [-60,60]     isotropic",
         "row3": "3  uniform 360         isotropic",
         "row4": "4  Robinson            anisotropic",
         "row4b": "4b Robinson            isotropic"}
rows, lines = [], []
for cond in names:
    vals = {}
    for f in sorted(glob.glob(os.path.join(res, cond, "seed*", "eval.csv"))):
        for r in csv.DictReader(open(f)):
            vals.setdefault(float(r["iou_threshold"]), []).append((float(r["precision"]), float(r["recall"])))
    if not vals:
        continue
    cells = []
    for thr in sorted(vals):
        p = [x[0] for x in vals[thr]]
        r = [x[1] for x in vals[thr]]
        v = p
        m = lambda x: statistics.mean(x)
        s = lambda x: statistics.stdev(x) if len(x) > 1 else 0.0
        rows.append({"condition": cond, "iou_threshold": thr, "n_seeds": len(p),
                     "mean_precision": round(m(p), 4), "sd_precision": round(s(p), 4),
                     "mean_recall": round(m(r), 4), "sd_recall": round(s(r), 4),
                     "precision_values": " ".join(f"{x:.4f}" for x in p),
                     "recall_values": " ".join(f"{x:.4f}" for x in r)})
        if p == r:
            cells.append(f"IoU {thr:.2f}: P=R {m(p):.3f} +- {s(p):.3f}")
        else:
            cells.append(f"IoU {thr:.2f}: P {m(p):.3f} +- {s(p):.3f}, R {m(r):.3f} +- {s(r):.3f}")
    lines.append(f"{names[cond]}  (n={len(v)})  " + "   ".join(cells))
with open(os.path.join(res, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "iou_threshold", "n_seeds", "mean_precision", "sd_precision",
                                      "mean_recall", "sd_recall", "precision_values", "recall_values"])
    w.writeheader(); w.writerows(rows)
text = "mean +- sd over seeds (P = precision, R = recall)\n" + "\n".join(lines) + "\n"
open(os.path.join(res, "summary.txt"), "w").write(text)
print(text)
EOF
