#!/bin/bash
# Generates the synthetic training data for Table 3 rows 3 and 4 with
# Geyer's Unity project, headless:
#
#   naive_360 -> sensfloor/data/train_naive_360.txt  (row 3)
#   robinson  -> sensfloor/data/train_robinson.txt   (row 4)
#
# For each mode: copies the Unity project into work/<mode>/ (the original
# is never modified), patches the copy (apply_unity_patch.py), runs the
# simulation in the Unity editor in batch mode until TARGET training lines
# with at least one foot exist (noise-only lines are dropped, as in row 2's
# data), and keeps the first TARGET lines -- 25918 by default, the size of
# row 2's data/train_synthetic_old_range.txt, so rows 2-4 train on equally
# many examples.
#
# Usage:  ./generate_table3_data.sh [naive_360|robinson|both]   (default: both)
# Env:    UNITY      editor binary (default: Unity Hub's 2022.3.26f1 install)
#         PROJECT    path to the original SensFloorSimulation Unity project (required;
#                    not part of this repository, see README.md)
#         TARGET     training lines per mode (default 25918)
#         TIMESCALE  Unity Time.timeScale (default 1 = as in the original run;
#                    higher is faster but only valid while the machine keeps up)
#         FORCE=1    overwrite an existing data/train_<mode>.txt
#         STALL      seconds without new output before Unity is killed and
#                    relaunched (default 600)
#
# Resumable: work/<mode>/ is kept between runs. Rerunning continues the
# existing TrainingData.txt instead of starting over, and a watchdog
# relaunches Unity if it stops writing output (e.g. a crash in the
# simulation coroutine leaves the editor alive but idle).
#
# Needs an activated Unity license (Personal is fine). Afterwards run
# ../../run_table3_final.sh, which picks the new files up automatically.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SENSFLOOR="$(cd "$HERE/../.." && pwd)"
UNITY="${UNITY:-$HOME/Unity/Hub/Editor/2022.3.26f1/Editor/Unity}"
PROJECT="${PROJECT:?set PROJECT=/path/to/SensFloorSimulation (the Unity project, see README.md)}"
TARGET="${TARGET:-25918}"
TIMESCALE="${TIMESCALE:-1}"
FORCE="${FORCE:-0}"
STALL="${STALL:-600}"
MAX_LAUNCHES="${MAX_LAUNCHES:-30}"
N_HEADINGS="${N_HEADINGS:-40000}"
HEADINGS_DIR="$SENSFLOOR/data_generation/headings"
MODES="${1:-both}"
[ "$MODES" = "both" ] && MODES="naive_360 robinson"

[ -x "$UNITY" ] || { echo "Unity editor not found: $UNITY (set UNITY=...)" >&2; exit 1; }
[ -f "$PROJECT/Assets/Scripts/Simulation.cs" ] || { echo "Unity project not found: $PROJECT (set PROJECT=...)" >&2; exit 1; }

for MODE in $MODES; do
  case "$MODE" in naive_360|robinson) ;; *) echo "unknown mode: $MODE" >&2; exit 1 ;; esac
  DEST="$SENSFLOOR/data/train_${MODE}.txt"
  if [ -f "$DEST" ] && [ "$FORCE" != "1" ]; then
    echo "=== $MODE: $DEST exists, skipping (FORCE=1 to regenerate)"
    continue
  fi

  HEADINGS="$HEADINGS_DIR/${MODE}_headings.csv"
  if [ ! -f "$HEADINGS" ]; then
    mkdir -p "$HEADINGS_DIR"
    python3 "$SENSFLOOR/data_generation/generate_headings.py" --mode "$MODE" \
      --n "$N_HEADINGS" --seed 0 --out "$HEADINGS"
  fi

  WORK="$HERE/work/$MODE"
  if [ ! -d "$WORK/project" ]; then
    echo "=== $MODE: copying project to $WORK"
    mkdir -p "$WORK"
    # Library/Temp/Logs are Unity caches (rebuilt on first open, and the
    # transferred ones are from Windows); generatedData holds Geyer's old output.
    rsync -a --exclude Library --exclude Temp --exclude Logs --exclude 'Assets/generatedData' \
      "$PROJECT/" "$WORK/project/"
  else
    echo "=== $MODE: resuming in existing $WORK"
  fi
  python3 "$HERE/apply_unity_patch.py" "$WORK/project"   # idempotent

  OUT_DIR="$WORK/output"
  mkdir -p "$OUT_DIR"
  feet_lines () { local n; n=$(grep -vc ':$' "$OUT_DIR/TrainingData.txt" 2>/dev/null); echo "${n:-0}"; }
  LAUNCH=0
  while [ "$(feet_lines)" -lt "$TARGET" ]; do
    LAUNCH=$((LAUNCH + 1))
    if [ "$LAUNCH" -gt "$MAX_LAUNCHES" ]; then
      echo "$MODE: gave up after $MAX_LAUNCHES Unity launches, see $WORK/unity_*.log" >&2; exit 1
    fi
    LOG="$WORK/unity_$LAUNCH.log"
    rm -f "$WORK/project/Temp/UnityLockfile"
    echo "=== $MODE: $(date +%H:%M) launch $LAUNCH, $(feet_lines)/$TARGET lines so far (log: $LOG)"
    T3_HEADINGS="$HEADINGS" T3_OUT_DIR="$OUT_DIR" T3_TARGET_LINES="$TARGET" T3_TIMESCALE="$TIMESCALE" \
      "$UNITY" -batchmode -projectPath "$WORK/project" \
      -executeMethod Table3DataGen.Run -logFile "$LOG" &
    PID=$!
    STARTED=$(date +%s)
    while kill -0 "$PID" 2>/dev/null; do
      sleep 30
      NOW=$(date +%s)
      LAST=$(stat -c %Y "$OUT_DIR/TrainingData.txt" 2>/dev/null || echo 0)
      [ "$LAST" -lt "$STARTED" ] && LAST=$STARTED   # project import after a launch
      if [ $((NOW - LAST)) -gt "$STALL" ]; then
        echo "=== $MODE: $(date +%H:%M) no new output for ${STALL}s, killing Unity (see $LOG)"
        kill "$PID" 2>/dev/null; sleep 20; kill -9 "$PID" 2>/dev/null || true
        break
      fi
    done
    wait "$PID" 2>/dev/null || true
  done

  # Keep only lines with at least one foot instance mask (not ending in ':'),
  # like train_synthetic_old_range.txt, which has no noise-only lines.
  grep -v ':$' "$OUT_DIR/TrainingData.txt" > "$OUT_DIR/TrainingData_withFeet.txt" || true
  LINES=$(wc -l < "$OUT_DIR/TrainingData_withFeet.txt")
  if [ "$LINES" -lt "$TARGET" ]; then
    echo "$MODE: only $LINES of $TARGET training lines with feet were written, see $WORK/unity_*.log" >&2
    exit 1
  fi
  head -n "$TARGET" "$OUT_DIR/TrainingData_withFeet.txt" > "$DEST"
  echo "=== $MODE: wrote $TARGET lines to $DEST"
done
