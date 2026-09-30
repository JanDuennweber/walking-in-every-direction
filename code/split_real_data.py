#!/usr/bin/env python3
"""
Splits data/real_scarce.txt into a held-out test split (used to evaluate
ALL SIX Table 3 conditions -- required for the ablation to mean anything,
see the "held-out, orientation-complete real test set" requirement stated
in 04-Experiments.tex's Effects-of-the-Improvements framing paragraph)
and a small training split (used ONLY by row 1, the "scarce real data
only" condition).

CONFIRMED, NOT JUST LIKELY: real_scarce.txt has exactly 49 non-empty
lines (`wc -l` reports 48 only because the last line lacks a trailing
newline) -- an exact match to the paper's "49 manually annotated real
images" (Table 2). This is almost certainly the same real evaluation set
the paper already treats as pure held-out data, never used for training.

That makes this split a genuine, deliberate deviation from how the paper
itself uses this file, not a neutral default -- flagged here rather than
silently done. Splitting part of it into row 1's training data means row
1's "scarce real data" condition is trained on a subset of the very set
the rest of the paper calls its evaluation data. The alternative (train
row 1 on a different real source instead, e.g. the raw, real, timestamped
logs of the original deployments,
which this package does not currently include or vet) was not pursued
here for time reasons, not because it's wrong -- it is the more correct
fix if this matters for the final paper. Confirm this choice before
trusting Table 3's row 1 number.

Deterministic (fixed seed), so re-running reproduces the same split.
"""

import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--n-test", type=int, default=16, help="held-out lines, shared across all 6 conditions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.input) as f:
        lines = [l for l in f if l.strip()]

    rng = random.Random(args.seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    test_idx = set(idx[: args.n_test])

    with open(args.train_out, "w") as f:
        f.writelines(l for i, l in enumerate(lines) if i not in test_idx)
    with open(args.test_out, "w") as f:
        f.writelines(l for i, l in enumerate(lines) if i in test_idx)

    print(f"{len(lines)} total -> {len(lines) - len(test_idx)} train, {len(test_idx)} held-out test")


if __name__ == "__main__":
    main()
