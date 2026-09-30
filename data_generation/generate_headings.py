#!/usr/bin/env python3
"""
Offline heading generator for the two data-generation conditions that do
NOT exist anywhere in Geyer's transferred Unity project (verified: grepped
SensFloorSimulation/Assets/Scripts/*.cs for "robinson|golden|aperiodic|
penrose", zero matches -- the only heading sampler implemented there is
the narrow-range [-60,60] one, Simulation.cs's generateRandomWalkers()):

  - naive_360: a plain uniform float heading over [0, 360) -- the control
    condition for row 3 (tests the paper's own claim that widening the
    range alone doesn't remove the bias).
  - robinson: a heading sampled from the Star-tiling scaffold's ten
    orientation classes {0,36,...,324} -- for row 4/6. A triangle's axis
    of symmetry points from its apex to its base, so it gives a direction,
    not just a line: the ten L triangles of the wheel seed point in ten
    different multiples of 36 degrees over the full circle. (An earlier
    version used only the five classes {0,...,144} taken mod 180, which
    as walking headings would all point into one half-plane -- a new
    directional bias instead of the removal of one.)

The robinson mode draws each heading with equal probability from the ten
directions k*36 degrees. This is exactly equivalent to sampling a triangle
uniformly from a wheel-seeded Robinson/Star tiling and using its axis of
symmetry: every triangle's axis points in one of these ten directions, and
all ten occur equally often (verify_robinson_axes.py checks this for up to
eight subdivision steps). The tiling's spatial structure is not needed:
the scaffold is only consulted once per simulated pedestrian for a single
heading angle (Subsection 3.2 of the paper).

Usage:
    python3 generate_headings.py --mode naive_360 --n 2000 --out naive_360_headings.csv
    python3 generate_headings.py --mode robinson --n 2000 --out robinson_headings.csv --crossover 60

Output: a one-column CSV of headings in degrees, one per simulated
pedestrian, read by the patched Simulation.cs (unity/apply_unity_patch.py).
"""

import argparse
import csv

import numpy as np

ORIENTATION_CLASSES = [36 * k for k in range(10)]  # degrees, full circle


def quota_allocate(n, classes=ORIENTATION_CLASSES, seed=0):
    k = len(classes)
    base, remainder = divmod(n, k)
    counts = {c: base for c in classes}
    rng = np.random.default_rng(seed)
    bonus_classes = rng.choice(classes, size=remainder, replace=False) if remainder else []
    for c in bonus_classes:
        counts[c] += 1
    labels = [c for c in classes for _ in range(counts[c])]
    rng.shuffle(labels)
    return labels


def sampled_allocate(n, classes=ORIENTATION_CLASSES, seed=0):
    rng = np.random.default_rng(seed)
    return list(rng.choice(classes, size=n, replace=True))


def robinson_headings(n, seed, crossover):
    if n < crossover:
        return quota_allocate(n, seed=seed)
    return sampled_allocate(n, seed=seed)


def naive_360_headings(n, seed):
    rng = np.random.default_rng(seed)
    return list(rng.uniform(0.0, 360.0, size=n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive_360", "robinson"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crossover", type=int, default=60,
                     help="robinson mode only: below this N, use exact quota balance; "
                          "at/above it, use sampled balance.")
    args = ap.parse_args()

    if args.mode == "naive_360":
        headings = naive_360_headings(args.n, args.seed)
    else:
        headings = robinson_headings(args.n, args.seed, args.crossover)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["heading_deg"])
        for h in headings:
            w.writerow([f"{h:.4f}"])
    print(f"wrote {len(headings)} headings ({args.mode}) to {args.out}")


if __name__ == "__main__":
    main()
