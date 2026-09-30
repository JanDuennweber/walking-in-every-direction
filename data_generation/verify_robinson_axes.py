#!/usr/bin/env python3
"""
Verifies the assumption behind generate_headings.py --mode robinson:
every triangle of a wheel-seeded Robinson tiling has its axis of symmetry
(apex -> midpoint of the base) in one of the ten directions k*36 degrees,
and all ten directions occur equally often. Sampling a triangle uniformly
from the tiling is then equivalent to drawing one of these ten directions
with equal probability, which is what generate_headings.py does.

Builds the tiling with the standard Robinson subdivision (golden triangle L
-> L + S, golden gnomon S -> S + S + L) from the ten-triangle wheel seed
and prints, per generation, the axis directions found and their counts
(in total and separately for L and S).

Scope: the complete wheel-seeded patch. A patch clipped to an irregular
footprint (paper Figures 5/6) is not exactly balanced in general.

Usage: python3 verify_robinson_axes.py [--generations 8]
"""
import argparse
import cmath
import math
from collections import Counter
phi = (1 + 5 ** 0.5) / 2

def wheel():
    tris = []
    for i in range(10):
        B = cmath.rect(1, (2 * i - 1) * math.pi / 10)
        C = cmath.rect(1, (2 * i + 1) * math.pi / 10)
        if i % 2 == 0:
            B, C = C, B          # mirror every second triangle (matching rule)
        tris.append((0, 0j, B, C))  # 0 = golden triangle L, apex A
    return tris

def subdivide(tris):
    out = []
    for color, A, B, C in tris:
        if color == 0:           # L -> L + S
            P = A + (B - A) / phi
            out += [(0, C, P, B), (1, P, C, A)]
        else:                    # S -> S + S + L
            Q = B + (A - B) / phi
            R = B + (C - B) / phi
            out += [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
    return out

def axis_deg(A, B, C):       # apex -> midpoint of base
    return round(math.degrees(cmath.phase((B + C) / 2 - A))) % 360

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    args = ap.parse_args()

    tris = wheel()
    balanced = True
    for gen in range(args.generations + 1):
        if gen:
            tris = subdivide(tris)
        c = Counter(axis_deg(A, B, C) for _, A, B, C in tris)
        cL = Counter(axis_deg(A, B, C) for col, A, B, C in tris if col == 0)
        cS = Counter(axis_deg(A, B, C) for col, A, B, C in tris if col == 1)
        dirs = sorted(c)
        ok = dirs == [36 * k for k in range(10)] and len(set(c.values())) == 1
        balanced &= ok
        print(f"gen {gen}: {len(tris)} triangles, directions {dirs}")
        print(f"   counts {[c[d] for d in dirs]}  (L {[cL[d] for d in dirs]}, S {[cS[d] for d in dirs]})"
              f"  -> {'balanced' if ok else 'NOT balanced'}")
    print("RESULT:", "all generations use exactly the ten directions k*36 deg, equally often"
          if balanced else "assumption does NOT hold")


if __name__ == "__main__":
    main()
