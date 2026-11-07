"""HDS pattern -> TouchSim -> spike counts -> digit similarity.

Usage: python scripts/run_pipeline.py [--density 1.0] [--out out/results.npz]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hds_touchsim import (HDSSpec, build_stimulus, counts_by_class, digit_tensor, fit_to_region,
                          hand_scale, load_touchsim, make_population, px_per_mm, region_mask_fn,
                          respond, similarity_matrix, spike_counts)
from hds_touchsim.adapter import REAL_HAND_LENGTH_MM, ts_hand_length_mm, ts_px_per_mm


def run(ts, density=1.0, step_px=4, search_box=(40, 400, 100, 520)):
    spec = HDSSpec()
    ppm = px_per_mm(ts)
    print(f"TouchSim hand: {ts_hand_length_mm(ts):.1f} mm, {ts_px_per_mm(ts):.3f} px/mm")
    print(f"HAND_SCALE ({REAL_HAND_LENGTH_MM:.0f} mm) = {hand_scale(ts):.4f} -> {ppm:.3f} px per real mm")

    best, n, _ = fit_to_region(spec, region_mask_fn(ts), search_box, pxl_per_mm=ppm, step_px=step_px)
    center = (float(best[0]), float(best[1]))
    print(f"HDS center (px): {center} -> {n}/15 pins inside the 6 palm regions")

    pop = make_population(ts, spec, center, density_multiplier=density)
    print(pop)

    all_counts, per_class = {}, {}
    for d in range(10):
        resp = respond(ts, build_stimulus(ts, digit_tensor(d), spec, center), pop)
        all_counts[d] = spike_counts(resp)
        per_class[d] = counts_by_class(all_counts[d], pop)
        print(f"digit {d}: {int(digit_tensor(d).sum()):2d} pins, {all_counts[d].sum():6d} spikes, {per_class[d]}")
    return center, pop, all_counts, similarity_matrix(all_counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--out", default="out/results.npz")
    args = ap.parse_args()

    center, pop, all_counts, S = run(load_touchsim(), density=args.density)
    np.set_printoptions(precision=2, linewidth=140)
    print("Pearson r between digits:\n", S)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, center=center, similarity=S, affclass=np.array(pop.affclass),
             **{f"counts_{d}": c for d, c in all_counts.items()})
    print("saved", args.out)


if __name__ == "__main__":
    main()
