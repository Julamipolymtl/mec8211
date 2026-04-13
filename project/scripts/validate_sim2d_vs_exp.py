"""
Validation: 2D ANSYS simulation vs experimental measurements.

For each test condition (L_span, delta):
  - Collects F_exp across all 6 specimens
  - Looks up the converged 2D simulation force
  - Reports mean, standard deviation, and relative error

Reads:  data/experimental.csv   (from extract_data.py)
        data/simulation.csv

Run with:  python scripts/validate_sim2d_vs_exp.py
"""

import csv
import os
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def read_csv(filename):
    path = os.path.join(DATA, filename)
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) if v.strip() else None
                         for k, v in row.items()})
    return rows


def group_exp_by_condition(exp_rows):
    """
    Returns dict: {(L_span_mm, delta_mm): np.array of F_exp_N values}
    """
    groups = {}
    for row in exp_rows:
        key = (row["L_span_mm"], row["delta_mm"])
        groups.setdefault(key, []).append(row["F_exp_N"])
    return {k: np.array(v) for k, v in groups.items()}


def build_sim_lookup(sim_rows):
    """
    Returns dict: {(L_span_mm, delta_mm): F_sim_N}
    """
    return {(r["L_span_mm"], r["delta_mm"]): r["F_sim_N"] for r in sim_rows}


if __name__ == "__main__":
    exp_rows  = read_csv("experimental.csv")
    sim_rows  = read_csv("simulation.csv")
    exp_groups = group_exp_by_condition(exp_rows)
    sim_lookup = build_sim_lookup(sim_rows)

    conditions = sorted(exp_groups.keys())
    n_cond     = len(conditions)
    n_matched  = sum(1 for k in conditions if k in sim_lookup)

    print("=" * 70)
    print("Validation: 2D ANSYS simulation vs experimental")
    print("=" * 70)
    print(f"  Conditions with exp. data : {n_cond}")
    print(f"  Matched to sim. data      : {n_matched} / {n_cond}")

    print(f"\n  {'L (mm)':>6}  {'d (mm)':>6}  {'F_exp mean':>12}  "
          f"{'F_exp std':>10}  {'F_sim':>10}  {'err (%)':>9}  {'n':>3}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*3}")

    abs_errors = []
    for (L, delta) in conditions:
        F_arr  = exp_groups[(L, delta)]
        F_mean = np.mean(F_arr)
        F_std  = np.std(F_arr, ddof=1)
        F_sim  = sim_lookup.get((L, delta))

        if F_sim is None:
            err_str = "   no sim"
        else:
            rel_err = (F_sim - F_mean) / F_mean * 100.0
            abs_errors.append(abs(rel_err))
            err_str = f"{rel_err:+9.2f}%"

        F_sim_str = f"{F_sim:10.5f}" if F_sim is not None else "       ---"
        print(f"  {L:>6.0f}  {delta:>6.1f}  {F_mean:>12.5f}  "
              f"{F_std:>10.5f}  {F_sim_str}  {err_str}  {len(F_arr):>3}")

    if abs_errors:
        print()
        print(f"  Mean |err| : {np.mean(abs_errors):.1f}%")
        print(f"  Max  |err| : {np.max(abs_errors):.1f}%")
