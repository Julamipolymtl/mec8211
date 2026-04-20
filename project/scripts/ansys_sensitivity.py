"""
ANSYS linear F-delta curves at multiple E values vs experimental means.

Shows the sensitivity of the ANSYS 2D linear model to Young's modulus,
motivating the E calibration step.  A vertical marker highlights the
calibrated E (back-calculated from L=60 mm experimental stiffness).

Reads:  data/simulation_ansys.csv  (model=linear rows)
        data/experimental.csv

Run with:  python scripts/ansys_sensitivity.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loaders import load_linear_sweep, load_experimental, load_calibrated_E

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

E_LIST = [6, 8, 10, 12, 14]
MARKERS = ["o-", "s-", "^-", "D-", "v-"]
COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]


TEST_CONDITIONS = [(60.0, d) for d in [3.0, 4.0, 5.0]] + \
                  [(40.0, d) for d in [3.0, 4.0, 5.0]]


if __name__ == "__main__":
    import csv as _csv
    sweep    = load_linear_sweep()
    exp_data = load_experimental()
    E_calib_mean, E_calib_std = load_calibrated_E()
    E_calib_MPa = E_calib_mean / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, L_mm in zip(axes, [60.0, 40.0]):
        deltas_exp = sorted(d for (L, d) in exp_data if L == L_mm)
        F_exp_means = [
            float(np.mean(list(exp_data[(L_mm, d)].values())))
            for d in deltas_exp
        ]

        for i, E in enumerate(E_LIST):
            deltas_sim = sorted(d for (L, d, e) in sweep if L == L_mm and e == float(E))
            F_sim = [sweep[(L_mm, d, float(E))] for d in deltas_sim]
            ax.plot(deltas_sim, F_sim, MARKERS[i], color=COLORS[i],
                    label=f"E = {E} MPa", markersize=4)

        ax.plot(deltas_exp, F_exp_means, "ks-", lw=2, markersize=7,
                label="Exp. mean", zorder=5)

        ax.set_xlabel("$\\delta$ [mm]")
        ax.set_ylabel("$F$ [N]")
        ax.set_title(f"L = {L_mm:.0f} mm  (L/d = {L_mm/5:.0f})")
        ax.legend(fontsize=8)
        ax.grid()
        ax.text(0.03, 0.97, f"E calib = {E_calib_MPa:.1f} MPa",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.suptitle("ANSYS 2D linear model: E sensitivity vs experimental",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS, "ansys_E_sensitivity.png")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")

    # --- Compute u_input per condition via dF/dE * sigma_E ---
    E_vals_Pa = sorted(set(e * 1e6 for (_, _, e) in sweep))
    u_rows = []
    for L_mm, delta_mm in TEST_CONDITIONS:
        F_at_E = [(e, sweep[(L_mm, delta_mm, e)]) for e in
                  sorted(set(e for (L, d, e) in sweep if L == L_mm and d == delta_mm))]
        if len(F_at_E) < 2:
            continue
        E_arr = np.array([e * 1e6 for e, _ in F_at_E])   # Pa
        F_arr = np.array([f for _, f in F_at_E])
        dFdE  = float(np.gradient(F_arr, E_arr)[np.argmin(np.abs(E_arr - E_calib_mean))])
        u_input = abs(dFdE) * E_calib_std
        u_rows.append({
            "L_span_mm": L_mm,
            "delta_mm":  delta_mm,
            "dF_dE":     round(dFdE, 10),
            "u_input_N": round(u_input, 6),
        })

    out_path = os.path.join(RESULTS, "ansys_u_input.csv")
    with open(out_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["L_span_mm", "delta_mm", "dF_dE", "u_input_N"])
        w.writeheader()
        w.writerows(u_rows)
    print(f"Saved {out_path}")

    # --- Sensitivity table ---
    print()
    print("=" * 72)
    print("ANSYS 2D -- input uncertainty from E sensitivity  (u_input = |dF/dE| * sigma_E)")
    print(f"  sigma_E = {E_calib_std/1e6:.4f} MPa  (calibrated from cantilever)")
    print("=" * 72)
    print(f"  {'cond':>10}  {'F_sim (N)':>10}  {'dF/dE (N/Pa)':>14}  "
          f"{'u_input (N)':>12}  {'u_input (%)':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*12}")
    for row in u_rows:
        L_mm     = row["L_span_mm"]
        d_mm     = row["delta_mm"]
        dFdE     = row["dF_dE"]
        u_inp    = row["u_input_N"]
        E_keys   = sorted(set(e for (L, d, e) in sweep if L == L_mm and d == d_mm))
        E_near   = min(E_keys, key=lambda e: abs(e - E_calib_MPa))
        F_nom    = sweep.get((L_mm, d_mm, E_near))
        pct      = u_inp / F_nom * 100.0 if F_nom else float("nan")
        cond     = f"L{L_mm:.0f}/d{d_mm:.0f}"
        print(f"  {cond:>10}  {F_nom:>10.5f}  {dFdE:>14.4e}  {u_inp:>12.5f}  {pct:>11.1f}%")
