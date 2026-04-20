"""
Validation plots: 1D EB FEM vs 2D ANSYS vs experimental.

Generates one figure per span (L=65mm, L=60mm, L=40mm) showing the
force-displacement curve for all three sources.

  - Experimental: individual specimen markers + mean +/- 1 std band
  - ANSYS 2D    : line over full simulation range
  - 1D EB       : line over full simulation range (nominal d=5mm)

Saves to results/plot_validation_L{span}.png
Run with:  python scripts/plot_validation.py
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beam import assemble_K, solve, solve_mr

DATA    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Material / geometry
# ---------------------------------------------------------------------------
C10  = 2.6643e5
C01  = 6.6007e5
MU0  = 2.0 * (C10 + C01)
E0   = 3.0 * MU0          # linearized E [Pa], nu=0.5
D_NOM = 5.0e-3             # nominal diameter [m]

# Epistemic bounds on E from TPU literature (Shore A ~70-80, quasi-static)
E_MIN = 5.0e6   # Pa
E_MAX = 15.0e6  # Pa

N_ELEM = 20


def compute_1d_force(L_span, delta, d=D_NOM, E=E0):
    I   = np.pi * d**4 / 64
    mid = N_ELEM // 2
    K   = assemble_K(N_ELEM, E, I, L_span)
    f   = np.zeros(2 * (N_ELEM + 1))
    _, R = solve(K, f, [0, 2*N_ELEM, 2*mid], [0.0, 0.0, delta])
    return R[2*mid]


def compute_1d_force_mr(L_span, delta, d=D_NOM):
    mid = N_ELEM // 2
    f   = np.zeros(2 * (N_ELEM + 1))
    _, R = solve_mr(N_ELEM, d, L_span, C10, C01, f,
                    [0, 2*N_ELEM, 2*mid], [0.0, 0.0, delta])
    return R[2*mid]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_simulation():
    rows = {}
    with open(os.path.join(DATA, "simulation.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            L = float(r["L_span_mm"])
            rows.setdefault(L, {})
            rows[L][float(r["delta_mm"])] = float(r["F_sim_N"])
    return rows


def load_experimental():
    groups = {}
    with open(os.path.join(DATA, "experimental.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            L = float(r["L_span_mm"])
            d = float(r["delta_mm"])
            groups.setdefault(L, {}).setdefault(d, []).append(float(r["F_exp_N"]))
    return groups


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(L_mm, sim, exp):
    L = L_mm * 1e-3
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # --- ANSYS 2D ---
    sim_L = sim.get(L_mm, {})
    if sim_L:
        sim_deltas = sorted(sim_L)
        sim_F      = [sim_L[d] for d in sim_deltas]
        ax.plot([0] + sim_deltas, [0] + sim_F,
                color="#0072B2", lw=1.8, label="ANSYS 2D (fine mesh)")

    # --- 1D EB ---
    if sim_L:
        d_plot = sorted(sim_L)
    else:
        d_plot = [1.0, 2.0, 3.0, 4.0, 5.0]
    fem_F     = [compute_1d_force(L, dd * 1e-3)           for dd in d_plot]
    fem_F_low = [compute_1d_force(L, dd * 1e-3, E=E_MIN)  for dd in d_plot]
    fem_F_hi  = [compute_1d_force(L, dd * 1e-3, E=E_MAX)  for dd in d_plot]

    ax.fill_between([0] + d_plot, [0] + fem_F_low, [0] + fem_F_hi,
                    color="#D55E00", alpha=0.15,
                    label=f"1D EB: $E$ ∈ [{E_MIN/1e6:.0f}, {E_MAX/1e6:.0f}] MPa")
    ax.plot([0] + d_plot, [0] + fem_F,
            color="#D55E00", lw=1.8, ls="--", label=f"1D EB FEM ($E_0$ = {E0/1e6:.1f} MPa)")

    # --- 1D EB + Mooney-Rivlin ---
    mr_F = [compute_1d_force_mr(L, dd * 1e-3) for dd in d_plot]
    ax.plot([0] + d_plot, [0] + mr_F,
            color="#CC79A7", lw=1.8, ls="-.", label="1D EB FEM (Mooney-Rivlin)")

    # --- Experimental: scatter + mean +/- 1 std band ---
    exp_L = exp.get(L_mm, {})
    test_deltas = sorted(exp_L)
    F_means = np.array([np.mean(exp_L[d]) for d in test_deltas])
    F_stds  = np.array([np.std(exp_L[d], ddof=1) for d in test_deltas])
    td = np.array(test_deltas)

    # shaded band
    ax.fill_between([0] + list(td),
                    [0] + list(F_means - F_stds),
                    [0] + list(F_means + F_stds),
                    color="#009E73", alpha=0.20, lw=0)

    # individual specimen points
    with open(os.path.join(DATA, "experimental.csv"), newline="") as fh:
        all_rows = list(csv.DictReader(fh))
    for row in all_rows:
        if abs(float(row["L_span_mm"]) - L_mm) < 0.1:
            ax.scatter(float(row["delta_mm"]), float(row["F_exp_N"]),
                       s=18, color="#009E73", alpha=0.55, zorder=3,
                       linewidths=0)

    # mean line
    ax.plot([0] + list(td), [0] + list(F_means),
            color="#009E73", lw=1.8, marker="o", ms=5,
            label=r"Exp. mean $\pm\,1\sigma$  (n=6)")

    # --- Decoration ---
    ax.set_xlabel("Midspan displacement  $\\delta$  (mm)", fontsize=11)
    ax.set_ylabel("Midspan reaction force  $F$  (N)", fontsize=11)
    ax.set_title(
        f"Three-point bending -- $L_{{span}}$ = {L_mm:.0f} mm "
        f"($L/d$ = {L_mm/5.0:.0f})",
        fontsize=11,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sim = load_simulation()
    exp = load_experimental()

    for L_mm in [65.0, 60.0, 40.0]:
        fig  = make_figure(L_mm, sim, exp)
        path = os.path.join(RESULTS, f"plot_validation_L{L_mm:.0f}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved {path}")
