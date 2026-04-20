"""
Plot displacement and internal force fields along the beam for both test cases.

Generates one figure per case (cantilever and three-point bending), each with
three panels: transverse displacement v(x), bending moment M(x), shear V(x).

Saves to results/beam_fields_cantilever.png and results/beam_fields_3pt.png

Run with:  python scripts/plot_beam_fields.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beam import (
    assemble_K,
    assemble_distributed_load,
    apply_point_load,
    solve,
    compute_internal_forces,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

N_PTS = 20   # evaluation points per element for smooth curves


# --- Case definitions ---

def setup_cantilever():
    n     = 20
    L     = 0.200        # m
    E     = 10e6         # Pa
    d     = 0.010        # m
    mass  = 0.050        # kg
    m_tip = 0.020        # kg  (mass hung at tip)

    I     = np.pi * d**4 / 64
    w     = -mass * 9.81 / L      # gravity = -y
    F_tip = -m_tip * 9.81         # tip point load [N]  (downward, -y)

    K     = assemble_K(n, E, I, L)
    f_ext = assemble_distributed_load(n, L, w)
    apply_point_load(f_ext, x=L, L=L, n=n, force=F_tip)
    u, R  = solve(K, f_ext, [0, 1], [0.0, 0.0])

    return dict(
        label="Cantilever",
        n=n, L=L, E=E, I=I, u=u, R=R,
        load_cell=None,
    )


def setup_three_pt():
    n     = 20           # must be even
    L     = 0.160        # m
    E     = 10e6         # Pa
    d     = 0.010        # m
    mass  = 0.050        # kg
    delta = -5e-3        # m  (downward, -y)

    I       = np.pi * d**4 / 64
    w       = -mass * 9.81 / L    # gravity = -y
    mid     = n // 2

    K     = assemble_K(n, E, I, L)
    f_ext = assemble_distributed_load(n, L, w)
    u, R  = solve(K, f_ext, [0, 2*n, 2*mid], [0.0, 0.0, delta])

    F_mid = R[2*mid]
    return dict(
        label="Three-point bending",
        n=n, L=L, E=E, I=I, u=u, R=R,
        load_cell=(L / 2, F_mid),
    )


# --- Plotting ---

def make_figure(case: dict) -> plt.Figure:
    n, L, E, I = case["n"], case["L"], case["E"], case["I"]
    u = case["u"]

    x_nodes = np.linspace(0, L, n + 1)
    v_nodes = u[0::2] * 1e3          # convert to mm

    x_cont, M_cont, V_cont = compute_internal_forces(u, E, I, L, n, n_pts=N_PTS)
    x_mm = x_nodes * 1e3
    xc_mm = x_cont * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    fig.suptitle(case["label"], fontsize=12, fontweight="bold")

    # --- Displacement ---
    ax = axes[0]
    ax.plot(x_mm, v_nodes, color="#0072B2", lw=1.8, marker="o", ms=3)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("$v(x)$  (mm)", fontsize=10)
    ax.grid(True, lw=0.4, alpha=0.4)

    # --- Bending moment ---
    ax = axes[1]
    ax.plot(xc_mm, M_cont, color="#D55E00", lw=1.8)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.fill_between(xc_mm, 0, M_cont, alpha=0.15, color="#D55E00")
    ax.set_ylabel("$M(x)$  (N*m)", fontsize=10)
    ax.grid(True, lw=0.4, alpha=0.4)

    # --- Shear force + load-cell reaction ---
    ax = axes[2]
    ax.step(xc_mm, V_cont, color="#009E73", lw=1.8, where="post")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.fill_between(xc_mm, 0, V_cont, step="post", alpha=0.15, color="#009E73")

    if case["load_cell"] is not None:
        x_lc_m, F_lc = case["load_cell"]
        x_lc_mm = x_lc_m * 1e3
        ax.annotate(
            "",
            xy=(x_lc_mm, F_lc),
            xytext=(x_lc_mm, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.8),
        )
        ax.scatter([x_lc_mm], [F_lc], color="#C0392B", zorder=5, s=30)
        ax.annotate(
            f"load cell\n{F_lc:.4f} N",
            xy=(x_lc_mm, F_lc),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8, color="#C0392B", va="center",
        )

    ax.set_ylabel("$V(x)$  (N)", fontsize=10)
    ax.set_xlabel("$x$  (mm)", fontsize=10)
    ax.grid(True, lw=0.4, alpha=0.4)

    fig.tight_layout()
    return fig


# --- Main ---

if __name__ == "__main__":
    cases = [
        ("beam_fields_cantilever.png", setup_cantilever()),
        ("beam_fields_3pt.png",        setup_three_pt()),
    ]

    for filename, case in cases:
        fig  = make_figure(case)
        path = os.path.join(RESULTS, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")
