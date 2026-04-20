"""
Plot PDF and CDF of the Monte Carlo output distribution for each test condition.

For each (L_span, delta) condition and each specimen, shows:
  - Top row: histogram + fitted normal PDF
  - Bottom row: empirical CDF + fitted normal CDF

Also marks the experimental mean with a vertical line so the model-experiment
gap is visible relative to the MC spread.

Saves to results/plot_mc_L{span}.png
Run with:  python scripts/plot_mc_distributions.py
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
C10 = 2.6643e5
C01 = 6.6007e5
E0  = 6.0 * (C10 + C01)

SIGMA_L     = 0.25e-3
SIGMA_DELTA = 0.005e-3
N_MC        = 10_000
RNG_SEED    = 42

SPEC_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_specimens():
    out = {}
    with open(os.path.join(DATA, "specimens.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            sid = int(r["specimen_id"])
            d_meas_mm = [float(r["d1_mm"]), float(r["d2_mm"]), float(r["d3_mm"])]
            out[sid] = {
                "d_mean_m":  float(r["d_mean_mm"]) * 1e-3,
                "sigma_d_m": np.std(d_meas_mm, ddof=1) * 1e-3,
            }
    return out


def load_experimental():
    groups = {}
    with open(os.path.join(DATA, "experimental.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            key = (float(r["L_span_mm"]), float(r["delta_mm"]))
            groups.setdefault(key, []).append(float(r["F_exp_N"]))
    return groups


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def F_analytical(d, L, delta):
    return 48.0 * E0 * (np.pi * d**4 / 64.0) / L**3 * delta


def run_mc(d_mean, sigma_d, L_nom, delta_nom, rng):
    d_s     = rng.normal(d_mean,    sigma_d,     N_MC)
    L_s     = rng.normal(L_nom,     SIGMA_L,     N_MC)
    delta_s = rng.normal(delta_nom, SIGMA_DELTA, N_MC)
    return F_analytical(d_s, L_s, delta_s)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(L_mm, specimens, exp_groups, rng):
    deltas = [3.0, 4.0, 5.0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(
        f"MC input uncertainty -- 1D EB model,  $L_{{span}}$ = {L_mm:.0f} mm "
        f"($L/d$ = {L_mm/5:.0f})",
        fontsize=11,
    )

    L = L_mm * 1e-3

    for col, delta_mm in enumerate(deltas):
        delta = delta_mm * 1e-3
        ax_pdf = axes[0, col]
        ax_cdf = axes[1, col]

        # experimental mean for this condition
        exp_key  = (L_mm, delta_mm)
        F_exp_arr = np.array(exp_groups.get(exp_key, []))
        F_exp_mean = F_exp_arr.mean() if len(F_exp_arr) else None

        all_F = []

        for sid in range(1, 7):
            s    = specimens[sid]
            F_mc = run_mc(s["d_mean_m"], s["sigma_d_m"], L, delta, rng)
            all_F.append(F_mc)
            mu, sigma = F_mc.mean(), F_mc.std(ddof=1)
            color = SPEC_COLORS[sid - 1]

            # --- PDF ---
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
            ax_pdf.plot(x * 1e3, stats.norm.pdf(x, mu, sigma) / 1e3,
                        color=color, lw=1.2, alpha=0.8,
                        label=f"sp.{sid}")

            # --- CDF ---
            x_cdf = np.sort(F_mc)
            y_cdf = np.arange(1, N_MC + 1) / N_MC
            ax_cdf.plot(x_cdf * 1e3, y_cdf,
                        color=color, lw=0.8, alpha=0.6)
            ax_cdf.plot(x * 1e3, stats.norm.cdf(x, mu, sigma),
                        color=color, lw=1.4, ls="--", alpha=0.9)

        # experimental mean vertical line
        if F_exp_mean is not None:
            for ax in (ax_pdf, ax_cdf):
                ax.axvline(F_exp_mean * 1e3, color="black", lw=1.4,
                           ls=":", label="Exp. mean" if col == 0 else None)

        # axes decoration
        ax_pdf.set_title(f"$\\delta$ = {delta_mm:.0f} mm", fontsize=10)
        ax_pdf.set_xlabel("$F_{sim}$ (mN)", fontsize=9)
        ax_pdf.set_ylabel("PDF", fontsize=9)
        ax_pdf.tick_params(labelsize=8)
        ax_pdf.grid(True, lw=0.3, alpha=0.4)

        ax_cdf.set_xlabel("$F_{sim}$ (mN)", fontsize=9)
        ax_cdf.set_ylabel("CDF", fontsize=9)
        ax_cdf.set_ylim(0, 1)
        ax_cdf.tick_params(labelsize=8)
        ax_cdf.grid(True, lw=0.3, alpha=0.4)

        if col == 2:
            ax_pdf.legend(fontsize=7, loc="upper right",
                          title="specimen", title_fontsize=7)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    specimens  = load_specimens()
    exp_groups = load_experimental()
    rng        = np.random.default_rng(RNG_SEED)

    for L_mm in [60.0, 40.0]:
        fig  = make_figure(L_mm, specimens, exp_groups, rng)
        path = os.path.join(RESULTS, f"plot_mc_L{L_mm:.0f}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")
