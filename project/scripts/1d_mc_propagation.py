"""
Monte Carlo input uncertainty propagation for the 1D EB beam model.

Propagates uncertainties in d, L, delta, and E (calibrated from cantilever
experiment) through the analytical 3-pt bending formula to estimate u_input
at 95% coverage for each (L_span, delta) test condition.

Writes:  results/1d_u_input.csv
         columns: L_span_mm, delta_mm, F_nom_N, u_input_N, F_mc_lo_N, F_mc_hi_N
         results/mc_distributions_L60.png
         results/mc_distributions_L40.png

Run with:  python scripts/1d_mc_propagation.py
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as _norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beam import F_analytical_3pt
from data_loaders import load_specimens, load_experimental, load_calibrated_E

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

TEST_CONDITIONS = [(60.0, d) for d in [3.0, 4.0, 5.0]] + \
                  [(40.0, d) for d in [3.0, 4.0, 5.0]]

SIGMA_L     = 0.25e-3
SIGMA_DELTA = 0.005e-3
N_MC        = 50_000
RNG_SEED    = 42
N_SPEC      = 6

SPEC_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def run_mc(d_mean, sigma_d, L_nom, delta_nom, E_mean, E_std, rng):
    d_s   = rng.normal(d_mean,    sigma_d,     N_MC)
    L_s   = rng.normal(L_nom,     SIGMA_L,     N_MC)
    dlt_s = rng.normal(delta_nom, SIGMA_DELTA, N_MC)
    E_s   = rng.normal(E_mean,    E_std,       N_MC)
    return F_analytical_3pt(d_s, L_s, dlt_s, E_s)


def plot_mc_distributions(specimens, exp_data, E_calib_mean, E_calib_std, rng):
    for L_mm in [60.0, 40.0]:
        L      = L_mm * 1e-3
        deltas = sorted(d for (Lm, d) in exp_data if Lm == L_mm)
        fig, axes = plt.subplots(2, len(deltas), figsize=(12, 7))
        fig.suptitle(
            f"MC input uncertainty -- 1D EB model,  $L_{{span}}$ = {L_mm:.0f} mm "
            f"($L/d$ = {L_mm/5:.0f})",
            fontsize=11,
        )

        for col, delta_mm in enumerate(deltas):
            delta   = delta_mm * 1e-3
            ax_pdf  = axes[0, col]
            ax_cdf  = axes[1, col]
            exp_key = (L_mm, delta_mm)
            F_exp_arr  = np.array(list(exp_data.get(exp_key, {}).values()))
            F_exp_mean = float(F_exp_arr.mean()) if len(F_exp_arr) else None

            for sid in range(1, N_SPEC + 1):
                s     = specimens[sid]
                d_s   = rng.normal(s["d_mean_m"],   s["sigma_d_m"], N_MC)
                L_s   = rng.normal(L,               SIGMA_L,        N_MC)
                dlt_s = rng.normal(delta,            SIGMA_DELTA,    N_MC)
                E_s   = rng.normal(E_calib_mean,     E_calib_std,    N_MC)
                F_mc  = F_analytical_3pt(d_s, L_s, dlt_s, E_s)

                mu, sigma = float(F_mc.mean()), float(F_mc.std(ddof=1))
                color = SPEC_COLORS[sid - 1]
                x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)

                ax_pdf.plot(x * 1e3, _norm.pdf(x, mu, sigma) / 1e3,
                            color=color, lw=1.2, alpha=0.8, label=f"sp.{sid}")
                x_cdf = np.sort(F_mc)
                y_cdf = np.arange(1, N_MC + 1) / N_MC
                ax_cdf.plot(x_cdf * 1e3, y_cdf, color=color, lw=0.8, alpha=0.6)
                ax_cdf.plot(x * 1e3, _norm.cdf(x, mu, sigma),
                            color=color, lw=1.4, ls="--", alpha=0.9)

            if F_exp_mean is not None:
                for ax in (ax_pdf, ax_cdf):
                    ax.axvline(F_exp_mean * 1e3, color="black", lw=1.4,
                               ls=":", label="Exp. mean" if col == 0 else None)

            ax_pdf.set_title(f"$\\delta$ = {delta_mm:.0f} mm", fontsize=10)
            ax_pdf.set_xlabel("$F_{sim}$ [mN]", fontsize=9)
            ax_pdf.set_ylabel("PDF", fontsize=9)
            ax_pdf.tick_params(labelsize=8)
            ax_pdf.grid(True, lw=0.3, alpha=0.4)
            ax_cdf.set_xlabel("$F_{sim}$ [mN]", fontsize=9)
            ax_cdf.set_ylabel("CDF", fontsize=9)
            ax_cdf.set_ylim(0, 1)
            ax_cdf.tick_params(labelsize=8)
            ax_cdf.grid(True, lw=0.3, alpha=0.4)

            if col == len(deltas) - 1:
                ax_pdf.legend(fontsize=7, loc="upper right",
                              title="specimen", title_fontsize=7)

        fig.tight_layout()
        path = os.path.join(RESULTS, f"mc_distributions_L{L_mm:.0f}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    specimens = load_specimens()
    exp_data  = load_experimental()
    E_mean, E_std = load_calibrated_E()
    rng = np.random.default_rng(RNG_SEED)

    print(f"1D MC input uncertainty propagation  (N={N_MC:,}, seed={RNG_SEED})")
    print(f"  E ~ Normal({E_mean/1e6:.3f}, {E_std/1e6:.3f}) MPa  (calibrated from cantilever)")
    print()
    print(f"  {'cond':>10}  {'F_nom':>8}  {'u_input':>9}  {'F_lo':>8}  {'F_hi':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")

    rows = []
    for L_mm, delta_mm in TEST_CONDITIONS:
        key = (L_mm, delta_mm)
        if key not in exp_data:
            continue
        L     = L_mm * 1e-3
        delta = delta_mm * 1e-3
        d_mean  = float(np.mean([specimens[s]["d_mean_m"] for s in exp_data[key]]))
        sigma_d = float(np.mean([specimens[s]["sigma_d_m"] for s in exp_data[key]]))

        F_mc  = run_mc(d_mean, sigma_d, L, delta, E_mean, E_std, rng)
        F_nom = float(F_analytical_3pt(d_mean, L, delta, E_mean))
        F_lo  = float(np.percentile(F_mc, 2.5))
        F_hi  = float(np.percentile(F_mc, 97.5))
        u_inp = (F_hi - F_lo) / 2.0

        cond = f"L{L_mm:.0f}/d{delta_mm:.0f}"
        print(f"  {cond:>10}  {F_nom:>8.4f}  {u_inp:>9.5f}  {F_lo:>8.4f}  {F_hi:>8.4f}")

        rows.append({
            "L_span_mm":  L_mm,
            "delta_mm":   delta_mm,
            "F_nom_N":    round(F_nom, 6),
            "u_input_N":  round(u_inp, 6),
            "F_mc_lo_N":  round(F_lo,  6),
            "F_mc_hi_N":  round(F_hi,  6),
        })

    out_path = os.path.join(RESULTS, "1d_u_input.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_path}")

    print("\nPlotting MC output distributions ...")
    rng2 = np.random.default_rng(RNG_SEED + 1)
    plot_mc_distributions(specimens, exp_data, E_mean, E_std, rng2)
