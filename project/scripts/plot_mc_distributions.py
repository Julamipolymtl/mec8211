"""
Monte Carlo output CDF -- 1D EB beam model (MEC8211 methodology).

Follows the course MC procedure (Oberkampf & Roy, Fig. 13.15 / slide 14):
  1. Définir les PDF pour les variables d'entrée (d, L, delta, E)
  2. Générer un échantillon représentatif pour chaque PDF
  3. Calculer le résultat du modèle (SRQ = F) pour chaque groupe d'échantillons
  4. Tracer la CDF sur la SRQ désirée

One empirical CDF per (L_span, delta) condition, pooling all 6 specimens
(captures both measurement uncertainty within each specimen and
specimen-to-specimen diameter variability).

Experimental measurements shown as vertical markers for comparison.

Layout: two figures (one per span), each with 1 row × 3 columns (one per delta).
Saves to results/plot_mc_cdf_L{span}.png
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
# Material (nominal / reference)
# ---------------------------------------------------------------------------
C10 = 2.6643e5
C01 = 6.6007e5
E0  = 6.0 * (C10 + C01)   # linearised Young's modulus [Pa], nu=0.5

# ---------------------------------------------------------------------------
# Uncertain inputs
# ---------------------------------------------------------------------------
# Aleatory
SIGMA_L     = 0.25e-3    # 1-sigma span measurement [m]  (assumed, k=2 tolerance 0.5 mm)
SIGMA_DELTA = 0.005e-3   # 1-sigma applied displacement [m] (assumed encoder, k=2)

# Epistemic: Young's modulus unknown for this TPU batch.
# Uniform(E_MIN, E_MAX) = maximum-entropy representation of a known interval.
# Range from literature (Shore A ~70-80 soft TPU, quasi-static):
#   Covestro datasheets; MDPI Materials 2025; PMC12114912
E_MIN = 5.0e6    # Pa  (~Shore A 70 lower bound)
E_MAX = 15.0e6   # Pa  (~Shore A 80 upper bound)

N_MC     = 10_000
RNG_SEED = 42

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

def F_analytical(d, L, delta, E=E0):
    """Midspan reaction force [N]. Accepts numpy arrays."""
    return 48.0 * E * (np.pi * d**4 / 64.0) / L**3 * delta


def run_mc_aleatory(d_mean, sigma_d, L_nom, delta_nom, rng, E=E0):
    """
    MC over aleatory inputs only for a fixed Young's modulus E.
    Returns F_mc array of length N_MC.
    Aleatory inputs:
      d      ~ Normal(d_mean, sigma_d)       [caliper repeatability]
      L      ~ Normal(L_nom,  SIGMA_L)       [fixture tolerance]
      delta  ~ Normal(delta_nom, SIGMA_DELTA)[machine encoder]
    """
    d_s     = rng.normal(d_mean,    sigma_d,     N_MC)
    L_s     = rng.normal(L_nom,     SIGMA_L,     N_MC)
    delta_s = rng.normal(delta_nom, SIGMA_DELTA, N_MC)
    return F_analytical(d_s, L_s, delta_s, E)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def empirical_cdf(F_arr):
    """Return (x_sorted, p) for plotting an empirical CDF."""
    x = np.sort(F_arr)
    p = np.arange(1, len(x) + 1) / len(x)
    return x, p


def make_figure(L_mm, specimens, exp_groups, rng):
    """
    One figure per span.  Three subplots (one per delta).

    Epistemic/aleatory split following course methodology (Fig 2.16):
      - Aleatory inputs (d, L, delta): sampled stochastically → one CDF per E
      - Epistemic input E ∈ [E_MIN, E_MAX]: sweep to produce a *band* of CDFs
        * lower CDF bound: E = E_MAX  (high stiffness → high F → CDF shifted right)
        * upper CDF bound: E = E_MIN  (low stiffness  → low F  → CDF shifted left)
      - Shaded region between the two bounds = p-box (combined uncertainty)
      - Nominal CDF at E = E0 shown for reference

    Experimental measurements overlaid as vertical markers.
    """
    deltas = [3.0, 4.0, 5.0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    fig.suptitle(
        f"MC output CDF -- SRQ: midspan force $F$,  "
        f"$L_{{span}}$ = {L_mm:.0f} mm ($L/d$ ≈ {L_mm/5:.0f})\n"
        f"Aleatory: $d,\\,L,\\,\\delta$ (Normal)  ·  "
        f"Epistemic: $E \\in [{E_MIN/1e6:.0f},\\,{E_MAX/1e6:.0f}]$ MPa  ·  "
        f"$N_{{MC}}$ = {N_MC:,}/specimen, pooled over 6",
        fontsize=9.5,
    )

    L = L_mm * 1e-3

    for col, delta_mm in enumerate(deltas):
        delta = delta_mm * 1e-3
        ax = axes[col]

        # Pool all 6 specimens for each epistemic bound and nominal E
        def pool_mc(E_val):
            return np.concatenate([
                run_mc_aleatory(specimens[sid]["d_mean_m"],
                                specimens[sid]["sigma_d_m"],
                                L, delta, rng, E=E_val)
                for sid in range(1, 7)
            ])

        F_lo  = pool_mc(E_MIN)   # E_MIN → low F  → upper CDF bound
        F_nom = pool_mc(E0)      # nominal E
        F_hi  = pool_mc(E_MAX)   # E_MAX → high F → lower CDF bound

        x_lo,  p_lo  = empirical_cdf(F_lo)
        x_nom, p_nom = empirical_cdf(F_nom)
        x_hi,  p_hi  = empirical_cdf(F_hi)

        # --- P-box: shaded band between the two extreme CDFs ---
        # Interpolate both CDFs onto a common F axis for fill_betweenx
        F_common = np.linspace(min(x_lo[0], x_hi[0]),
                               max(x_lo[-1], x_hi[-1]), 600)
        p_lo_interp = np.interp(F_common, x_lo, p_lo, left=0.0, right=1.0)
        p_hi_interp = np.interp(F_common, x_hi, p_hi, left=0.0, right=1.0)

        ax.fill_betweenx(
            # x_lo gives upper CDF, x_hi gives lower CDF
            np.linspace(0, 1, 600),
            np.interp(np.linspace(0, 1, 600), p_lo, x_lo * 1e3,
                      left=x_lo[0]*1e3, right=x_lo[-1]*1e3),
            np.interp(np.linspace(0, 1, 600), p_hi, x_hi * 1e3,
                      left=x_hi[0]*1e3, right=x_hi[-1]*1e3),
            color="#0072B2", alpha=0.18,
            label=f"Epistemic band\n$E\\in[{E_MIN/1e6:.0f},{E_MAX/1e6:.0f}]$ MPa",
        )

        # --- Extreme CDFs (bounds of the band) ---
        ax.step(x_lo * 1e3, p_lo, color="#0072B2", lw=1.0, ls="--", alpha=0.7,
                label=f"$E_{{min}}$ = {E_MIN/1e6:.0f} MPa")
        ax.step(x_hi * 1e3, p_hi, color="#0072B2", lw=1.0, ls="-.", alpha=0.7,
                label=f"$E_{{max}}$ = {E_MAX/1e6:.0f} MPa")

        # --- Nominal CDF (E = E0) ---
        ax.step(x_nom * 1e3, p_nom, color="#D55E00", lw=1.8,
                label=f"$E_0$ = {E0/1e6:.1f} MPa")

        # --- Experimental measurements ---
        exp_key = (L_mm, delta_mm)
        F_exp   = np.array(exp_groups.get(exp_key, []))
        if len(F_exp):
            for i, fv in enumerate(F_exp):
                ax.axvline(fv * 1e3, color="#009E73", lw=0.9, ls=":",
                           alpha=0.7,
                           label="Exp. measurements" if i == 0 else None)
            ax.axvline(F_exp.mean() * 1e3, color="#009E73", lw=2.0, ls="-",
                       label=f"Exp. mean  ({F_exp.mean()*1e3:.1f} mN)")

        # --- Decoration ---
        ax.set_title(f"$\\delta$ = {delta_mm:.0f} mm", fontsize=10)
        ax.set_xlabel("$F_{sim}$  (mN)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Cumulative probability  $P(F \\leq x)$", fontsize=9)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    specimens  = load_specimens()
    exp_groups = load_experimental()
    rng        = np.random.default_rng(RNG_SEED)

    for L_mm in [60.0, 40.0]:
        fig  = make_figure(L_mm, specimens, exp_groups, rng)
        path = os.path.join(RESULTS, f"plot_mc_cdf_L{L_mm:.0f}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")
