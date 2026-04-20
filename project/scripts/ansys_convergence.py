"""
ANSYS mesh convergence analysis.

Reads convergence rows from data/simulation_ansys.csv (nonlinear rows where
h_mm varies for a fixed L_span_mm and delta_mm) and produces two figures:

  Figure 1 - Total view   : relative error vs h for all mesh sizes (log-log).
  Figure 2 - Asymptotic   : Richardson extrapolation + GCI for the fine-mesh
                             subset (h <= H_MAX).

Run with:  python scripts/ansys_convergence.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

DATA    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# --- Parameters ---
L_REF      = 40.0   # loading span for convergence study [mm]
D_REF      = 2.0    # imposed displacement [mm]
H_MAX      = 1.0    # asymptotic analysis: only rows with h_mm <= H_MAX
P_EXPECTED = 2      # expected order for linear triangular elements

# --- Load convergence rows ---
rows = []
with open(os.path.join(DATA, "simulation_ansys.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        if (r["model"] == "nonlinear"
                and float(r["L_span_mm"]) == L_REF
                and float(r["delta_mm"]) == D_REF):
            rows.append((float(r["h_mm"]), float(r["F_sim_N"])))

rows.sort(key=lambda x: x[0])
data     = np.array(rows)
h_values = data[:, 0]
forces   = data[:, 1]
F_ref    = forces[0]

err_all  = np.abs((forces[1:] - F_ref) / F_ref)
h_all    = h_values[1:]


# --- Figure 1: total view ---

def plot_total(h_plot, err, h_ref):
    pente, intercept = np.polyfit(np.log(h_plot), np.log(err), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(h_plot, err, "ro-", markersize=8, linewidth=2,
              label="Erreur relative")
    ax.loglog(h_plot, np.exp(intercept) * h_plot**pente, "k--", alpha=0.5,
              label=f"Pente (ordre p) = {pente:.2f}")

    for h, e in zip(h_plot, err):
        ax.annotate(f"{e*100:.1f}%", (h, e),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    ax.grid(True, which="both", ls="-", alpha=0.7)
    ax.set_xlabel("Taille du maillage h (mm)")
    ax.set_ylabel("Erreur relative")
    ax.set_title(
        f"Convergence de l'erreur - vue complete\n"
        f"(L={L_REF:.0f} mm, delta={D_REF:.0f} mm, "
        f"reference h={h_ref:.4f} mm)"
    )
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS, "ansys_conv_total.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")
    print(f"Ordre de convergence observe (p) : {pente:.2f}")


# --- Figure 2: asymptotic range (Richardson / GCI) ---

def plot_asymptotic(h_values_all, forces_all, h_max, p_expected):
    mask     = h_values_all <= h_max
    h_sub    = h_values_all[mask]
    f_sub    = forces_all[mask]
    F_ref_s  = f_sub[0]
    h_plot   = h_sub[1:]
    err      = np.abs((f_sub[1:] - F_ref_s) / F_ref_s)

    h1, h2, h3 = h_sub[0], h_sub[1], h_sub[2]
    f1, f2, f3 = f_sub[0], f_sub[1], f_sub[2]
    r       = h2 / h1
    p_rich  = np.log((f3 - f2) / (f2 - f1)) / np.log(r)
    f_rich  = f1 + (f1 - f2) / (r**p_rich - 1)

    asympt = abs(p_expected - p_rich) / p_rich
    if asympt <= 0.1:
        Fs    = 1.25
        p_gci = 2
    else:
        Fs    = 3
        p_gci = min(max(0.5, p_rich), p_expected)
    GCI = Fs * abs(f2 - f1) / (r**p_gci - 1)

    pente, intercept = np.polyfit(np.log(h_plot), np.log(err), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(h_plot, err, "o-", markersize=8, linewidth=2,
              label="Erreur relative")
    ax.loglog(h_plot, np.exp(intercept) * h_plot**pente, "--", alpha=0.6,
              label=f"Fit global p = {pente:.2f}")

    for h, e in zip(h_plot, err):
        ax.annotate(f"{e*100:.2f}%", (h, e),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=12)

    info = "\n".join([
        f"p (polyfit)      = {pente:.2f}",
        f"p (Richardson)   = {p_rich:.2f}",
        f"GCI              = {GCI:.4f} N",
        f"Verif. asymptot. = {asympt:.2f} (<= 0.1)",
        f"Sol. extrapolee  = {f_rich:.6f} N",
    ])
    ax.text(0.90, 0.05, info, transform=ax.transAxes,
            fontsize=12, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", alpha=0.3))

    ax.grid(True, which="both", alpha=1)
    ax.set_xlabel("Taille du maillage h (mm)")
    ax.set_ylabel("Erreur relative")
    ax.set_title(
        f"Convergence asymptotique - Richardson / GCI\n"
        f"(L={L_REF:.0f} mm, delta={D_REF:.0f} mm, h <= {h_max} mm)"
    )
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS, "ansys_conv_asymptotic.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")

    print("\n===== RESULTATS ASYMPTOTIQUES =====")
    print(f"p (fit global)    = {pente:.4f}")
    print(f"p (Richardson)    = {p_rich:.4f}")
    print(f"GCI               = {GCI:.4f} N")
    print(f"Verif. asymptot.  = {asympt:.4f}")
    print(f"Sol. extrapolee   = {f_rich:.6f} N")


def compute_gci(h_values, forces, h_max, p_expected):
    """Return (GCI_N, GCI_pct, p_rich, F_rich) for the asymptotic subset."""
    mask = h_values <= h_max
    h_sub, f_sub = h_values[mask], forces[mask]
    h1, h2, h3   = h_sub[0], h_sub[1], h_sub[2]
    f1, f2, f3   = f_sub[0], f_sub[1], f_sub[2]
    r      = h2 / h1
    p_rich = np.log((f3 - f2) / (f2 - f1)) / np.log(r)
    f_rich = f1 + (f1 - f2) / (r**p_rich - 1)
    asympt = abs(p_expected - p_rich) / p_rich
    Fs     = 1.25 if asympt <= 0.1 else 3.0
    p_gci  = p_expected if asympt <= 0.1 else min(max(0.5, p_rich), p_expected)
    GCI_N  = Fs * abs(f2 - f1) / (r**p_gci - 1)
    return GCI_N, GCI_N / f1, p_rich, f_rich


if __name__ == "__main__":
    plot_total(h_all, err_all, h_values[0])
    plot_asymptotic(h_values, forces, H_MAX, P_EXPECTED)

    GCI_N, GCI_pct, p_rich, F_rich = compute_gci(h_values, forces, H_MAX, P_EXPECTED)

    import csv as _csv
    out_path = os.path.join(RESULTS, "ansys_gci.csv")
    with open(out_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=[
            "L_span_mm", "delta_mm", "GCI_N", "GCI_pct", "p_rich", "F_extrapolated_N"])
        w.writeheader()
        w.writerow({
            "L_span_mm":       L_REF,
            "delta_mm":        D_REF,
            "GCI_N":           round(GCI_N, 6),
            "GCI_pct":         round(GCI_pct, 6),
            "p_rich":          round(p_rich, 4),
            "F_extrapolated_N": round(F_rich, 6),
        })
    print(f"Saved {out_path}")
