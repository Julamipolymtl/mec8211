"""
Monte Carlo propagation of input uncertainties -- 1D EB beam model.

Computes u_input = std(F_sim | uncertain inputs) for each test condition,
following the ASME V&V 20 framework:
    delta_S = delta_model + delta_input + delta_num
    u_val   = sqrt(u_num^2 + u_input^2 + u_D^2)

Uncertain inputs
----------------
  d         : beam diameter     -- Normal(d_mean_i, sigma_d_i)
                sigma from 3 caliper measurements per specimen (ddof=1)
  L_span    : fixture span      -- Normal(L_nom, SIGMA_L)   [assumed]
  delta     : applied disp.     -- Normal(delta_nom, SIGMA_DELTA) [assumed]

Model used
----------
  F = 48 * E0 * (pi*d^4/64) / L^3 * delta    (analytical limit of 1D EB FEM;
  convergence to this formula was verified in the convergence study at N=20.)

Output
------
  Prints summary table per (L_span, delta)
  Prints per-specimen detail and variance decomposition
  Saves results/mc_u_input.csv

Run with:  python scripts/mc_u_input.py
"""

import csv
import os
import numpy as np

DATA    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
C10 = 2.6643e5          # Pa
C01 = 6.6007e5          # Pa
E0  = 6.0 * (C10 + C01) # linearized E [Pa], nu=0.5 (incompressible)

# ---------------------------------------------------------------------------
# Uncertainty assumptions for inputs without repeated measurements
# ---------------------------------------------------------------------------
# L_span: machined aluminum fixture, estimated tolerance +/- 0.5 mm (k=2 -> 95%)
SIGMA_L = 0.25e-3        # 1-sigma [m]

# delta: servo test machine, typical encoder resolution ~0.001 mm,
#        assume combined uncertainty +/- 0.01 mm (k=2)
SIGMA_DELTA = 0.005e-3   # 1-sigma [m]

N_MC     = 10_000
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_specimens():
    """Returns {specimen_id: {d_mean_m, sigma_d_m}}."""
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


# ---------------------------------------------------------------------------
# Analytical model (verified equivalent to FEM at convergence)
# ---------------------------------------------------------------------------

def F_analytical(d, L, delta):
    """Midspan reaction force [N]. Accepts numpy arrays."""
    I = np.pi * d**4 / 64.0
    return 48.0 * E0 * I / L**3 * delta


# ---------------------------------------------------------------------------
# MC per condition
# ---------------------------------------------------------------------------

def run_mc(d_mean, sigma_d, L_nom, delta_nom, rng):
    """
    Propagate (d, L, delta) uncertainties through the beam model.
    Returns (F_mc array of length N_MC).
    """
    d_s     = rng.normal(d_mean,    sigma_d,     N_MC)
    L_s     = rng.normal(L_nom,     SIGMA_L,     N_MC)
    delta_s = rng.normal(delta_nom, SIGMA_DELTA, N_MC)
    return F_analytical(d_s, L_s, delta_s)


def variance_fractions(d_mean, sigma_d, L_nom, delta_nom, rng):
    """
    One-at-a-time variance decomposition.
    Returns (frac_d, frac_L, frac_delta) as fractions summing to 1.
    (Approximate; valid when inputs are independent and effects near-linear.)
    """
    d_s     = rng.normal(d_mean,    sigma_d,     N_MC)
    L_s     = rng.normal(L_nom,     SIGMA_L,     N_MC)
    delta_s = rng.normal(delta_nom, SIGMA_DELTA, N_MC)

    var_d     = F_analytical(d_s,    L_nom,    delta_nom).var()
    var_L     = F_analytical(d_mean, L_s,      delta_nom).var()
    var_delta = F_analytical(d_mean, L_nom,    delta_s).var()
    total     = var_d + var_L + var_delta + 1e-40
    return var_d/total, var_L/total, var_delta/total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    specimens = load_specimens()
    rng       = np.random.default_rng(RNG_SEED)

    print("=" * 72)
    print("MC input uncertainty propagation -- 1D EB beam model (ASME V&V 20)")
    print("=" * 72)
    print(f"  N_MC         = {N_MC:,}  (seed = {RNG_SEED})")
    print(f"  sigma_L      = {SIGMA_L*1e3:.3f} mm  (fixture, assumption)")
    print(f"  sigma_delta  = {SIGMA_DELTA*1e3:.4f} mm  (machine, assumption)")
    print()

    # --- Diameter uncertainty per specimen ---
    print("  Diameter uncertainty from 3 caliper readings per specimen:")
    print(f"  {'spec':>4}  {'d_mean (mm)':>12}  {'sigma_d (mm)':>13}  {'CV (%)':>7}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*13}  {'-'*7}")
    for sid, s in specimens.items():
        cv = s["sigma_d_m"] / s["d_mean_m"] * 100.0
        print(f"  {sid:>4}  {s['d_mean_m']*1e3:>12.3f}  "
              f"{s['sigma_d_m']*1e3:>13.4f}  {cv:>6.3f}%")

    csv_rows = []

    for L_mm in [60.0, 40.0]:
        L = L_mm * 1e-3
        print()
        print(f"  {'='*68}")
        print(f"  L_span = {L_mm:.0f} mm")
        print(f"  {'='*68}")

        for delta_mm in [3.0, 4.0, 5.0]:
            delta = delta_mm * 1e-3
            print()
            print(f"  delta = {delta_mm:.0f} mm")
            print(f"  {'spec':>4}  {'F_nom (N)':>10}  {'u_input (N)':>12}  "
                  f"{'u_rel (%)':>10}  "
                  f"{'var_d':>7}  {'var_L':>7}  {'var_dlt':>9}")
            print(f"  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*10}  "
                  f"{'-'*7}  {'-'*7}  {'-'*9}")

            u_inputs = []
            for sid in range(1, 7):
                s     = specimens[sid]
                F_nom = F_analytical(s["d_mean_m"], L, delta)
                F_mc  = run_mc(s["d_mean_m"], s["sigma_d_m"], L, delta, rng)
                u_inp = F_mc.std(ddof=1)
                u_rel = u_inp / F_nom * 100.0
                fd, fL, fdelta = variance_fractions(
                    s["d_mean_m"], s["sigma_d_m"], L, delta, rng)

                print(f"  {sid:>4}  {F_nom:>10.5f}  {u_inp:>12.6f}  "
                      f"{u_rel:>9.3f}%  "
                      f"{fd*100:>6.1f}%  {fL*100:>6.1f}%  {fdelta*100:>8.1f}%")

                u_inputs.append(u_inp)
                csv_rows.append({
                    "L_span_mm":   L_mm,
                    "delta_mm":    delta_mm,
                    "specimen_id": sid,
                    "d_mean_mm":   s["d_mean_m"] * 1e3,
                    "sigma_d_mm":  s["sigma_d_m"] * 1e3,
                    "F_nom_N":     round(F_nom, 7),
                    "u_input_N":   round(u_inp, 7),
                    "u_input_pct": round(u_rel, 4),
                })

            u_arr = np.array(u_inputs)
            print(f"  {'Mean':>4}  {'':>10}  {u_arr.mean():>12.6f}  "
                  f"{(u_arr.mean()/np.mean([F_analytical(specimens[s]['d_mean_m'], L, delta) for s in range(1,7)]))*100:>9.3f}%")

    # Save CSV
    out_path = os.path.join(RESULTS, "mc_u_input.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print()
    print(f"  Saved {out_path}")
    print()
    print("  Note: u_input combines with u_num (GCI) and u_D (experimental)")
    print("  to form the validation uncertainty:")
    print("    u_val = sqrt(u_num^2 + u_input^2 + u_D^2)")
