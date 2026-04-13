"""
Linear-regime stiffness comparison: 1D EB FEM vs 2D ANSYS vs experimental.

For each span, extracts the linear stiffness k = F/delta [N/mm] from each
source and back-calculates the effective Young's modulus:
    E_eff = k * L^3 / (48 * I)

Sources
-------
  1D EB    : analytical k = 48*E0*I/L^3  (E0 = 6*(C10+C01) = 5.559 MPa)
  ANSYS    : k estimated from the two smallest delta values in simulation.csv
             (delta = 0.05 and 0.1 mm, confirmed linear by F/delta = const)
  Exp      : k from linear regression of F_mean vs delta through the origin
             (L=60mm data covers delta=3-5mm, confirmed linear by F/delta ≈ const;
              L=40mm shows mild non-linearity at delta=3-5mm -- noted)

Run with:  python scripts/linear_stiffness.py
"""

import csv
import os
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# ---------------------------------------------------------------------------
# Material and geometry
# ---------------------------------------------------------------------------
C10 = 2.6643e5
C01 = 6.6007e5
E0  = 6.0 * (C10 + C01)   # linearized E [Pa]
d   = 5.0e-3               # nominal diameter [m]
I   = np.pi * d**4 / 64   # second moment of area [m^4]


def E_from_k(k_N_per_mm, L_mm):
    """Back-calculate E [MPa] from stiffness k [N/mm] and span L [mm]."""
    return k_N_per_mm * L_mm**3 / (48.0 * I * 1e12) * 1e-6
    # units: (N/mm) * mm^3 / (m^4 * (mm/m)^4) ... let's keep consistent:
    # k [N/m], L [m] -> E [Pa]


def E_from_k_SI(k_N_per_m, L_m):
    """Back-calculate E [Pa] from stiffness k [N/m] and span L [m]."""
    return k_N_per_m * L_m**3 / (48.0 * I)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_simulation():
    rows = {}
    with open(os.path.join(DATA, "simulation.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            L = float(row["L_span_mm"])
            d_mm = float(row["delta_mm"])
            rows.setdefault(L, {})[d_mm] = float(row["F_sim_N"])
    return rows


def load_experimental():
    """Returns {L_mm: {delta_mm: [F_exp per specimen]}}"""
    groups = {}
    with open(os.path.join(DATA, "experimental.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            L = float(row["L_span_mm"])
            delta = float(row["delta_mm"])
            groups.setdefault(L, {}).setdefault(delta, []).append(float(row["F_exp_N"]))
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = load_simulation()
    exp = load_experimental()

    print("=" * 68)
    print("Linear stiffness and effective E -- small-deformation regime")
    print("=" * 68)
    print(f"  Geometry : d = {d*1e3:.1f} mm,  I = {I*1e12:.3f} mm^4")
    print(f"  Model E0 : {E0/1e6:.4f} MPa  (linearized Mooney-Rivlin)")

    for L_mm in [60.0, 40.0]:
        L = L_mm * 1e-3
        print()
        print(f"  {'='*60}")
        print(f"  L_span = {L_mm:.0f} mm   (L/d = {L_mm/5:.0f})")
        print(f"  {'='*60}")

        # --- 1D EB analytical stiffness ---
        k_1D = 48.0 * E0 * I / L**3   # [N/m]
        E_1D = E0 / 1e6
        print(f"\n  1D EB (analytical):")
        print(f"    k = {k_1D*1e-3:8.4f} N/mm    E_eff = {E_1D:.4f} MPa")

        # --- ANSYS: stiffness from smallest two delta values ---
        sim_L = sim.get(L_mm, {})
        small_deltas = sorted([d_mm for d_mm in sim_L if d_mm <= 0.1])
        if small_deltas:
            k_vals = [sim_L[d_mm] / (d_mm * 1e-3) for d_mm in small_deltas]
            k_ANSYS = np.mean(k_vals)  # [N/m]
            E_ANSYS = E_from_k_SI(k_ANSYS, L) / 1e6
            print(f"\n  ANSYS 2D (linear regime, delta <= 0.1 mm):")
            for d_mm in small_deltas:
                k_pt = sim_L[d_mm] / (d_mm * 1e-3)
                print(f"    delta={d_mm:.2f}mm: F={sim_L[d_mm]:.5f} N  "
                      f"k={k_pt*1e-3:.4f} N/mm  E_eff={E_from_k_SI(k_pt, L)/1e6:.4f} MPa")
            print(f"    Mean: k = {k_ANSYS*1e-3:8.4f} N/mm    "
                  f"E_eff = {E_ANSYS:.4f} MPa  "
                  f"({(k_ANSYS/k_1D - 1)*100:+.1f}% vs 1D EB)")

        # --- Experimental: linearity check and stiffness estimate ---
        exp_L = exp.get(L_mm, {})
        test_deltas = sorted(exp_L.keys())
        F_means = [np.mean(exp_L[d_mm]) for d_mm in test_deltas]
        F_stds  = [np.std(exp_L[d_mm], ddof=1) for d_mm in test_deltas]
        k_ratios = [F_means[i] / (test_deltas[i] * 1e-3) for i in range(len(test_deltas))]

        # linear regression through origin: k = sum(delta*F)/sum(delta^2)
        deltas_arr = np.array(test_deltas) * 1e-3  # [m]
        F_arr      = np.array(F_means)
        k_exp      = np.dot(deltas_arr, F_arr) / np.dot(deltas_arr, deltas_arr)
        E_exp      = E_from_k_SI(k_exp, L) / 1e6

        print(f"\n  Experimental (per-delta F/delta ratio):")
        print(f"  {'delta (mm)':>10}  {'F_mean (N)':>10}  {'F_std (N)':>9}  "
              f"{'k (N/mm)':>9}  {'E_eff (MPa)':>11}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*11}")
        for i, d_mm in enumerate(test_deltas):
            k_pt   = k_ratios[i]
            E_pt   = E_from_k_SI(k_pt, L) / 1e6
            linear = "  (linear)" if abs(k_pt/k_ratios[0] - 1) < 0.02 else ""
            print(f"  {d_mm:>10.1f}  {F_means[i]:>10.5f}  {F_stds[i]:>9.5f}  "
                  f"{k_pt*1e-3:>9.4f}  {E_pt:>11.4f}{linear}")

        variation = (max(k_ratios) - min(k_ratios)) / max(k_ratios) * 100
        print(f"  F/delta variation: {variation:.1f}% "
              f"({'linear' if variation < 2 else 'non-linear'})")
        print(f"  Regression k   = {k_exp*1e-3:.4f} N/mm    "
              f"E_eff = {E_exp:.4f} MPa  "
              f"({(k_exp/k_1D - 1)*100:+.1f}% vs 1D EB)")

        # --- Summary ---
        print(f"\n  Summary:")
        print(f"  {'Source':>12}  {'k (N/mm)':>10}  {'E_eff (MPa)':>12}  {'vs 1D EB':>9}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*9}")
        print(f"  {'1D EB':>12}  {k_1D*1e-3:>10.4f}  {E_1D:>12.4f}  {'---':>9}")
        if small_deltas:
            print(f"  {'ANSYS 2D':>12}  {k_ANSYS*1e-3:>10.4f}  {E_ANSYS:>12.4f}  "
                  f"{(E_ANSYS/E_1D - 1)*100:>+8.1f}%")
        print(f"  {'Exp.':>12}  {k_exp*1e-3:>10.4f}  {E_exp:>12.4f}  "
              f"{(E_exp/E_1D - 1)*100:>+8.1f}%")

    print()
