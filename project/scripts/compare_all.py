"""
Three-way comparison: 1D Euler-Bernoulli FEM vs 2D ANSYS vs experimental.

Three-point bending setup
-------------------------
  SS beam, prescribed midspan displacement delta, measure reaction force.
  Self-weight included as a uniform distributed load over the span.

Material model (same for 1D and ANSYS)
---------------------------------------
  TPU-60A modelled as a 2nd order Polynomial (Mooney-Rivlin) hyperelastic
  material with coefficients from the literature:
    C10 = 2.6643e5 Pa,  C01 = 6.6007e5 Pa  (d1 = d2 = 0: incompressible)
  Linearized (small-strain) equivalents used by the 1D model:
    mu0 = 2*(C10+C01)  [initial shear modulus]
    E0  = 3*mu0        [incompressible: nu = 0.5]

Geometry
--------
  1D vs experimental: per-specimen d_mean and mass from data/specimens.csv.
  1D vs ANSYS       : nominal d = 5.0 mm (what the ANSYS model used).

Note on expected agreement
--------------------------
  - L_span = 60 mm (L/d = 12): small strains, linear material regime.
    1D EB should agree closely with ANSYS; both may differ from experiment
    if the material model constants do not match the actual specimens.
  - L_span = 40 mm (L/d = 8): larger strains, geometric and material
    non-linearities are significant. ANSYS captures them (large-deformation
    solver, full hyperelastic law); 1D EB does not.

Reads:  data/experimental.csv
        data/simulation.csv
        data/specimens.csv

Run with:  python scripts/compare_all.py
"""

import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beam import assemble_K, assemble_distributed_load, solve

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# ---------------------------------------------------------------------------
# Material constants
# ---------------------------------------------------------------------------
C10  = 2.6643e5          # Pa
C01  = 6.6007e5          # Pa
MU0  = 2.0 * (C10 + C01) # initial shear modulus [Pa]
E0   = 3.0 * MU0          # linearized Young's modulus (nu=0.5) [Pa]
G    = 9.81               # m/s^2

N_ELEM = 20  # must be even

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def read_csv(filename):
    path = os.path.join(DATA, filename)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_specimens():
    """Returns dict: {specimen_id: {"d_m": d [m], "mass_kg": m [kg]}}"""
    out = {}
    for row in read_csv("specimens.csv"):
        out[int(row["specimen_id"])] = {
            "d_m":     float(row["d_mean_mm"]) * 1e-3,
            "mass_kg": float(row["mass_g"]) * 1e-3,
        }
    return out


def load_experimental():
    """Returns dict: {(L_mm, delta_mm): {specimen_id: F_exp_N}}"""
    groups = {}
    for row in read_csv("experimental.csv"):
        key  = (float(row["L_span_mm"]), float(row["delta_mm"]))
        sid  = int(row["specimen_id"])
        groups.setdefault(key, {})[sid] = float(row["F_exp_N"])
    return groups


def load_simulation():
    """Returns dict: {(L_mm, delta_mm): F_sim_N}"""
    return {
        (float(r["L_span_mm"]), float(r["delta_mm"])): float(r["F_sim_N"])
        for r in read_csv("simulation.csv")
    }


# ---------------------------------------------------------------------------
# 1D FEM solver
# ---------------------------------------------------------------------------

def compute_1d_force(L_span, delta, d, mass=None):
    """
    3-point bending: SS beam, prescribed midspan displacement delta [m].
    Self-weight is NOT included: the test machine zeroes the load cell at
    contact, so only the incremental stiffness is measured. ANSYS likewise
    ran without self-weight (F/delta is constant down to delta -> 0).
    Returns midspan reaction force [N].
    """
    I   = np.pi * d**4 / 64
    n   = N_ELEM
    mid = n // 2
    K   = assemble_K(n, E0, I, L_span)
    f   = np.zeros(2 * (n + 1))
    _, R = solve(K, f, [0, 2*n, 2*mid], [0.0, 0.0, delta])
    return R[2*mid]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _err(pred, ref):
    return (pred - ref) / ref * 100.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    specimens  = load_specimens()
    exp_data   = load_experimental()
    sim_data   = load_simulation()

    d_nom  = 5.0e-3                      # nominal diameter for 1D vs ANSYS [m]
    I_nom  = np.pi * d_nom**4 / 64
    m_mean = np.mean([s["mass_kg"] for s in specimens.values()])

    print("=" * 72)
    print("Three-way comparison: 1D EB FEM vs 2D ANSYS vs experimental")
    print("=" * 72)
    print(f"  E0 = {E0/1e6:.4f} MPa   (linearized: E0 = 6*(C10+C01))")
    print(f"  nu = 0.5 (incompressible)")
    print(f"  Nominal geometry: d = {d_nom*1e3:.1f} mm,  "
          f"I = {I_nom*1e12:.3f} mm^4")
    print(f"  Mean specimen mass: {m_mean*1e3:.3f} g")

    test_deltas = sorted({delta for (_, delta) in exp_data.keys()})

    for L_mm in [60.0, 40.0]:
        section(f"L_span = {L_mm:.0f} mm  (L/d = {L_mm/5.0:.0f})")

        # --- 1D vs ANSYS over the full force-displacement curve ---
        sim_deltas = sorted(
            delta for (L, delta) in sim_data if L == L_mm
        )
        print(f"\n  1D EB vs 2D ANSYS (nominal d={d_nom*1e3:.0f} mm, "
              f"m={m_mean*1e3:.2f} g)")
        print(f"  {'delta (mm)':>10}  {'F_1D (N)':>10}  "
              f"{'F_2D (N)':>10}  {'err_1D vs 2D':>14}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")
        for delta_mm in sim_deltas:
            delta = delta_mm * 1e-3
            F_1d  = compute_1d_force(L_mm*1e-3, delta, d_nom, m_mean)
            F_2d  = sim_data.get((L_mm, delta_mm))
            if F_2d is None:
                continue
            print(f"  {delta_mm:>10.2f}  {F_1d:>10.5f}  "
                  f"{F_2d:>10.5f}  {_err(F_1d, F_2d):>+13.1f}%")

        # --- 1D, 2D, experimental at tested displacements ---
        print(f"\n  All three vs experimental (per-specimen 1D uses actual d and m)")
        print(f"  {'delta':>5}  {'F_1D':>8}  {'F_2D':>8}  "
              f"{'F_exp mean':>10}  {'F_exp std':>9}  "
              f"{'err_1D':>8}  {'err_2D':>8}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*9}  "
              f"{'-'*8}  {'-'*8}")

        errs_1d = []
        errs_2d = []
        for delta_mm in sorted(test_deltas):
            key    = (L_mm, delta_mm)
            delta  = delta_mm * 1e-3
            F_2d   = sim_data.get(key)
            if key not in exp_data:
                continue
            per_spec = exp_data[key]
            # per-specimen 1D predictions
            F_1d_per = {
                sid: compute_1d_force(
                    L_mm*1e-3, delta,
                    specimens[sid]["d_m"],
                    specimens[sid]["mass_kg"],
                )
                for sid in per_spec
            }
            F_1d_mean = np.mean(list(F_1d_per.values()))
            F_exp_arr = np.array(list(per_spec.values()))
            F_exp_mean = np.mean(F_exp_arr)
            F_exp_std  = np.std(F_exp_arr, ddof=1)

            e1 = _err(F_1d_mean, F_exp_mean)
            e2 = _err(F_2d, F_exp_mean) if F_2d else None
            errs_1d.append(abs(e1))
            if e2 is not None:
                errs_2d.append(abs(e2))

            e2_str = f"{e2:+7.1f}%" if e2 is not None else "      ---"
            print(f"  {delta_mm:>4.0f}mm  {F_1d_mean:>8.5f}  "
                  f"{F_2d if F_2d else float('nan'):>8.5f}  "
                  f"{F_exp_mean:>10.5f}  {F_exp_std:>9.5f}  "
                  f"{e1:>+7.1f}%  {e2_str}")

        if errs_1d:
            print(f"\n  Summary vs experimental:")
            print(f"    1D EB  mean|err| = {np.mean(errs_1d):.1f}%,  "
                  f"max|err| = {np.max(errs_1d):.1f}%")
        if errs_2d:
            print(f"    2D FEM mean|err| = {np.mean(errs_2d):.1f}%,  "
                  f"max|err| = {np.max(errs_2d):.1f}%")

    print()
