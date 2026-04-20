"""
Calibrate Young's modulus from cantilever tip deflection measurements.

Setup: each specimen is clamped at one end.  Free length (L_cant) and
tip mass (m_tip) are read per-row from cantilever_exp.csv, so future
specimens with different setups are handled automatically.

Small-strain EB formula (includes self-weight):
    v_tip = [F_tip * L^3/3 + w * L^4/8] / (E * I)

Inverted:
    E = [F_tip * L^3/3 + w * L^4/8] / (I * v_tip)

where
    F_tip = m_TIP * g              [N]
    w     = mass_free * g / L_cant [N/m]  (beam self-weight per unit length)
    I     = pi * d^4 / 64         [m^4]

Reads:  data/specimens.csv
        data/cantilever_exp.csv
Saves:  results/calibrate_E.csv

Run with:  python scripts/postprocess_exp_cantilever_E.py
"""

import csv
import os
import numpy as np

DATA    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# --- Constants ---
G = 9.81   # m/s^2

# Reference E from Mooney-Rivlin constants (literature)
C10 = 2.6643e5
C01 = 6.6007e5
E0  = 6.0 * (C10 + C01)

# --- Loaders ---

def load_specimens():
    out = {}
    with open(os.path.join(DATA, "specimens.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            sid = int(r["specimen_id"])
            out[sid] = {
                "d_m":      float(r["d_mean_mm"]) * 1e-3,
                "mass_kg":  float(r["mass_g"]) * 1e-3,
                "length_m": float(r["length_mm"]) * 1e-3,
            }
    return out


def load_deflections():
    out = {}
    with open(os.path.join(DATA, "cantilever_exp.csv"), newline="") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    for r in csv.DictReader(lines):
        sid = int(r["specimen_id"])
        out[sid] = {
            "vs":     [float(r["v1_mm"]), float(r["v2_mm"]), float(r["v3_mm"])],
            "m_tip":  float(r["tip_mass_g"]) * 1e-3,
            "L_cant": float(r["L_cant_mm"]) * 1e-3,
        }
    return out


# --- E back-calculation ---

def calibrate_E(d, mass_full, length_full, L_cant, m_tip, v_tip_mm):
    """
    Back-calculate E [Pa] from a single tip deflection measurement.

    Parameters
    ----------
    d          : rod diameter [m]
    mass_full  : full rod mass [kg]
    length_full: full rod length [m]
    L_cant     : free (clamped-to-tip) length [m]
    m_tip      : tip mass [kg]
    v_tip_mm   : measured tip deflection [mm] (positive = downward)

    Returns
    -------
    E : float [Pa]
    """
    I         = np.pi * d**4 / 64
    F_tip     = m_tip * G
    mass_free = mass_full * (L_cant / length_full)
    w         = mass_free * G / L_cant
    numerator = F_tip * L_cant**3 / 3 + w * L_cant**4 / 8
    return numerator / (I * v_tip_mm * 1e-3)


# --- Main ---

if __name__ == "__main__":
    specimens   = load_specimens()
    deflections = load_deflections()

    print("=" * 68)
    print("Young's modulus calibration from cantilever experiment")
    print("=" * 68)
    print(f"  L_cant, m_tip = read per specimen from cantilever_exp.csv")
    print(f"  E0 (Mooney-Rivlin, literature) = {E0/1e6:.3f} MPa\n")

    print(f"  {'spec':>4}  {'d (mm)':>7}  {'v1':>6}  {'v2':>6}  {'v3':>6}"
          f"  {'E_mean (MPa)':>13}  {'std (MPa)':>10}  {'CV (%)':>7}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}"
          f"  {'-'*13}  {'-'*10}  {'-'*7}")

    csv_rows = []
    E_all = []

    for sid in sorted(specimens):
        s   = specimens[sid]
        rec = deflections[sid]
        vs  = rec["vs"]
        Es  = [calibrate_E(s["d_m"], s["mass_kg"], s["length_m"],
                           rec["L_cant"], rec["m_tip"], v) for v in vs]
        E_m = np.mean(Es)
        E_s = np.std(Es, ddof=1)
        cv  = E_s / E_m * 100
        E_all.extend(Es)

        print(f"  {sid:>4}  {s['d_m']*1e3:>7.3f}  "
              f"{vs[0]:>6.1f}  {vs[1]:>6.1f}  {vs[2]:>6.1f}  "
              f"{E_m/1e6:>13.3f}  {E_s/1e6:>10.4f}  {cv:>6.2f}%")

        csv_rows.append({
            "specimen_id": sid,
            "d_mean_mm":   round(s["d_m"]*1e3, 3),
            "v1_mm": vs[0], "v2_mm": vs[1], "v3_mm": vs[2],
            "E_mean_MPa":  round(E_m/1e6, 4),
            "E_std_MPa":   round(E_s/1e6, 5),
            "CV_pct":      round(cv, 3),
        })

    E_arr       = np.array(E_all)
    E_grand     = E_arr.mean()
    E_grand_std = E_arr.std(ddof=1)
    u_E         = E_grand_std / np.sqrt(len(E_arr))

    print()
    print(f"  Grand mean : E = {E_grand/1e6:.3f} +/- {E_grand_std/1e6:.3f} MPa"
          f"  (u_E = {u_E/1e6:.4f} MPa, n={len(E_arr)})")
    print(f"  E0 (lit.)  :     {E0/1e6:.3f} MPa  "
          f"(ratio E_exp/E0 = {E_grand/E0:.2f})")

    out_path = os.path.join(RESULTS, "calibrate_E.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n  Saved {out_path}")
