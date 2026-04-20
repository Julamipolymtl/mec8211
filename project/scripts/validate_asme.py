"""
Validation: 1D EB FEM + 2D ANSYS vs experimental data  (ASME V&V 20).

Reads pre-computed intermediate CSVs:
  results/1d_u_input.csv      -- F_nom, u_input, MC band  (from 1d_mc_propagation.py)
  results/1d_gci.csv          -- u_num ~ 0 (Hermite exactness)
  results/ansys_u_input.csv   -- dF/dE * sigma_E per condition  (ansys_sensitivity.py)
  results/ansys_gci.csv       -- GCI for reference condition     (ansys_convergence.py)

ASME V&V 20 (1D EB FEM):
  u_num   ~ 0    : 3-pt bending is cubic -> Hermite exactness
  u_input        : 95% MC half-width (from 1d_u_input.csv)
  U_D            : t_{0.975,5} * std(F_exp) / sqrt(6)
  U_val = sqrt(u_num^2 + u_input^2 + U_D^2)

  E = F_exp_mean - F_sim;  pass if |E| < U_val

ASME V&V 20 (ANSYS 2D):
  u_num          : GCI from mesh convergence study (ansys_gci.csv)
  u_input        : |dF/dE| * sigma_E  (ansys_u_input.csv)
  U_D            : same as 1D
  U_val = sqrt(u_num^2 + u_input^2 + U_D^2)

Generates:
  results/validation_ansys.png
  results/validation_1d.png
  results/validation_error_budget_ansys.png
  results/validation_error_budget.png

Run with:  python scripts/validate_asme.py
"""

import csv
import sys
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loaders import load_specimens, load_experimental, load_simulation, load_calibrated_E

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

TEST_CONDITIONS = [(60.0, d) for d in [3.0, 4.0, 5.0]] + \
                  [(40.0, d) for d in [3.0, 4.0, 5.0]]

N_SPEC = 6
T_95   = stats.t.ppf(0.975, df=N_SPEC - 1)


# ---------------------------------------------------------------------------
# Intermediate CSV loaders
# ---------------------------------------------------------------------------

def _try_float(v):
    v = v.strip()
    try:
        return float(v)
    except (ValueError, AttributeError):
        return v


def _load_results_csv(filename):
    path = os.path.join(RESULTS, filename)
    with open(path, newline="") as fh:
        return [{k: _try_float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def load_1d_u_input():
    """Returns {(L_mm, delta_mm): row_dict} from results/1d_u_input.csv."""
    return {(r["L_span_mm"], r["delta_mm"]): r for r in _load_results_csv("1d_u_input.csv")}


def load_1d_gci():
    """Returns u_num_N (0.0 for Hermite exactness) from results/1d_gci.csv."""
    return _load_results_csv("1d_gci.csv")[0]["u_num_N"]


def load_ansys_u_input():
    """Returns {(L_mm, delta_mm): u_input_N} from results/ansys_u_input.csv."""
    return {(r["L_span_mm"], r["delta_mm"]): r["u_input_N"]
            for r in _load_results_csv("ansys_u_input.csv")}


def load_ansys_gci():
    """Returns GCI_N for the reference mesh-convergence condition."""
    return _load_results_csv("ansys_gci.csv")[0]["GCI_N"]


# ---------------------------------------------------------------------------
# Experimental uncertainty
# ---------------------------------------------------------------------------

def u_D_95(F_exp_arr):
    n = len(F_exp_arr)
    return float(stats.t.ppf(0.975, df=n - 1) * np.std(F_exp_arr, ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------------------
# Assemble 1D validation table
# ---------------------------------------------------------------------------

def build_validation_table(exp_data, u1d_lookup, u_num_1d):
    rows = []
    for L_mm, delta_mm in TEST_CONDITIONS:
        key = (L_mm, delta_mm)
        if key not in exp_data or key not in u1d_lookup:
            continue

        F_exp_arr  = np.array(list(exp_data[key].values()))
        F_exp_mean = float(F_exp_arr.mean())
        F_exp_std  = float(F_exp_arr.std(ddof=1))
        UD         = u_D_95(F_exp_arr)

        mc      = u1d_lookup[key]
        F_sim   = mc["F_nom_N"]
        UI      = mc["u_input_N"]
        F_lo    = mc["F_mc_lo_N"]
        F_hi    = mc["F_mc_hi_N"]

        E_val   = F_exp_mean - F_sim
        U_val   = float(np.sqrt(u_num_1d**2 + UI**2 + UD**2))
        passed  = abs(E_val) < U_val

        rows.append({
            "L_mm":       L_mm,
            "delta_mm":   delta_mm,
            "F_exp_mean": F_exp_mean,
            "F_exp_std":  F_exp_std,
            "U_D":        UD,
            "F_sim_mean": F_sim,
            "F_mc_lo":    F_lo,
            "F_mc_hi":    F_hi,
            "U_input":    UI,
            "U_num":      u_num_1d,
            "E":          E_val,
            "U_val":      U_val,
            "passed":     passed,
        })
    return rows


# ---------------------------------------------------------------------------
# Plot: F vs delta (ANSYS or 1D)
# ---------------------------------------------------------------------------

def _exp_scatter(ax, L_mm, deltas, exp_data, rows):
    for d_mm in deltas:
        key = (L_mm, d_mm)
        for F in exp_data[key].values():
            ax.scatter(d_mm, F, color="tab:blue", alpha=0.3, s=15, zorder=2)
    F_means = [r["F_exp_mean"] for r in rows]
    U_Ds    = [r["U_D"]        for r in rows]
    ax.errorbar(deltas, F_means, yerr=U_Ds, fmt="o-", color="tab:blue",
                capsize=4, label="Exp. mean $\\pm U_D$ (95%)", zorder=4)
    ax.set_xlabel("$\\delta$ [mm]")
    ax.set_ylabel("$F$ [N]")
    ax.grid()


def plot_ansys_validation(rows, exp_data, sim_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, L_mm in zip(axes, [60.0, 40.0]):
        cond   = [r for r in rows if r["L_mm"] == L_mm]
        deltas = [r["delta_mm"] for r in cond]
        _exp_scatter(ax, L_mm, deltas, exp_data, cond)

        ansys_d = sorted(d for (L, d) in sim_data if L == L_mm and d in deltas)
        ansys_F = [sim_data[(L_mm, d)] for d in ansys_d]
        if ansys_d:
            ax.plot(ansys_d, ansys_F, "^:", color="tab:green",
                    label="ANSYS 2D", zorder=3)

        ax.set_title(f"L = {L_mm:.0f} mm  (L/d = {L_mm/5.0:.0f})")
        ax.legend(fontsize=8)

    fig.suptitle("Force-Displacement: ANSYS 2D vs Experimental", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS, "validation_ansys.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_1d_validation(rows, exp_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, L_mm in zip(axes, [60.0, 40.0]):
        cond   = [r for r in rows if r["L_mm"] == L_mm]
        deltas = [r["delta_mm"] for r in cond]
        _exp_scatter(ax, L_mm, deltas, exp_data, cond)

        F_sims = [r["F_sim_mean"] for r in cond]
        F_los  = [r["F_mc_lo"]    for r in cond]
        F_his  = [r["F_mc_hi"]    for r in cond]
        ax.plot(deltas, F_sims, "s--", color="tab:orange",
                label="1D FEM (calibrated $E$)", zorder=4)
        ax.fill_between(deltas, F_los, F_his, alpha=0.2, color="tab:orange",
                        label="1D FEM 95% MC band")

        ax.set_title(f"L = {L_mm:.0f} mm  (L/d = {L_mm/5.0:.0f})")
        ax.legend(fontsize=8)

    fig.suptitle("Force-Displacement: 1D EB FEM vs Experimental", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS, "validation_1d.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot: validation error budget (continuous band)
# ---------------------------------------------------------------------------

C_VV20 = 7.0


def _error_budget_axes(ax, deltas, E_vals, U_vals):
    from scipy.interpolate import interp1d
    from matplotlib.patches import Patch

    d = np.array(deltas)
    E = np.array(E_vals)
    U = np.array(U_vals)

    kind   = "quadratic" if len(d) >= 3 else "linear"
    d_fine = np.linspace(d[0], d[-1], 300)
    E_fine = interp1d(d, E, kind=kind)(d_fine)
    U_fine = interp1d(d, U, kind=kind)(d_fine)

    ax.fill_between(d_fine, E_fine - U_fine, E_fine + U_fine,
                    color="tab:blue", alpha=0.15)
    ax.plot(d_fine, E_fine + U_fine, color="tab:blue", lw=1.2, ls="--")
    ax.plot(d_fine, E_fine - U_fine, color="tab:blue", lw=1.2, ls="--")
    ax.plot(d_fine, E_fine, color="tab:blue", lw=2)

    ax.scatter(d, E, color="tab:blue", s=70, zorder=5)

    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.set_xlabel("Applied displacement $\\delta$ [mm]")
    ax.set_ylabel("$E = F_{exp} - F_{sim}$ [N]")
    ax.set_xticks(d)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        plt.Line2D([0], [0], color="tab:blue", lw=2, label="$E(\\delta)$"),
        Patch(color="tab:blue", alpha=0.2, label="$E \\pm U_{val}$ (95%)"),
        plt.Line2D([0], [0], color="k", lw=1.2, ls="--", label="$E = 0$"),
    ], fontsize=9)


def plot_error_budget_ansys(exp_data, sim_lookup, ansys_u_input, ansys_gci_N):
    ansys_rows = []
    for L_mm, delta_mm in TEST_CONDITIONS:
        key = (L_mm, delta_mm)
        if key not in exp_data:
            continue
        F_arr = np.array(list(exp_data[key].values()))
        F_sim = sim_lookup.get(key)
        if F_sim is None:
            continue
        UD    = u_D_95(F_arr)
        u_inp = ansys_u_input.get(key, 0.0)
        U_val = float(np.sqrt(ansys_gci_N**2 + u_inp**2 + UD**2))
        E_val = float(F_arr.mean()) - F_sim
        ansys_rows.append({
            "L_mm": L_mm, "delta_mm": delta_mm,
            "E": E_val, "U": U_val, "passed": abs(E_val) < U_val,
        })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, L_mm in zip(axes, [60.0, 40.0]):
        cond = [r for r in ansys_rows if r["L_mm"] == L_mm]
        _error_budget_axes(ax,
                           [r["delta_mm"] for r in cond],
                           [r["E"]        for r in cond],
                           [r["U"]        for r in cond])
        ax.set_title(f"L = {L_mm:.0f} mm  (L/d = {L_mm/5:.0f})")
    fig.suptitle(
        "ANSYS 2D -- Validation Error vs $U_{val}$ (95%)\n"
        "$U_{val} = \\sqrt{u_{num}^2 + u_{input}^2 + U_D^2}$",
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(RESULTS, "validation_error_budget_ansys.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_error_budget(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, L_mm in zip(axes, [60.0, 40.0]):
        cond = [r for r in rows if r["L_mm"] == L_mm]
        _error_budget_axes(ax,
                           [r["delta_mm"] for r in cond],
                           [r["E"]        for r in cond],
                           [r["U_val"]    for r in cond])
        ax.set_title(f"L = {L_mm:.0f} mm  (L/d = {L_mm/5:.0f})")
    fig.suptitle(
        "1D EB FEM -- Validation Error vs $U_{val}$ (95%)\n"
        "$U_{val} = \\sqrt{u_{num}^2 + u_{input}^2 + U_D^2}$",
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(RESULTS, "validation_error_budget.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Console reports
# ---------------------------------------------------------------------------

def print_2d_table(exp_data, sim_lookup):
    conditions = sorted(exp_data.keys())
    print()
    print("=" * 70)
    print("2D ANSYS simulation vs experimental  (finest mesh per span)")
    print("=" * 70)
    print(f"  {'L (mm)':>6}  {'d (mm)':>6}  {'F_exp mean':>12}  "
          f"{'F_exp std':>10}  {'F_sim':>10}  {'err (%)':>9}  {'n':>3}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*3}")

    abs_errors = []
    for (L, delta) in conditions:
        F_arr  = np.array(list(exp_data[(L, delta)].values()))
        F_mean = float(F_arr.mean())
        F_std  = float(F_arr.std(ddof=1))
        F_sim  = sim_lookup.get((L, delta))

        if F_sim is None:
            err_str   = "   no sim"
            F_sim_str = "       ---"
        else:
            rel_err = (F_sim - F_mean) / F_mean * 100.0
            abs_errors.append(abs(rel_err))
            err_str   = f"{rel_err:+9.2f}%"
            F_sim_str = f"{F_sim:10.5f}"

        print(f"  {L:>6.0f}  {delta:>6.1f}  {F_mean:>12.5f}  "
              f"{F_std:>10.5f}  {F_sim_str}  {err_str}  {len(F_arr):>3}")

    if abs_errors:
        print()
        print(f"  Mean |err| : {np.mean(abs_errors):.1f}%")
        print(f"  Max  |err| : {np.max(abs_errors):.1f}%")
    print()


def print_ansys_report(exp_data, sim_lookup, ansys_u_input, ansys_gci_N):
    print()
    print("=" * 88)
    print(f"ASME V&V 20 validation summary -- ANSYS 2D  (C = {C_VV20:.0f}, 95% coverage)")
    print("=" * 88)
    print(f"  {'cond':>10}  {'|E| (N)':>9}  {'U_val (N)':>10}  "
          f"{'C*U_val':>9}  {'U_val/C':>9}  {'result':>8}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}")

    for L_mm, delta_mm in TEST_CONDITIONS:
        key = (L_mm, delta_mm)
        if key not in exp_data:
            continue
        F_arr = np.array(list(exp_data[key].values()))
        F_sim = sim_lookup.get(key)
        if F_sim is None:
            print(f"  {'L'+str(int(L_mm))+'/d'+str(int(delta_mm)):>10}  {'---':>9}  {'---':>10}  "
                  f"{'---':>9}  {'---':>9}  {'no sim':>8}")
            continue
        UD    = u_D_95(F_arr)
        u_inp = ansys_u_input.get(key, 0.0)
        U_val = float(np.sqrt(ansys_gci_N**2 + u_inp**2 + UD**2))
        E_abs = abs(float(F_arr.mean()) - F_sim)
        cond  = f"L{L_mm:.0f}/d{delta_mm:.0f}"
        result = "PASS" if E_abs < U_val else "FAIL"
        print(f"  {cond:>10}  {E_abs:>9.4f}  {U_val:>10.4f}  "
              f"{C_VV20*U_val:>9.4f}  {U_val/C_VV20:>9.4f}  {result:>8}")

    print()
    print(f"  Notes:")
    print(f"    u_num  = GCI from ansys_convergence.py")
    print(f"    U_D    = t_{{0.975,{N_SPEC-1}}} * std(F_exp) / sqrt({N_SPEC})  [t = {T_95:.3f}]")
    print(f"    u_input = |dF/dE| * sigma_E  from ansys_sensitivity.py")
    print()


def print_report(rows, u_num_1d, E_calib_mean, E_calib_std):
    print()
    print("=" * 80)
    print("ASME V&V 20 validation summary  (all uncertainties at 95% coverage)")
    print("=" * 80)
    print(f"  {'cond':>10}  {'F_exp':>8}  {'F_sim':>8}  {'E':>8}  "
          f"{'U_D':>8}  {'U_input':>8}  {'U_val':>8}  {'result':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in rows:
        cond   = f"L{r['L_mm']:.0f}/d{r['delta_mm']:.0f}"
        result = "PASS" if r["passed"] else "FAIL"
        print(f"  {cond:>10}  {r['F_exp_mean']:>8.4f}  {r['F_sim_mean']:>8.4f}  "
              f"{r['E']:>+8.4f}  "
              f"{r['U_D']:>8.4f}  {r['U_input']:>8.4f}  {r['U_val']:>8.4f}  "
              f"{result:>8}")
    print()
    print("  Notes:")
    print(f"    u_num = {u_num_1d:.2e}  (3-pt bending: Hermite exactness)")
    print(f"    U_D   = t_{{0.975,{N_SPEC-1}}} * std(F_exp) / sqrt({N_SPEC})  [t = {T_95:.3f}]")
    print(f"    u_input from 1d_mc_propagation.py  "
          f"[E ~ Normal({E_calib_mean/1e6:.2f}, {E_calib_std/1e6:.2f}) MPa]")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    specimens  = load_specimens()
    exp_data   = load_experimental()
    sim_data   = load_simulation()
    E_calib_mean, E_calib_std = load_calibrated_E()

    print(f"Calibrated E (cantilever): mean = {E_calib_mean/1e6:.3f} MPa,  "
          f"std = {E_calib_std/1e6:.3f} MPa")

    print("Loading intermediate CSVs ...")
    u1d_lookup     = load_1d_u_input()
    u_num_1d       = load_1d_gci()
    ansys_u_inp    = load_ansys_u_input()
    ansys_gci_N    = load_ansys_gci()

    print_2d_table(exp_data, sim_data)

    print_ansys_report(exp_data, sim_data, ansys_u_inp, ansys_gci_N)

    rows = build_validation_table(exp_data, u1d_lookup, u_num_1d)

    print_report(rows, u_num_1d, E_calib_mean, E_calib_std)

    plot_ansys_validation(rows, exp_data, sim_data)
    plot_error_budget_ansys(exp_data, sim_data, ansys_u_inp, ansys_gci_N)
    plot_1d_validation(rows, exp_data)
    plot_error_budget(rows)

    print("Done.")
