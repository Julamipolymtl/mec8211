"""
Shared data loading utilities for project scripts.

All loaders derive paths relative to this file's location so scripts
can be run from any working directory.
"""

import csv
import os
import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _read_csv(filename):
    path = os.path.join(_DATA, filename)
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            def _cast(v):
                v = v.strip()
                if not v:
                    return None
                try:
                    return float(v)
                except ValueError:
                    return v
            rows.append({k: _cast(v) for k, v in row.items()})
    return rows


def load_specimens():
    """
    Returns dict: {specimen_id (int): {d_mean_m, sigma_d_m, mass_kg, length_m}}
    sigma_d_m is the sample std of the three diameter readings.
    """
    out = {}
    for r in _read_csv("specimens.csv"):
        sid = int(r["specimen_id"])
        ds  = np.array([r["d1_mm"], r["d2_mm"], r["d3_mm"]]) * 1e-3
        out[sid] = {
            "d_mean_m":  float(r["d_mean_mm"]) * 1e-3,
            "sigma_d_m": float(np.std(ds, ddof=1)),
            "mass_kg":   float(r["mass_g"]) * 1e-3,
            "length_m":  float(r["length_mm"]) * 1e-3,
        }
    return out


def load_experimental():
    """
    Returns dict: {(L_span_mm, delta_mm): {specimen_id: F_exp_N}}
    Keys use mm values as floats.
    """
    out = {}
    for r in _read_csv("experimental.csv"):
        key = (r["L_span_mm"], r["delta_mm"])
        out.setdefault(key, {})[int(r["specimen_id"])] = r["F_exp_N"]
    return out


def load_simulation():
    """
    Returns dict: {(L_span_mm, delta_mm): F_sim_N} for the nonlinear ANSYS
    runs at the finest mesh per span (production results).
    """
    rows = [r for r in _read_csv("simulation_ansys.csv") if r["model"] == "nonlinear"]

    counts = {}
    for r in rows:
        key = (r["L_span_mm"], r["h_mm"])
        counts[key] = counts.get(key, 0) + 1

    prod_h = {}
    for (L, h), n in counts.items():
        if n > prod_h.get(L, (0, None))[0]:
            prod_h[L] = (n, h)
    prod_h = {L: v[1] for L, v in prod_h.items()}

    return {(r["L_span_mm"], r["delta_mm"]): r["F_sim_N"]
            for r in rows if r["h_mm"] == prod_h[r["L_span_mm"]]}


def load_linear_sweep():
    """
    Returns dict: {(L_span_mm, delta_mm, E_MPa): F_sim_N} for the linear
    elastic ANSYS parametric E sweep.
    """
    rows = [r for r in _read_csv("simulation_ansys.csv") if r["model"] == "linear"]
    return {(r["L_span_mm"], r["delta_mm"], r["E_MPa"]): r["F_sim_N"] for r in rows}


def load_calibrated_E(results_dir=None):
    """
    Load per-specimen calibrated E from results/calibrate_E.csv
    (written by postprocess_exp_cantilever_E.py).
    Returns (E_mean_Pa, E_std_Pa).
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    path = os.path.join(results_dir, "calibrate_E.csv")
    E_vals = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            E_vals.append(float(r["E_mean_MPa"]) * 1e6)
    arr = np.array(E_vals)
    return float(arr.mean()), float(arr.std(ddof=1))
