"""
Extract three-point bending SRQs from raw test-machine output files and
write data/experimental.csv.

Raw data layout
---------------
  data/vv raw data/<config>/   one sub-folder per loading condition (A-F)
    1.txt ... 6.txt             one file per specimen

Each .txt file is a plain tab-separated file (Mach-1 output):
  time [s], position [mm], force [gf]

The test protocol repeats loading/unloading cycles to the target displacement.
The SRQ per specimen is the mean force [N] across all displacement peaks
where force >= F_THRESHOLD_GF (10 gf), converted from gf to N.

Config-to-condition mapping (A-F defined at the top of this file):
  update CONFIGS if the folder naming changes.

Run with:  python scripts/postprocess_exp_3pt_bending.py
"""

import os
import csv
import numpy as np
from scipy.signal import find_peaks

RAW_DATA  = os.path.join(os.path.dirname(__file__), "..", "data", "experimental_raw")
OUT_CSV   = os.path.join(os.path.dirname(__file__), "..", "data", "experimental.csv")

CONV_GF_TO_N  = 0.00980665
F_THRESHOLD_GF = 10.0   # minimum peak force to count as a valid loading cycle [gf]

# Config folder -> (L_span_mm, delta_mm)
CONFIGS = {
    "A": (60.0, 5.0),
    "B": (60.0, 3.0),
    "C": (60.0, 4.0),
    "D": (40.0, 5.0),
    "E": (40.0, 4.0),
    "F": (40.0, 3.0),
}


def load_mach1(path):
    """Return (time, disp_mm, force_gf) arrays from a Mach-1 .txt file."""
    rows = []
    skip_next = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s == "<DATA>":
                skip_next = True
                continue
            if s == "<END DATA>":
                break
            if skip_next:
                skip_next = False
                continue
            parts = s.split("\t")
            if len(parts) == 3:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    pass
    arr = np.array(rows).T
    return arr[0], arr[1], arr[2]


def srq_from_file(path):
    """
    Return (delta_mm, F_N) for one specimen file.
    F_N is the mean force across all valid displacement peaks.
    Returns None if no valid peaks found.
    """
    t, disp, force = load_mach1(path)
    peaks, _ = find_peaks(disp)
    peak_forces = [force[i] for i in peaks if force[i] >= F_THRESHOLD_GF]
    if not peak_forces:
        return None
    delta_mm = disp.max() - disp[0]
    F_N = np.mean(peak_forces) * CONV_GF_TO_N
    return delta_mm, F_N


if __name__ == "__main__":
    print("=" * 60)
    print("Three-point bending SRQ extraction from raw Mach-1 files")
    print("=" * 60)

    csv_rows = []

    for cfg in sorted(CONFIGS):
        L_mm, delta_mm = CONFIGS[cfg]
        cfg_path = os.path.join(RAW_DATA, cfg)
        if not os.path.isdir(cfg_path):
            print(f"\nConfig {cfg}: folder not found, skipping")
            continue

        files = sorted(f for f in os.listdir(cfg_path) if f.endswith(".txt"))
        print(f"\nConfig {cfg}  (L={L_mm:.0f} mm, delta={delta_mm:.0f} mm):")

        for fname in files:
            sid = int(os.path.splitext(fname)[0].lstrip("no"))
            result = srq_from_file(os.path.join(cfg_path, fname))
            if result is None:
                print(f"  {fname}: SKIPPED (no valid peaks)")
                continue
            meas_delta, F_N = result
            print(f"  {fname}: delta={meas_delta:.3f} mm  F={F_N:.5f} N")
            csv_rows.append({
                "L_span_mm":   L_mm,
                "delta_mm":    delta_mm,
                "specimen_id": sid,
                "F_exp_N":     F_N,
            })

    csv_rows.sort(key=lambda r: (r["L_span_mm"], r["delta_mm"], r["specimen_id"]))

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["L_span_mm", "delta_mm", "specimen_id", "F_exp_N"])
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\nSaved {OUT_CSV}  ({len(csv_rows)} rows)")
