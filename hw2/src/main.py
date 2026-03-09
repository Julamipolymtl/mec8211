"""
Main entry point for HW2 — MMS verification and physical problem.

Usage
-----
    python main.py path/to/params.json

The params.json file controls all physical, numerical, convergence, and
output parameters. See data/params_template.json for a reference template.
"""

import argparse
import os

from plots import (
    plot_concentration_profiles,
    plot_convergence,
    plot_mms,
    plot_sourceterm,
)
from convergence import convergence_study_spatial, convergence_study_temporal
from solver import DiffusionParams


def main():
    parser = argparse.ArgumentParser(
        description="Run HW2 diffusion solver and generate all figures."
    )
    parser.add_argument("params", help="Path to the JSON parameter file.")
    args = parser.parse_args()

    print(f"Loading parameters from: {args.params}")
    params = DiffusionParams.from_json(args.params)
    print(f"  D_eff={params.D_eff}, R={params.R}, Ce={params.Ce}, k={params.k}")
    print(f"  N_r={params.N_r}, N_t={params.N_t}, t_max={params.t_max:.2e}")
    print(f"  run_name='{params.run_name}'")

    # Ensure output directories exist
    base = os.path.join(os.path.dirname(__file__), "..")
    os.makedirs(os.path.join(base, "data",    params.run_name), exist_ok=True)
    os.makedirs(os.path.join(base, "results", params.run_name), exist_ok=True)

    if params.mms:
        print("\n>>> Plotting MMS solution and source term...")
        plot_mms(params)
        plot_sourceterm(params)

    if params.run_convergence:
        print("\n>>> Running spatial convergence study (MMS)...")
        results_space = convergence_study_spatial(params)
        plot_convergence(params, results_space, filename="convergence_spatial.png")
        print(f"  {'N_r':>6}  {'dr':>10}  {'L1':>10}  {'L2':>10}  {'Linf':>10}  {'p_L2':>6}")
        for i, N in enumerate(results_space["N"]):
            order = f"{results_space['order_L2'][i-1]:.3f}" if i > 0 else "—"
            print(f"  {N:>6}  {results_space['dr'][i]:>10.4e}  {results_space['L1'][i]:>10.3e}"
                  f"  {results_space['L2'][i]:>10.3e}  {results_space['Linf'][i]:>10.3e}  {order:>6}")

        print("\n>>> Running temporal convergence study (MMS)...")
        results_time = convergence_study_temporal(params)
        plot_convergence(params, results_time, ctime=True, filename="convergence_temporal.png")
        print(f"  {'N_t':>6}  {'dt':>10}  {'L1':>10}  {'L2':>10}  {'Linf':>10}  {'p_L2':>6}")
        for i, N in enumerate(results_time["N"]):
            order = f"{results_time['order_L2'][i-1]:.3f}" if i > 0 else "—"
            print(f"  {N:>6}  {results_time['dr'][i]:>10.4e}  {results_time['L1'][i]:>10.3e}"
                  f"  {results_time['L2'][i]:>10.3e}  {results_time['Linf'][i]:>10.3e}  {order:>6}")
    else:
        print("\n>>> Skipping convergence studies (run_convergence=false).")

    print("\n>>> Plotting physical concentration profile...")
    plot_concentration_profiles(params, filename="concentration_profile.png")

    print(f"\nDone. All figures saved to results/{params.run_name}/")


if __name__ == "__main__":
    main()
