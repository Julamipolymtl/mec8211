"""Entry point: run both FD schemes, generate all figures and convergence tables."""

import numpy as np

from convergence import convergence_study
from plotting import plot_comparison, plot_concentration_profiles, plot_convergence


def print_convergence_table(results, scheme_label):
    """Print a formatted convergence table to stdout."""
    print(f"\n{'='*72}")
    print(f"Convergence Table — {scheme_label}")
    print(f"{'='*72}")
    header = f"{'N':>6s}  {'dr':>10s}  {'L1':>12s}  {'L2':>12s}  {'Linf':>12s}  {'p(L2)':>7s}"
    print(header)
    print("-" * len(header))

    for k, N in enumerate(results["N"]):
        order_str = f"{results['order_L2'][k-1]:7.2f}" if k > 0 else "    ---"
        print(
            f"{N:6d}  {results['dr'][k]:10.6f}  "
            f"{results['L1'][k]:12.4e}  {results['L2'][k]:12.4e}  "
            f"{results['Linf'][k]:12.4e}  {order_str}"
        )
    print()


def main():
    print("=" * 72)
    print("HW1: Finite Difference Solver for Salt Diffusion in Concrete Pillar")
    print("=" * 72)

    # --- Scheme 1: Forward difference (Q.C/D) ---
    print("\n>>> Scheme 1: Forward difference for dC/dr")
    results_fwd = convergence_study(scheme="forward")
    print_convergence_table(results_fwd, "Scheme 1 (forward)")

    print("  Generating plots...")
    plot_concentration_profiles(N=65, scheme="forward", filename="concentration_forward.png")
    plot_convergence(results_fwd, "Scheme 1 (forward)", "convergence_forward.png")

    # --- Scheme 2: Central difference (Q.E) ---
    print("\n>>> Scheme 2: Central difference for dC/dr")
    results_ctr = convergence_study(scheme="central")
    print_convergence_table(results_ctr, "Scheme 2 (central)")

    print("  Generating plots...")
    plot_concentration_profiles(N=65, scheme="central", filename="concentration_central.png")
    plot_convergence(results_ctr, "Scheme 2 (central)", "convergence_central.png")

    # --- Comparison plot ---
    print("\n>>> Generating comparison plot (both schemes vs analytical)...")
    plot_comparison(N=65, filename="comparison_both_schemes.png")

    print("\nDone. All results saved to hw1/results/")


if __name__ == "__main__":
    main()
