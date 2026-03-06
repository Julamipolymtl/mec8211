"""
Main script to generate all relevant figures for HW1.
"""

from plots import *

print("Running main.py to generate all figures and convergence tables...")

print("\n>>> Scheme 1: Forward difference for dC/dr")
plot_concentration_profiles(N=5, filename="concentration_forward.png")
plot_convergence(filename="convergence_forward.png")

print("\n>>> Generating comparison plot (both schemes vs analytical)...")
plot_comparison(N=5, filename="comparison_both_schemes.png")
