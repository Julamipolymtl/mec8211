"""
Main script to generate all relevant figures for HW1.
"""

from plots import *

print("Running main.py to generate all figures and convergence tables...")

print("\n>>> Plotting MMS grah")
#plot_concentration_profiles(N=5, filename="concentration_forward.png")
#plot_convergence(filename="convergence_forward.png")
plot_mms()
plot_sourceterm()

print("\n>>> Generating comparison plot (both schemes vs analytical)...")
plot_comparison(N=5, filename="comparison_both_schemes.png")
