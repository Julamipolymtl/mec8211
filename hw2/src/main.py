"""
Main script to generate all relevant figures for HW1.
"""

from plots import *

print("Running main.py to generate all figures and convergence tables...")

print("\n>>> Plotting concentration profile")
plot_concentration_profiles(N=11, filename="concentration_forward.png")
print("\n>>> Plotting MMS graphs")
plot_mms()
plot_sourceterm()

print("\n>>> Generating convergence plots (Solver vs MMS)...")
results_radius = convergence_study()
plot_convergence(results_radius, filename="convergence_radius.png")
results_time = convergence_study_time()
plot_convergence(results_time, ctime=True, filename="convergence_time.png")
