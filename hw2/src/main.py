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

print("\n>>> Generating convergence plot (Solver vs MMS)...")
results_radius = convergence_study()
plot_convergence(results_radius, filename="convergence_radius.png")
results_time = convergence_study_time()
plot_convergence(results_time, filename="convergence_time.png")
