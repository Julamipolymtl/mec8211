# Part B: estimation de u_input par Monte Carlo
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import devoir3_lbm_accelerated as lbm

# ============================================================
# PARAMETERS
# ============================================================
DELTA_P = 0.1
MEAN_FIBER_D = 12.5
STD_D = 2.85

PORO_MEAN = 0.900
PORO_STD = 7.50e-3

# Fine mesh (use NX=150, dx=1.33e-6 as a reasonable compromise)
DX = 1.333e-6
NX = 150

N_SAMPLES = 50

# ============================================================
# MONTE CARLO
# ============================================================
os.makedirs("mc_results", exist_ok=True)

rng = np.random.default_rng(42)
porosities = rng.normal(PORO_MEAN, PORO_STD, N_SAMPLES)
porosities = np.clip(porosities, 0.85, 0.95)

permeabilities = []

for i, poro in enumerate(porosities):
    print(f"\n===== Sample {i+1}/{N_SAMPLES}, porosity = {poro:.5f} =====")

    filename = f"mc_results/fiber_mc_{i}.tiff"

    d_eq = lbm.Generate_sample(
        0,
        filename,
        MEAN_FIBER_D,
        STD_D,
        poro,
        NX,
        DX,
    )

    k_val = lbm.LBM(
        filename,
        NX,
        DELTA_P,
        DX,
        d_eq,
    )

    permeabilities.append(k_val)
    plt.close("all")

    np.savetxt(
        "mc_results/permeabilities.txt",
        np.column_stack([porosities[:len(permeabilities)], permeabilities]),
        header="porosity\tk_micron2",
        fmt="%.6f",
    )

    print(f"  k = {k_val:.4f} um2")

permeabilities = np.array(permeabilities)

# ============================================================
# ANALYSIS
# ============================================================
log_k = np.log(permeabilities)
mu_log = np.mean(log_k)
sigma_log = np.std(log_k, ddof=1)

median_k = np.exp(mu_log)
fvg = np.exp(sigma_log)

u_input_minus = median_k - median_k / fvg
u_input_plus = median_k * fvg - median_k

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"N samples:        {len(permeabilities)}")
print(f"mu_log:           {mu_log:.4f}")
print(f"sigma_log:        {sigma_log:.4f}")
print(f"Median k:         {median_k:.4f} um2")
print(f"FVG:              {fvg:.4f}")
print(f"u_input- :        {u_input_minus:.4f} um2")
print(f"u_input+ :        {u_input_plus:.4f} um2")
print(f"Mean k:           {np.mean(permeabilities):.4f} um2")
print(f"Std k:            {np.std(permeabilities, ddof=1):.4f} um2")

with open("mc_results/summary.txt", "w") as f:
    f.write(f"N_samples = {len(permeabilities)}\n")
    f.write(f"mu_log = {mu_log:.6f}\n")
    f.write(f"sigma_log = {sigma_log:.6f}\n")
    f.write(f"median_k = {median_k:.6f}\n")
    f.write(f"FVG = {fvg:.6f}\n")
    f.write(f"u_input_minus = {u_input_minus:.6f}\n")
    f.write(f"u_input_plus = {u_input_plus:.6f}\n")

# --- PDF + CDF plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(permeabilities, bins=15, density=True, alpha=0.7, label="Histogram")
x_plot = np.linspace(permeabilities.min() * 0.8, permeabilities.max() * 1.2, 200)
pdf_fit = stats.lognorm.pdf(x_plot, s=sigma_log, scale=np.exp(mu_log))
ax.plot(x_plot, pdf_fit, "r-", lw=2, label="Log-normal fit")
ax.axvline(median_k, color="k", ls="--", label=f"Median = {median_k:.1f}")
ax.set_xlabel("Perméabilité k [$\mu m^2$]")
ax.set_ylabel("Densité de probabilité")
ax.set_title("PDF des perméabilités (Monte Carlo)")
ax.legend()

ax = axes[1]
k_sorted = np.sort(permeabilities)
cdf_emp = np.arange(1, len(k_sorted) + 1) / len(k_sorted)
ax.step(k_sorted, cdf_emp, where="post", label="Empirical CDF")
cdf_fit = stats.lognorm.cdf(x_plot, s=sigma_log, scale=np.exp(mu_log))
ax.plot(x_plot, cdf_fit, "r-", lw=2, label="Log-normal fit")
ax.axvline(median_k, color="k", ls="--", label=f"Median = {median_k:.1f}")
ax.set_xlabel("Perméabilité k [$\mu m^2$]")
ax.set_ylabel("CDF")
ax.set_title("CDF des perméabilités (Monte Carlo)")
ax.legend()

plt.tight_layout()
plt.savefig("mc_results/pdf_cdf.png", dpi=300)
print("\nPlots saved to mc_results/pdf_cdf.png")

# --- Porosity vs permeability scatter ---
plt.figure()
plt.scatter(porosities, permeabilities, alpha=0.7)
plt.xlabel("Porosité")
plt.ylabel("Perméabilité k [$\mu m^2$]")
plt.title("Perméabilité vs Porosité (Monte Carlo)")
plt.grid(True)
plt.savefig("mc_results/poro_vs_k.png", dpi=300)
print("Scatter plot saved to mc_results/poro_vs_k.png")
