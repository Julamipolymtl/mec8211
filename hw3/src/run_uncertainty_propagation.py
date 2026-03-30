# Part B: estimation de u_input par Monte Carlo
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image

import devoir3_lbm_accelerated as lbm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main(delta_p, mean_fiber_d, std_d, poro_mean, poro_std, dx, nx, n_samples):
    mc_dir = RESULTS_DIR / "montecarlo"
    mc_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    porosities = rng.normal(poro_mean, poro_std, n_samples)
    porosities = np.clip(porosities, 0.85, 0.95)

    permeabilities = []
    poro_eff_vals = []

    for i, poro in enumerate(porosities):
        print(f"\n===== Sample {i+1}/{n_samples}, porosity = {poro:.5f} =====")

        filename = str(mc_dir / f"fiber_mc_{i}.tiff")

        d_eq = lbm.Generate_sample(0, filename, mean_fiber_d, std_d, poro, nx, dx)

        img = np.array(Image.open(filename)).astype(bool)
        poro_eff = 1.0 - img.sum() / img.size
        poro_eff_vals.append(poro_eff)
        print(f"  poro demandee = {poro:.5f}, poro effective = {poro_eff:.5f}, "
              f"erreur = {abs(poro_eff - poro):.5f}")

        k_val = lbm.LBM(filename, nx, delta_p, dx, d_eq)
        permeabilities.append(k_val)
        plt.close("all")

        np.savetxt(
            mc_dir / "permeabilities.txt",
            np.column_stack([
                porosities[:len(permeabilities)],
                poro_eff_vals,
                permeabilities,
            ]),
            header="porosity_demandee\tporosity_effective\tk_micron2",
            fmt="%.6f",
        )

        print(f"  k = {k_val:.4f} um2")

    permeabilities = np.array(permeabilities)
    poro_eff_vals = np.array(poro_eff_vals)
    poro_errors = np.abs(poro_eff_vals - porosities)

    print(f"\nVerification porosite effective :")
    print(f"  erreur max  = {poro_errors.max():.5f}")
    print(f"  erreur moy  = {poro_errors.mean():.5f}")

    # Analyse log-normale
    log_k = np.log(permeabilities)
    mu_log = np.mean(log_k)
    sigma_log = np.std(log_k, ddof=1)

    median_k = np.exp(mu_log)
    fvg = np.exp(sigma_log)

    u_input_minus = median_k - median_k / fvg
    u_input_plus = median_k * fvg - median_k

    print("\n" + "=" * 50)
    print("RESULTATS")
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

    with open(mc_dir / "summary.txt", "w") as f:
        f.write(f"N_samples = {len(permeabilities)}\n")
        f.write(f"mu_log = {mu_log:.6f}\n")
        f.write(f"sigma_log = {sigma_log:.6f}\n")
        f.write(f"median_k = {median_k:.6f}\n")
        f.write(f"FVG = {fvg:.6f}\n")
        f.write(f"u_input_minus = {u_input_minus:.6f}\n")
        f.write(f"u_input_plus = {u_input_plus:.6f}\n")
        f.write(f"poro_erreur_max = {poro_errors.max():.6f}\n")
        f.write(f"poro_erreur_moy = {poro_errors.mean():.6f}\n")

    # PDF + CDF
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(permeabilities, bins=15, density=True, alpha=0.7, label="Histogramme")
    x_plot = np.linspace(permeabilities.min() * 0.8, permeabilities.max() * 1.2, 200)
    pdf_fit = stats.lognorm.pdf(x_plot, s=sigma_log, scale=np.exp(mu_log))
    ax.plot(x_plot, pdf_fit, "r-", lw=2, label="Fit log-normal")
    ax.axvline(median_k, color="k", ls="--", label=f"Mediane = {median_k:.1f}")
    ax.set_xlabel(r"Permeabilite k [$\mu m^2$]")
    ax.set_ylabel("Densite de probabilite")
    ax.set_title("PDF des permeabilites (Monte Carlo)")
    ax.legend()

    ax = axes[1]
    k_sorted = np.sort(permeabilities)
    cdf_emp = np.arange(1, len(k_sorted) + 1) / len(k_sorted)
    ax.step(k_sorted, cdf_emp, where="post", label="CDF empirique")
    cdf_fit = stats.lognorm.cdf(x_plot, s=sigma_log, scale=np.exp(mu_log))
    ax.plot(x_plot, cdf_fit, "r-", lw=2, label="Fit log-normal")
    ax.axvline(median_k, color="k", ls="--", label=f"Mediane = {median_k:.1f}")
    ax.set_xlabel(r"Permeabilite k [$\mu m^2$]")
    ax.set_ylabel("CDF")
    ax.set_title("CDF des permeabilites (Monte Carlo)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(mc_dir / "pdf_cdf.png", dpi=300)
    print(f"\nFigures sauvegardees dans {mc_dir}")

    plt.figure()
    plt.scatter(porosities, permeabilities, alpha=0.7)
    plt.xlabel("Porosite")
    plt.ylabel(r"Permeabilite k [$\mu m^2$]")
    plt.title("Permeabilite vs Porosite (Monte Carlo)")
    plt.grid(True)
    plt.savefig(mc_dir / "poro_vs_k.png", dpi=300)


if __name__ == "__main__":
    main(
        delta_p=0.1,
        mean_fiber_d=12.5,
        std_d=2.85,
        poro_mean=0.900,
        poro_std=7.50e-3,
        dx=1e-6,       # maillage fin d'apres Part A (nx=200)
        nx=200,
        n_samples=50,
    )
