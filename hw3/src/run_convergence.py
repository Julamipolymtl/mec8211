from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import devoir3_lbm_accelerated as lbm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONV_DIR = RESULTS_DIR / "convergence"


def etude_convergence(seed, delta_p, poro, mean_fiber_d, std_d, dx_base, nx_base, raff):
    """Étude de convergence single-seed, géométrie fixée."""
    CONV_DIR.mkdir(parents=True, exist_ok=True)

    domaine = nx_base * dx_base

    dx_vals = []
    k_vals = []

    for r in raff:
        dx = dx_base / r
        nx = int(domaine / dx)

        nom_fichier = f"fiber_s{seed}_nx{nx}.tiff"
        print(f"\nnx = {nx}, dx = {dx:.2e}")

        d_eq = lbm.Generate_sample(seed, nom_fichier, mean_fiber_d, std_d, poro, nx, dx)
        k_val = lbm.LBM(nom_fichier, nx, delta_p, dx, d_eq)

        dx_vals.append(dx)
        k_vals.append(k_val)

        sauvegarder_figures(f"s{seed}_nx{nx}")
        plt.close("all")

    dx_vals = np.array(dx_vals)
    k_vals = np.array(k_vals)

    analyser_resultats(dx_vals, k_vals, raff, domaine, seed)


def sauvegarder_figures(prefixe):
    """Sauvegarde les figures ouvertes."""
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig.savefig(
            CONV_DIR / f"{prefixe}_fig{i}.png",
            dpi=300,
            bbox_inches="tight",
        )


def analyser_resultats(dx_vals, k_vals, raff, domaine, seed):
    """Analyse des résultats et estimation de l'ordre de convergence."""
    k_ref = k_vals[-1]
    erreur = np.abs((k_ref - k_vals) / k_ref)

    print("\nErreurs relatives :", erreur)

    # Ordre par fit log-log (on exclut le point de reference)
    mask = (erreur > 0) & np.isfinite(erreur)
    ordre = None
    if mask.sum() >= 2:
        p = np.polyfit(np.log(dx_vals[mask]), np.log(erreur[mask]), 1)
        ordre = p[0]
        print(f"\nOrdre de convergence estimé : {ordre:.4f}")
    else:
        print("\nPas assez de points valides pour estimer l'ordre.")

    # u_num par extrapolation de Richardson (r=2 entre maillages consécutifs)
    r = raff[-1] / raff[-2]
    u_num = None
    if ordre is not None and ordre > 0:
        u_num = abs(k_vals[-1] - k_vals[-2]) / (r**ordre - 1)
        print(f"u_num (Richardson) : {u_num:.4f} um^2")
        print(f"u_num / k_ref      : {u_num / k_ref:.4f}")
    else:
        print("u_num non calculé (ordre invalide).")

    tracer_resultats(dx_vals, k_vals, erreur, ordre)
    enregistrer_donnees(dx_vals, k_vals, erreur, ordre, u_num, domaine, seed)


def tracer_resultats(dx_vals, k_vals, erreur, ordre):
    """Trace les courbes de convergence."""
    plt.figure()
    plt.loglog(dx_vals[:-1], erreur[:-1], "o-", label="Erreur relative")
    if ordre is not None:
        ref = erreur[0] * (dx_vals / dx_vals[0]) ** 2
        plt.loglog(dx_vals[:-1], ref[:-1], "k--", label="Ordre 2 (ref.)")
    plt.xlabel("dx [m]")
    plt.ylabel("Erreur relative sur k")
    plt.title("Convergence spatiale de k")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig(CONV_DIR / "convergence.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.semilogx(dx_vals, k_vals, "o-")
    plt.xlabel("dx [m]")
    plt.ylabel("Perméabilité k [µm²]")
    plt.title("Perméabilité en fonction de dx")
    plt.grid(True)
    plt.savefig(CONV_DIR / "permeabilite.png", dpi=300, bbox_inches="tight")


def enregistrer_donnees(dx_vals, k_vals, erreur, ordre, u_num, domaine, seed):
    """Sauvegarde les résultats dans un fichier texte."""
    nom_fichier = CONV_DIR / "resultats_convergence.dat"
    k_ref = k_vals[-1]

    with open(nom_fichier, "w") as f:
        f.write(f"# Seed : {seed}\n")
        f.write("# Nx\t dx [m]\t\t k [um^2]\t erreur_rel\n")
        for i, dx in enumerate(dx_vals):
            nx = int(domaine / dx)
            f.write(f"{nx}\t {dx:.3e}\t {k_vals[i]:.6f}\t {erreur[i]:.6f}\n")
        if ordre is not None:
            f.write(f"\n# Ordre de convergence estimé : {ordre:.4f}\n")
        if u_num is not None:
            f.write(f"# u_num (Richardson)          : {u_num:.6f} um^2\n")
            f.write(f"# u_num / k_ref               : {u_num / k_ref:.6f}\n")

    print(f"\nDonnées sauvegardées dans {nom_fichier}")


if __name__ == "__main__":
    etude_convergence(
        seed=101,
        delta_p=0.1,
        poro=0.9,
        mean_fiber_d=12.5,
        std_d=2.85,
        dx_base=2e-6,
        nx_base=100,
        raff=[1, 2, 4],
    )
