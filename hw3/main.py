import os

import numpy as np

import matplotlib.pyplot as plt

import devoir3_lbm_accelerated as lbm


# ============================================================
# PARAMETRES
# ============================================================

SEEDS = [101, 351, 651, 1001, 1501]
DELTA_P = 0.1
PORO = 0.9
MEAN_FIBER_D = 12.5
STD_D = 2.85

DX_BASE = 2e-6
NX_BASE = 100

# Garder le domaine constant
DOMAINE = NX_BASE * DX_BASE

# Raffinnement de maillage constant
RAFF = [0.5, 0.75, 1, 1.5, 2, 4]

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def etude_convergence():
    """
    Effectue une étude de convergence robuste avec moyenne
    sur plusieurs seeds.
    """
    os.makedirs("figures", exist_ok=True)

    k_all = []

    for seed in SEEDS:

        print(f"\n===== SEED {seed} =====")

        k_seed = []

        for raffinement in RAFF:

            dx = DX_BASE / raffinement
            nx = int(DOMAINE / dx)

            nom_fichier = f"fiber_s{seed}_nx{nx}.tiff"

            print(f"nx = {nx}, dx = {dx:.2e}")

            d_eq = lbm.Generate_sample(
                seed,
                nom_fichier,
                MEAN_FIBER_D,
                STD_D,
                PORO,
                nx,
                dx,
            )

            k_val = lbm.LBM(
                nom_fichier,
                nx,
                DELTA_P,
                dx,
                d_eq,
            )

            k_seed.append(k_val)

            sauvegarder_figures(f"s{seed}_nx{nx}")
            plt.close("all")

        k_all.append(k_seed)

    k_all = np.array(k_all)

    analyser_resultats(k_all)


# ============================================================
# SAUVEGARDE FIGURES
# ============================================================

def sauvegarder_figures(prefixe):
    """
    Sauvegarde toutes les figures ouvertes avec un nom personnalisé.
    """
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig.savefig(
            f"figures/{prefixe}_fig{i}.png",
            dpi=300,
            bbox_inches="tight",
        )


# ============================================================
# ANALYSE
# ============================================================

def analyser_resultats(k_all):
    """
    Analyse statistique + estimation ordre de convergence.
    """
    dx_vals = np.array([DX_BASE / r for r in RAFF])

    k_moy = np.mean(k_all, axis=0)

    k_ref = k_moy[-1]
    # Enregistrer les données dans un fichier texte
    enregistrer_donnees(k_all, dx_vals)
    # ========================================================
    # ERREUR ROBUSTE (corrigée)
    # ========================================================

    erreur = np.abs((k_ref - k_moy) / k_ref)

    print("\nErreurs relatives :", erreur)

    # ========================================================
    # PLOT AVEC BARRES D'ERREUR
    # ========================================================

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dx")
    plt.ylabel("Erreur relative")
    plt.title("Convergence de k")
    plt.loglog(dx_vals, erreur, "o-", label="erreur")
    plt.grid(True)
    plt.savefig("figures/convergence.png", dpi=300)
    plt.show()

    plt.figure()
    plt.semilogx(dx_vals, k_moy, "o-", label="k moyen")

    plt.xlabel("dx")
    plt.ylabel("Perméabilité k [µm²]")
    plt.title("Perméabilité en fonction de dx")
    plt.grid(True)
    plt.legend()

    plt.savefig("figures/permeabilite.png", dpi=300)
    plt.show()

    # ========================================================
    # FIT ROBUSTE (sans log(0))
    # ========================================================

    mask = (erreur > 0) & np.isfinite(erreur)

    dx_fit = dx_vals[mask]
    err_fit = erreur[mask]

    if len(err_fit) >= 2:
        p = np.polyfit(
            np.log(dx_fit),
            np.log(err_fit),
            1,
        )

        ordre = p[0]
        print("\nOrdre de convergence estimé :", ordre)

    else:
        print("\nPas assez de points valides pour estimer l'ordre.")

    # ========================================================
    # BIAIS (INTERVALLE UNILATÉRAL)
    # ========================================================

    k_max = np.max(k_all, axis=0)
    biais = (k_max - k_moy) / k_ref

    print("\nBiais (borne supérieure) :", biais)

# ========================================================

def enregistrer_donnees(k_all, dx_vals, nom_fichier="resultats_convergence.txt"):
    """
    Enregistre les données de perméabilité et d'erreur dans un fichier texte.

    Paramètres
    ----------
    k_all : ndarray
        Tableau (seeds x raffinement) des perméabilités
    dx_vals : ndarray
        Tableau des tailles de mailles correspondantes
    nom_fichier : str
        Nom du fichier texte de sortie
    """

    k_moy = np.mean(k_all, axis=0)
    k_ref = k_moy[-1]
    erreur = np.abs((k_ref - k_moy) / k_ref)
    k_max = np.max(k_all, axis=0)
    biais = (k_max - k_moy) / k_ref

    with open(nom_fichier, "w") as f:
        f.write("# Nx\t dx [m]\t k_moy [µm2]\t  "
                "erreur_rel\t biais\n")
        for i, dx in enumerate(dx_vals):
            nx = int(dx_vals[-1] * len(dx_vals) / dx)  # estimation Nx
            f.write(f"{nx}\t {dx:.3e}\t {k_moy[i]:.6f}\t "
                    f"{erreur[i]:.6f}\t {biais[i]:.6f}\n")

    print(f"✔ Données sauvegardées dans {nom_fichier}")
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    etude_convergence()