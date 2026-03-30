from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ============================================================
# Donnees experimentales (Tableau 1)
# ============================================================
D = 80.6          # Permeabilite mediane mesuree [um^2]
SIGMA_REPRO = 14.7  # Ecart-type reproductibilite [um^2]
SIGMA_PERMEA = 10.0  # Incertitude permeametre [um^2]
K_COVERAGE = 2    # k = 2 pour intervalle 95.4% (ASME V&V20)


# ============================================================
# Lecture des resultats Part A (convergence)
# ============================================================
def lire_convergence():
    """Lit les k_moy et dx depuis le fichier de convergence."""
    fichier = RESULTS_DIR / "convergence" / "resultats_convergence.txt"
    dx_vals = []
    k_moy_vals = []
    with open(fichier, encoding="latin-1") as f:
        for ligne in f:
            if ligne.startswith("#") or ligne.strip() == "":
                continue
            parts = ligne.split()
            dx_vals.append(float(parts[1]))
            k_moy_vals.append(float(parts[2]))
    return np.array(dx_vals), np.array(k_moy_vals)


# ============================================================
# Lecture des resultats Part B (Monte Carlo)
# ============================================================
def lire_montecarlo():
    """Lit median_k et u_input depuis le fichier de synthese MC."""
    fichier = RESULTS_DIR / "montecarlo" / "summary.txt"
    resultats = {}
    with open(fichier) as f:
        for ligne in f:
            if "=" in ligne:
                cle, val = ligne.split("=")
                resultats[cle.strip()] = float(val.strip())
    return resultats


# ============================================================

def main():
    val_dir = RESULTS_DIR / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    # --- Lecture des donnees ---
    dx_vals, k_moy = lire_convergence()
    mc = lire_montecarlo()

    S = mc["median_k"]
    u_input_minus = mc["u_input_minus"]
    u_input_plus = mc["u_input_plus"]

    # ============================================================
    # Partie C : u_D
    # ============================================================
    u_D = np.sqrt(SIGMA_REPRO**2 + SIGMA_PERMEA**2)

    # ============================================================
    # Partie D : E = S - D
    # ============================================================
    E = S - D

    # ============================================================
    # Partie E : u_num, u_val, delta_model
    # ============================================================

    # Ordre de convergence par fit log-log (erreur relative vs dx)
    k_ref = k_moy[-1]
    erreur = np.abs((k_ref - k_moy) / k_ref)
    mask = (erreur > 0) & np.isfinite(erreur)
    p_hat = np.polyfit(np.log(dx_vals[mask]), np.log(erreur[mask]), 1)[0]
    P_THEORIQUE = 2.0  # ordre theorique LBM en espace

    # p clamp selon V&V20 : p = min(max(0.5, p_hat), p_theorique)
    p_clamp = min(max(0.5, p_hat), P_THEORIQUE)

    # u_num par GCI (facteur de securite 3, deux maillages les plus fins)
    r = dx_vals[-2] / dx_vals[-1]
    GCI_FACTOR = 3.0
    u_num = GCI_FACTOR / (r**p_clamp - 1) * abs(k_moy[-1] - k_moy[-2])

    # u_input : valeur conservative (max des deux bornes asymetriques)
    u_input = max(u_input_minus, u_input_plus)

    # u_val
    u_val = np.sqrt(u_num**2 + u_input**2 + u_D**2)
    U_val = K_COVERAGE * u_val

    # Estimation directe de delta_model (hypothese : biais numerique et input
    # s'opposent a E, biais experimental va dans le sens de E)
    delta_model_est = E - u_num - u_input + u_D

    # Analyse V&V20 par cas (C = 7 selon notes de cours)
    C = 7.0
    abs_E = abs(E)
    if abs_E > C * U_val:
        cas = 1
        conclusion = f"Cas 1 : |E| > C*U_val -> delta_model = E = {E:.4f} um^2"
    elif abs_E >= U_val:
        cas = "3a"
        conclusion = (f"Cas 3a : C*U_val >= |E| >= U_val\n"
                      f"  |delta_model| < |E| + U_val = {abs_E + U_val:.4f} um^2\n"
                      f"  signe(delta_model) = signe(E) < 0 (sous-estimation)")
    elif abs_E >= U_val / C:
        cas = "3b"
        conclusion = (f"Cas 3b : U_val >= |E| >= U_val/C\n"
                      f"  |delta_model| < |E| + U_val = {abs_E + U_val:.4f} um^2\n"
                      f"  signe(delta_model) inconnu")
    else:
        cas = 2
        conclusion = (f"Cas 2 : |E| < U_val/C\n"
                      f"  |delta_model| < U_val = {U_val:.4f} um^2\n"
                      f"  signe(delta_model) inconnu")

    # ============================================================
    # Affichage
    # ============================================================
    lignes = [
        "=" * 55,
        "VALIDATION ASME V&V20",
        "=" * 55,
        "",
        "--- Partie C : incertitude experimentale ---",
        f"  sigma_repro    = {SIGMA_REPRO:.2f} um^2",
        f"  sigma_permea   = {SIGMA_PERMEA:.2f} um^2",
        f"  u_D            = {u_D:.4f} um^2",
        "",
        "--- Partie D : erreur de simulation ---",
        f"  S (median num) = {S:.4f} um^2",
        f"  D (median exp) = {D:.4f} um^2",
        f"  E = S - D      = {E:.4f} um^2",
        "",
        "--- Partie E : validation ---",
        f"  p_hat (observe) = {p_hat:.4f}",
        f"  p (clamp)       = {p_clamp:.4f}",
        f"  r               = {r:.4f}",
        f"  u_num (GCI)     = {u_num:.4f} um^2",
        f"  u_input         = {u_input:.4f} um^2  (max bornes asymetriques)",
        f"  u_D             = {u_D:.4f} um^2",
        f"  u_val           = {u_val:.4f} um^2",
        f"  U_val = k*u_val = {U_val:.4f} um^2  (k={K_COVERAGE})",
        f"  C*U_val         = {C*U_val:.4f} um^2  (C={C:.0f})",
        f"  |E|             = {abs_E:.4f} um^2",
        "",
        f"  delta_model (estimation directe) = {delta_model_est:.4f} um^2",
        "",
        f"  {conclusion}",
        "=" * 55,
    ]

    for l in lignes:
        print(l)

    with open(val_dir / "summary.txt", "w") as f:
        f.write("\n".join(lignes) + "\n")

    print(f"\nResultats sauvegardes dans {val_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
