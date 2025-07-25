import numpy as np
from HiggsML.systematics import systematics
from scipy import stats
from iminuit import Minuit
import pandas as pd
import matplotlib.pyplot as plt


"""
Task 1a : Counting Estimator
1.write the saved_info dictionary such that it contains the following keys
    1. beta
    2. gamma
2. Estimate the mu using the formula
    mu = (sum(score * weight) - beta) / gamma
3. return the mu and its uncertainty

Task 1b : Stat-Only Likelihood Estimator
1. Modify the estimation of mu such that it uses the likelihood function
    1. Write a function for the likelihood function which profiles over mu
    2. Use Minuit to minimize the NLL

Task 2 : Systematic Uncertainty
1. substitute the beta and gamma with the tes_fit and jes_fit functions
2. Write a function to likelihood function which profiles over mu, tes and jes
3. Use Minuit to minimize the NLL
4. return the mu and its uncertainty
"""


def compute_mu(score, weight, saved_info):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters
    """

    score = score.flatten() > 0.5
    score = score.astype(int)

    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    del_mu_sys = abs(0.0 * mu)
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def signal(xe, ns, mu, sigma):
    return ns * stats.norm(mu, sigma).cdf(xe)


def background(xe, nb, lambd):
    return nb * stats.expon(0.0, lambd).cdf(xe)


def total(xe, ns, mu, sigma, nb, lambd):
    return signal(xe, ns, mu, sigma) + background(xe, nb, lambd)


def extended_binned_nll(obs_counts, bin_edges, ns, mu, sigma, nb, lambd):
    # Calcule les comptes cumulés attendus par bin
    xe = bin_edges
    expected_cdf = total(xe, ns, mu, sigma, nb, lambd)

    # Comptes attendus dans chaque bin = différence entre les valeurs CDF successives
    expected_counts = np.diff(expected_cdf)

    # Sert à éviter log(0)
    expected_counts = np.clip(expected_counts, 1e-9, None)

    # Formula de la log-vraisemblance négative binnie étendue
    nll = np.sum(expected_counts - obs_counts * np.log(expected_counts))
    return nll


def plot_score_distributions(score, labels, weights=None, bins=50):
    """
    Affiche la distribution des scores pour le signal et le bruit de fond.
    Paramètres :
    - score : array (probabilité prédite par le modèle)
    - labels : array (0 = background, 1 = signal)
    - weights : array facultatif (poids des événements)
    - bins : nombre de bins de l'histogramme
    """
    score = score.flatten()
    labels = labels.flatten()

    # Séparer signal et bruit
    score_signal = score[labels == 1]
    score_background = score[labels == 0]

    weights_signal = weights[labels == 1] if weights is not None else None
    weights_background = weights[labels == 0] if weights is not None else None

    # Tracer les histogrammes
    plt.figure(figsize=(10, 6))
    plt.hist(
        score_background,
        bins=bins,
        weights=weights_background,
        alpha=0.6,
        label="Background",
        color="skyblue",
        density=True,
    )
    plt.hist(
        score_signal,
        bins=bins,
        weights=weights_signal,
        alpha=0.6,
        label="Signal",
        color="orange",
        density=True,
    )

    # Ligne verticale pour le seuil courant (0.5)
    plt.axvline(0.5, color="red", linestyle="--", label="Seuil = 0.5")

    plt.xlabel("Score du modèle")
    plt.ylabel("Distribution normalisée")
    plt.title("Distribution des scores du modèle pour signal et background")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_saved_info(model, holdout_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    score = model.predict(holdout_set["data"])

    from systematic_analysis import tes_fitter
    from systematic_analysis import jes_fitter

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
    score = score.astype(int)

    label = holdout_set["labels"]

    print("score shape after threshold", score.shape)

    gamma = np.sum(holdout_set["weights"] * score * label)

    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {
        "beta": beta,
        "gamma": gamma,
        "tes_fit": tes_fitter(model, holdout_set),
        "jes_fit": jes_fitter(model, holdout_set),
    }

    print("saved_info", saved_info)

    return saved_info


def compute_ams(s, b):
    if b <= 0:
        return 0
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))


def scan_threshold_ams(score, labels, weights):
    min_score = np.min(score)
    max_score = np.max(score)

    thresholds = np.linspace(min_score + 1e-4, max_score - 1e-4, 100)

    best_threshold = 0.5
    best_ams = 0
    for thresh in thresholds:
        selected = score.flatten() >= thresh
        s = np.sum(weights[selected & (labels == 1)])
        b = np.sum(weights[selected & (labels == 0)])
        ams = compute_ams(s, b)
        if ams > best_ams:
            best_ams = ams
            best_threshold = thresh
    return best_threshold, best_ams
