import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "/Users/lucasdesgranges/Documents/EI - Higgs/Higgs_collaboration_B/blackSwan_data/blackSwan_data.parquet"
df = pd.read_parquet(file_path)


def four_momentum(pt, eta, phi):
    """Calcule le quadrivecteur (Px, Py, Pz, E=Pt*cosh(eta))"""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = pt * np.cosh(eta)  # Masse négligée comme dans le doc
    return np.array([px, py, pz, e])


def apply_systematic_bias(event, alpha_tes, alpha_jes, et_soft_sigma):
    # Reconstituer 4-momenta
    P_had = four_momentum(
        event["PRI_had_pt"], event["PRI_had_eta"], event["PRI_had_phi"]
    )
    P_jet_leading = four_momentum(
        event["PRI_jet_leading_pt"],
        event["PRI_jet_leading_eta"],
        event["PRI_jet_leading_phi"],
    )
    P_jet_subleading = four_momentum(
        event["PRI_jet_subleading_pt"],
        event["PRI_jet_subleading_eta"],
        event["PRI_jet_subleading_phi"],
    )

    # Missing ET vector (transverse only)
    Px_MET = event["PRI_met"] * np.cos(event["PRI_met_phi"])
    Py_MET = event["PRI_met"] * np.sin(event["PRI_met_phi"])
    P_MET = np.array(
        [Px_MET, Py_MET, 0, event["PRI_met"]]
    )  # 4ème composante arbitraire ici

    # Appliquer biais multiplicatifs
    P_had_biased = alpha_tes * P_had
    P_jet_leading_biased = alpha_jes * P_jet_leading
    P_jet_subleading_biased = alpha_jes * P_jet_subleading

    # Ajouter bruit gaussien 2D pour soft_met (Px, Py)
    soft_noise = np.random.normal(0, et_soft_sigma, size=2)
    P_MET_biased = P_MET.copy()
    P_MET_biased[0] += (
        soft_noise[0]
        + (1 - alpha_tes) * P_had[0]
        + (1 - alpha_jes) * P_jet_leading[0]
        + (1 - alpha_jes) * P_jet_subleading[0]
    )
    P_MET_biased[1] += (
        soft_noise[1]
        + (1 - alpha_tes) * P_had[1]
        + (1 - alpha_jes) * P_jet_leading[1]
        + (1 - alpha_jes) * P_jet_subleading[1]
    )

    # Recalculer les features scalaires biaisés (exemple pour had_pt)
    PRI_had_pt_biased = np.linalg.norm(P_had_biased[:2])  # transverse momentum biased
    PRI_jet_leading_pt_biased = np.linalg.norm(P_jet_leading_biased[:2])
    PRI_jet_subleading_pt_biased = np.linalg.norm(P_jet_subleading_biased[:2])
    PRI_met_biased = np.linalg.norm(P_MET_biased[:2])
    PRI_met_phi_biased = np.arctan2(P_MET_biased[1], P_MET_biased[0])

    return {
        "PRI_had_pt_biased": PRI_had_pt_biased,
        "PRI_jet_leading_pt_biased": PRI_jet_leading_pt_biased,
        "PRI_jet_subleading_pt_biased": PRI_jet_subleading_pt_biased,
        "PRI_met_biased": PRI_met_biased,
        "PRI_met_phi_biased": PRI_met_phi_biased,
    }


# Exemple d’application sur un DataFrame entier
def apply_biases_to_df(df, alpha_tes=1.0, alpha_jes=1.0, et_soft_sigma=0.0):
    biased_features = []
    for _, event in df.iterrows():
        biased_vals = apply_systematic_bias(event, alpha_tes, alpha_jes, et_soft_sigma)
        biased_features.append(biased_vals)
    biased_df = pd.DataFrame(biased_features, index=df.index)
    return pd.concat([df, biased_df], axis=1)


# Appliquer les biais sur le DataFrame original
df_biased = apply_biases_to_df(df, alpha_tes=1.05, alpha_jes=0.95, et_soft_sigma=5.0)

import seaborn as sns
import matplotlib.pyplot as plt

# Appliquer les biais sur le DataFrame original
df_biased = apply_biases_to_df(df, alpha_tes=1.05, alpha_jes=0.95, et_soft_sigma=5.0)

# Choix de la feature à comparer

feature_orig = "PRI_had_pt"
feature_biased = "PRI_had_pt_biased"


plt.figure(figsize=(8, 5))

plt.hist(
    df[feature_orig],
    bins=15,
    alpha=0.5,
    label="Original",
    density=True,
    color="blue",
    range=(0, 200),
)
plt.hist(
    df_biased[feature_biased],
    bins=15,
    alpha=0.5,
    label="Biaisé",
    density=True,
    color="red",
    range=(0, 200),
)

plt.title(f"Histogramme superposé : {feature_orig} vs biaisé")
plt.xlabel(feature_orig)
plt.ylabel("Densité")
plt.grid()
plt.legend()
plt.show()
