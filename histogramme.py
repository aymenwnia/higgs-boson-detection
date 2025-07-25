import pandas as pd
import matplotlib.pyplot as plt

# Chemin vers ton fichier parquet (à adapter)
file_path = "/Users/lucasdesgranges/Documents/EI - Higgs/Higgs_collaboration_B/blackSwan_data/blackSwan_data.parquet"

# Chargement du dataset
df = pd.read_parquet(file_path)

# Liste des features à tracer
features = [
    "PRI_lep_pt",
    "PRI_lep_eta",
    "PRI_lep_phi",
    "PRI_had_pt",
    "PRI_had_eta",
    "PRI_had_phi",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
    "PRI_n_jets",
    "PRI_jet_all_pt",
    "PRI_met",
    "PRI_met_phi",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_deltar_had_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_had",
    "DER_met_phi_centrality",
    "DER_lep_eta_centrality",
]

# Séparer signal et background
df_signal = df[df["labels"] == 1]
df_background = df[df["labels"] == 0]

for feature in features:
    plt.figure(figsize=(8, 4))

    # Données pour signal et background
    data_to_plot = [df_signal[feature], df_background[feature]]

    plt.hist(
        df_signal[feature],
        bins=50,
        alpha=0.5,
        label="Signal",
        density=True,
        color="blue",
        edgecolor="black",
        linewidth=1,
    )
    plt.hist(
        df_background[feature],
        bins=50,
        alpha=0.5,
        label="Background",
        density=True,
        color="pink",
        edgecolor="black",
        linewidth=1,
    )

    plt.title(f"Histogramme empilé de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Densité")
    plt.legend()
    plt.grid()
    plt.show()
