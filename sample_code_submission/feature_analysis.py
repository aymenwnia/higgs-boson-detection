import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du fichier parquet (chemin à adapter)
file_path = (
    "/Users/raph/Desktop/Higgs_collaboration_B/blackSwan_data/blackSwan_data.parquet"
)
df = pd.read_parquet(file_path)


df_jet0 = df[df["PRI_n_jets"] == 0].copy()
df_jet1 = df[df["PRI_n_jets"] == 1].copy()
df_jet2plus = df[df["PRI_n_jets"] >= 4].copy()


# Liste des features PRI et DER (selon metadata)
features_PRI = [col for col in df_jet0.columns if col.startswith("PRI_")]
features_DER = [col for col in df_jet0.columns if col.startswith("DER_")]


# Toutes les features à analyser
features = features_PRI + features_DER

# Calcul matrice de corrélation sur ces features
corr_matrix = df_jet2plus[features].corr()

# Focus : corrélations PRI vs DER
corr_PRI_DER = corr_matrix.loc[features_PRI, features_PRI]

plt.figure(figsize=(14, 8))
sns.heatmap(corr_PRI_DER, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Correlation PRI (rows) vs PRI (columns)")
plt.show()

# Optionnel : liste des corrélations fortes PRI-DER > seuil
seuil = 0.7
strong_corr = (
    corr_PRI_DER[(corr_PRI_DER.abs() > seuil)].stack().sort_values(ascending=False)
)
print("Strong correlations PRI vs DER (>|0.7|):")
print(strong_corr)


def feature_corrilations(data):
    pass


def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns
