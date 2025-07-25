import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du fichier parquet (chemin à adapter)
file_path = "/Users/lucasdesgranges/Documents/EI - Higgs/Higgs_collaboration_B/blackSwan_data/blackSwan_data.parquet"
df = pd.read_parquet(file_path)

# Liste des features PRI et DER (selon metadata)
features_PRI = [col for col in df.columns if col.startswith("PRI_")]
features_DER = [col for col in df.columns if col.startswith("DER_")]

# Toutes les features à analyser
features = features_PRI + features_DER

# Calcul matrice de corrélation sur ces features
corr_matrix = df[features].corr()

# Affichage heatmap complète (grande, donc attention à la taille)
plt.figure(figsize=(18, 16))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation matrix between all features (PRI and DER)")
plt.show()

# Focus : corrélations PRI vs DER
corr_PRI_DER = corr_matrix.loc[features_PRI, features_DER]
plt.figure(figsize=(14, 8))
sns.heatmap(corr_PRI_DER, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Correlation PRI (rows) vs DER (columns)")
plt.show()

# Optionnel : liste des corrélations fortes PRI-DER > seuil
seuil = 0.7
strong_corr = (
    corr_PRI_DER[(corr_PRI_DER.abs() > seuil)].stack().sort_values(ascending=False)
)
print("Strong correlations PRI vs DER (>|0.7|):")
print(strong_corr)
