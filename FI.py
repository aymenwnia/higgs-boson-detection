import pickle
import matplotlib.pyplot as plt
import pandas as pd


file_path = "blackSwan_data/blackSwan_data.parquet"
data = pd.read_parquet(file_path)


# Charger le modèle
import joblib

model = joblib.load("models/model.pkl")

import xgboost as xgb

# Si model est un Booster XGBoost ou un wrapper sklearn
model.get_booster().get_score(importance_type="weight")

n_cols = data.shape[1]
n_feat = len(model.feature_importances_)

print(data.columns[:n_feat])
print(model.feature_importances_)

# 4. Prendre les bonnes colonnes si nécessaire
columns_to_plot = data.columns[:n_feat]

plt.figure(figsize=(14, 8))
plt.bar(columns_to_plot, model.feature_importances_)
plt.xticks(rotation=90)
plt.title("Feature importances XGBoost Hist")
# plt.savefig(new_dir + "/VarImp_BDT_XGBoost_Hist.pdf",bbox_inches='tight')
plt.show()

# plt.savefig(new_dir + "/VarImp_BDT_LightGBM.pdf",bbox_inches='tigh
