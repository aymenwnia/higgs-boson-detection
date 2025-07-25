from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import joblib
import os
from pathlib import Path
import numpy as np


def amsasimov(s_in, b_in):
    """
    asimov significance arXiv:1007.1727 eq. 97 (reduces to s/sqrt(b) if s<<b)
    """
    # if b==0 ams is undefined, but return 0 without warning for convenience (hack)
    s=np.copy(s_in)
    b=np.copy(b_in)
    s=np.where( (b_in == 0) , 0., s_in)
    b=np.where( (b_in == 0) , 1., b)

    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    ams = np.where((s < 0) | (b < 0), np.nan, ams)  # nan if unphysical values.
    if np.isscalar(s_in):
        return float(ams)
    else:
        return  ams

def significance_vscore(y_true, y_score, sample_weight=None):
    """
    Calculate the significance using the Asimov method.
    """
    if sample_weight is None:
        # Provide a default value of 1.
        sample_weight = np.full(len(y_true), 1.0)

    # Define bins for y_score, adapt the number as needed for your data
    bins = np.linspace(0, 1.0, 101)

    # Fills s and b weighted binned distributions
    s_hist, bin_edges = np.histogram(
        y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
    )
    b_hist, bin_edges = np.histogram(
        y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
    )

    # Compute cumulative sums (from the right!)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    # Compute significance
    significance = amsasimov(s_cumul, b_cumul)

    # Find the bin with the maximum significance
    max_value = np.max(significance)

    return significance

def significance_vscore_scorer(y_true, y_score):
    """
    Returns the maximum significance score for the given true labels and predicted scores.
    This function is designed to be used as a scorer in model evaluation.
    """
    return np.mean(significance_vscore(y_true, y_score))

class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    
    def __init__(self, train_data=None, model_type="sklearn"):
        if model_type == "xgb":
            self.model = XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss',
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                verbosity=2
                )
        elif model_type == "sklearn":
            self.model = GradientBoostingClassifier()
        elif model_type == "lgbm":
            self.model = LGBMClassifier(device="gpu")
        else:
            raise ValueError(f"Modèle non supporté : {model_type}")
        #self.model = XGBClassifier(learning_rate=0.36954584046859273,max_depth=6,n_estimators=194,use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()

    def save(self):
        """
        Save the model and scaler with unique filenames to avoid overwriting.
        """
        base_dir = Path().resolve()

        models_dir = base_dir / "models"
        scalers_dir = base_dir / "scalers"
        models_dir.mkdir(exist_ok=True)
        scalers_dir.mkdir(exist_ok=True)

        # File name generation
        i = 0
        while True:
            model_filename = models_dir / f"model_{i}.pkl"
            scaler_filename = scalers_dir / f"scaler_{i}.pkl"
            if not model_filename.exists() and not scaler_filename.exists():
                break
            i += 1

        joblib.dump(self.model, model_filename)
        joblib.dump(self.scaler, scaler_filename)

        print(f" Model saved : {model_filename.name}")
        print(f" Scaler saved : {scaler_filename.name}")

    def fit(self, train_data, labels, weights=None):
        """
        Fit the model to the training data.
        """
        self.scaler.fit_transform(train_data)
        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, weights)
        self.save()

    def fit_HPO(self, train_data, labels, weights=None):
        """
        Fit the model to the training data.
        """
    
        gsearch = self.model
        self.scaler.fit_transform(train_data)
        X_train_data = self.scaler.transform(train_data)

        # Recherche aléatoire (HPO)
        param_dist = {
            "max_depth": stats.randint(3, 10),
            "n_estimators": stats.randint(100, 300),
            "learning_rate": stats.uniform(0.05, 0.5),
        }

        gsearch = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            scoring=make_scorer(significance_vscore_scorer, needs_proba=True),
            n_iter=20,
            cv=2,
            random_state=42,
            verbose=1,
            n_jobs=-1
        )

        print("Starting model training with hyperparameter optimization...")
        gsearch.fit(X_train_data, labels, sample_weight=weights)
        self.model = gsearch.best_estimator_
        print("Best hyperparameters found: ", gsearch.best_params_)
        
        # Courbe AUC
        auc_scores = gsearch.cv_results_["mean_test_score"]
        param_strings = [str(p) for p in gsearch.cv_results_["params"]]
        iterations = range(1, len(auc_scores) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(iterations, auc_scores, marker='o', linestyle='-', color='tab:blue')
        plt.xlabel("Trial #")
        plt.ylabel("AUC (mean CV)")
        plt.title("AUC over HPO trials")
        plt.grid(True)
        plt.xticks(iterations)
        plt.tight_layout()
        plt.show()

        self.save()

    def predict(self, test_data):
        """ 
            Predict the labels for the test data.
            This method applies the same scaling to the test data as was applied to the training data.
        """
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]

    def load(self, models_dir, scalers_dir):
        """
        Load the most recently saved model and scaler from the specified directories.
        """
        # Trouver tous les fichiers modèle et scaler existants
        model_files = list(Path(models_dir).glob("model_*.pkl"))
        scaler_files = list(Path(scalers_dir).glob("scaler_*.pkl"))

        if not model_files or not scaler_files:
            raise FileNotFoundError(
                "No model or scaler found in models and scalers folders."
            )

        # Trier par date de modification (du plus récent au plus ancien)
        model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        scaler_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Charger les fichiers les plus récents
        latest_model = model_files[0]
        latest_scaler = scaler_files[0]

        print(f"Loading the model : {latest_model.name}")
        print(f"Loading the scaler : {latest_scaler.name}")

        self.model = joblib.load(latest_model)
        self.scaler = joblib.load(latest_scaler)

    def evaluate_AUC(self, test_data, labels):
        """
        Evaluate the model using AUC.
        """
        predictions = self.predict(test_data)
        return roc_auc_score(labels, predictions)

    def evaluate_significance(self, test_data, labels):
        """
        Evaluate the model using significance.
        """
        predictions = self.predict(test_data)
        return significance_vscore(labels, predictions)