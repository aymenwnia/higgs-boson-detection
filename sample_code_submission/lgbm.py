import numpy as np
import lightgbm as lgb


class LGBM:
    def __init__(self, train_data, model_type="classifier", lgbm_params=None):
        self.model_type = model_type
        self.lgbm_params = lgbm_params if lgbm_params is not None else {}
        self.model = None
        if "random_state" not in self.lgbm_params:  # For reproducibility
            self.lgbm_params["random_state"] = 42
        if "n_estimators" not in self.lgbm_params:  # Default number of trees
            self.lgbm_params["n_estimators"] = 100
        if (
            "verbose" not in self.lgbm_params
        ):  # Suppress LightGBM's own verbosity by default
            self.lgbm_params["verbose"] = -1

        print(
            f"LightGBM Model wrapper initialized for '{model_type}' with params: {self.lgbm_params}"
        )

    def fit(self, train_data, labels, weights):
        train_np = np.array(train_data)
        labels_np = np.array(labels)
        weights_np = np.array(weights) if weights is not None else None
        if weights_np is not None:
            if len(weights_np) != len(train_np):
                raise ValueError("Length of weights must match length of X_train.")
            print("  Using sample weights during training.")

        if self.model_type == "regressor":
            self.model = lgb.LGBMRegressor(**self.lgbm_params)
        elif self.model_type == "classifier":
            self.model = lgb.LGBMClassifier(**self.lgbm_params)
        else:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. Choose 'regressor' or 'classifier'."
            )

        self.model.fit(train_np, labels_np, sample_weight=weights_np)
        print("Model training complete.")
        print("LGBM DEBUG:")
        print("\n  X train shape:", train_np.shape)
        print(" \n y shape:", labels_np.shape)
        print(
            " \n weights shape:", weights_np.shape if weights_np is not None else "None"
        )
        print(f"\nStarting LightGBM model training ({self.model_type})...")

    def predict(self, test_data):
        test_np = np.array(test_data)
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        if self.model_type != "classifier":
            raise AttributeError(
                "predict_proba is only available for classifier models."
            )
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                "The underlying LightGBM model does not have a predict_proba method."
            )
        print("  Predict input shape:", test_np.shape)
        print("\nMaking probability predictions with LightGBM classifier...")
        probabilities = self.model.predict_proba(test_np)
        print(f"  Probabilities (first 5 if available): {probabilities[:5]}")
        print("Probability prediction complete.")
        return probabilities
