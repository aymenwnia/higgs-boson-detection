# ------------------------------
# Dummy Sample Submission
# ------------------------------

BDT = True
NN = False

from statistical_analysis import calculate_saved_info, compute_mu
from boosted_decision_tree import significance_vscore
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def amsasimov(s_in, b_in):
    """
    asimov significance arXiv:1007.1727 eq. 97 (reduces to s/sqrt(b) if s<<b)
    """
    # if b==0 ams is undefined, but return 0 without warning for convenience (hack)
    s = np.copy(s_in)
    b = np.copy(b_in)
    s = np.where((b_in == 0), 0.0, s_in)
    b = np.where((b_in == 0), 1.0, b)

    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    ams = np.where((s < 0) | (b < 0), np.nan, ams)  # nan if unphysical values.
    if np.isscalar(s_in):
        return float(ams)
    else:
        return ams


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


class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init :
        takes 3 arguments: train_set systematics and model_type.
        can be used for initializing variables, classifier etc.
    2) fit :
        takes no arguments
        can be used to train a classifier
    3) predict:
        takes 1 argument: test sets
        can be used to get predictions of the test set.
        returns a dictionary

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods

            When you add another file with the submission model e.g. a trained model to be loaded and used,
            load it in the following way:

            # get to the model directory (your submission directory)
            model_dir = os.path.dirname(os.path.abspath(__file__))

            your trained model file is now in model_dir, you can load it from here
    """

    def __init__(
        self,
        get_train_set=None,
        systematics=None,
        model_type="sample_model",
        force_retrain=True,
    ):
        """
        Model class constructor

        Params:
            train_set:
                a dictionary with data, labels, weights and settings

            systematics:
                a class which you can use to get a dataset with systematics added
                See sample submission for usage of systematics


        Returns:
            None
        """

        indices = np.arange(1400000)

        np.random.shuffle(indices)

        train_indices = indices[:500000]
        holdout_indices = indices[500000:550000]
        valid_indices = indices[550000:]

        training_df = get_train_set(selected_indices=train_indices)

        self.training_set = {
            "labels": training_df.pop("labels"),
            "weights": training_df.pop("weights"),
            "detailed_labels": training_df.pop("detailed_labels"),
            "data": training_df,
        }

        del training_df

        self.systematics = systematics

        print("Training Data: ", self.training_set["data"].shape)
        print("Training Labels: ", self.training_set["labels"].shape)
        print("Training Weights: ", self.training_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 0].sum(),
        )

        valid_df = get_train_set(selected_indices=valid_indices)

        self.valid_set = {
            "labels": valid_df.pop("labels"),
            "weights": valid_df.pop("weights"),
            "detailed_labels": valid_df.pop("detailed_labels"),
            "data": valid_df,
        }
        del valid_df

        print()
        print("Valid Data: ", self.valid_set["data"].shape)
        print("Valid Labels: ", self.valid_set["labels"].shape)
        print("Valid Weights: ", self.valid_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.valid_set["weights"][self.valid_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.valid_set["weights"][self.valid_set["labels"] == 0].sum(),
        )

        holdout_df = get_train_set(selected_indices=holdout_indices)

        self.holdout_set = {
            "labels": holdout_df.pop("labels"),
            "weights": holdout_df.pop("weights"),
            "detailed_labels": holdout_df.pop("detailed_labels"),
            "data": holdout_df,
        }

        del holdout_df

        print()
        print("Holdout Data: ", self.holdout_set["data"].shape)
        print("Holdout Labels: ", self.holdout_set["labels"].shape)
        print("Holdout Weights: ", self.holdout_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.holdout_set["weights"][self.holdout_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.holdout_set["weights"][self.holdout_set["labels"] == 0].sum(),
        )
        print(" \n ")

        print("Training Data: ", self.training_set["data"].shape)
        # print(f"DEBUG: model_type = {repr(model_type)}")

        if model_type == "BDT":
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree(train_data=self.training_set["data"])

            if not force_retrain:
                try:
                    base_dir = Path(__file__).resolve().parent.parent
                    models_dir = base_dir / "models"
                    scalers_dir = base_dir / "scalers"
                    models_dir.mkdir(exist_ok=True)
                    scalers_dir.mkdir(exist_ok=True)
                    self.model.load(models_dir=models_dir, scalers_dir=scalers_dir)
                    print("Pretrained model and scaler loaded successfully.")
                    self.model_loaded = True
                except Exception as e:
                    print(f"Error loading pretrained model: {e}")
                    self.model_loaded = False
            else:
                print("Force retraining the model, loading pretrained model skipped.")
                self.model_loaded = False
                
       

         
            
        elif model_type == "NN":
            from neural_network import NeuralNetwork

            self.model = NeuralNetwork(train_data=self.training_set["data"])
            
        #ajout
        elif model_type == "SKLEARN_BDT":
            from boosted_decision_tree import BoostedDecisionTree
            self.model = BoostedDecisionTree(train_data=self.training_set["data"], model_type="sklearn")
        elif model_type == "sample_model":
            pass
        
        else:
            print(f"model_type {model_type} not found")
            raise ValueError(f"model_type {model_type} not found")
        self.name = model_type

        print(f" Model is { self.name}")

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model

        Returns:
            None
        """
        if getattr(self, "model_loaded", False):
            print("Model already loaded. We skip the training step.")
            self.holdout_set = self.systematics(self.holdout_set)

            self.saved_info = calculate_saved_info(self.model, self.holdout_set)

            self.training_set = self.systematics(self.training_set)

            # Compute  Results
            train_score = self.model.predict(self.training_set["data"])
            train_results = compute_mu(
                train_score, self.training_set["weights"], self.saved_info
            )

            holdout_score = self.model.predict(self.holdout_set["data"])
            holdout_results = compute_mu(
                holdout_score, self.holdout_set["weights"], self.saved_info
            )

            self.valid_set = self.systematics(self.valid_set)

            valid_score = self.model.predict(self.valid_set["data"])

            valid_results = compute_mu(
                valid_score, self.valid_set["weights"], self.saved_info
            )

            print("Train Results: ")
            for key in train_results.keys():
                print("\t", key, " : ", train_results[key])

            print("Holdout Results: ")
            for key in holdout_results.keys():
                print("\t", key, " : ", holdout_results[key])

            print("Valid Results: ")
            for key in valid_results.keys():
                print("\t", key, " : ", valid_results[key])

            print("Significance (Asimov):")
            significance = significance_vscore(
                y_true=self.holdout_set["labels"],
                y_score=holdout_score,
                sample_weight=self.holdout_set["weights"],
            )
            max_significance = max(significance)
            print(f"\tMaximum Asimov significance: {max_significance:.4f}")


            x = np.linspace(0, 1, num=len(significance))


            plt.plot(x, significance)


            plt.title("BDT Significance")
            plt.xlabel("Threshold")
            plt.ylabel("Significance")
            plt.legend()
            plt.show()



            # self.model.significancecurve(
            #     X_test=self.valid_set["data"],
            #     y_test=self.valid_set["labels"],
            #     weights_test=self.valid_set["weights"]
            # )

            self.valid_set["data"]["score"] = valid_score
            from utils import roc_curve_wrapper, histogram_dataset

            histogram_dataset(
                self.valid_set["data"],
                self.valid_set["labels"],
                self.valid_set["weights"],
                columns=["score"],
            )

            from HiggsML.visualization import stacked_histogram

            stacked_histogram(
                self.valid_set["data"],
                self.valid_set["labels"],
                self.valid_set["weights"],
                self.valid_set["detailed_labels"],
                "score",
            )

            roc_curve_wrapper(
                score=valid_score,
                labels=self.valid_set["labels"],
                weights=self.valid_set["weights"],
                plot_label="valid_set" + self.name,
            )
            
                        
            return
        balanced_set = self.training_set.copy()

        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
        class_weights_train = (
            weights_train[train_labels == 0].sum(),
            weights_train[train_labels == 1].sum(),
        )

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= (
                max(class_weights_train) / class_weights_train[i]
            )
            # test dataset : increase test weight to compensate for sampling

        balanced_set["weights"] = weights_train

        self.model.fit_HPO(
            balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
        )

        self.holdout_set = self.systematics(self.holdout_set)

        self.saved_info = calculate_saved_info(self.model, self.holdout_set)

        self.training_set = self.systematics(self.training_set)

        # Compute  Results
        train_score = self.model.predict(self.training_set["data"])
        train_results = compute_mu(
            train_score, self.training_set["weights"], self.saved_info
        )

        holdout_score = self.model.predict(self.holdout_set["data"])

        holdout_results = compute_mu(
            holdout_score, self.holdout_set["weights"], self.saved_info
        )

        self.valid_set = self.systematics(self.valid_set)

        valid_score = self.model.predict(self.valid_set["data"])

        valid_results = compute_mu(
            valid_score, self.valid_set["weights"], self.saved_info
        )

        print("Train Results: ")
        for key in train_results.keys():
            print("\t", key, " : ", train_results[key])

        print("Holdout Results: ")
        for key in holdout_results.keys():
            print("\t", key, " : ", holdout_results[key])

        print("Valid Results: ")
        for key in valid_results.keys():
            print("\t", key, " : ", valid_results[key])

        print("Significance (Asimov):")
        significance = significance_vscore(
            y_true=self.holdout_set["labels"],
            y_score=holdout_score,
            sample_weight=self.holdout_set["weights"],
        )
        max_significance = max(significance)
        max_significance = max(significance)
        print(f"\tMaximum Asimov significance: {max_significance:.4f}")

        x = np.linspace(0, 1, num=len(significance))

        plt.plot(x, significance)
        plt.title("BDT Significance")
        plt.xlabel("Threshold")
        plt.ylabel("Significance")
        plt.legend()
        plt.show()



        x = np.linspace(0, 1, num=len(significance))


        plt.plot(x, significance)


        plt.title("BDT Significance")
        plt.xlabel("Threshold")
        plt.ylabel("Significance")
        plt.legend()
        plt.show()
            
        self.valid_set["data"]["score"] = valid_score
        
        from utils import roc_curve_wrapper, histogram_dataset

        histogram_dataset(
            self.valid_set["data"],
            self.valid_set["labels"],
            self.valid_set["weights"],
            columns=["score"],
        )

        from HiggsML.visualization import stacked_histogram

        stacked_histogram(
            self.valid_set["data"],
            self.valid_set["labels"],
            self.valid_set["weights"],
            self.valid_set["detailed_labels"],
            "score",
        )

        roc_curve_wrapper(
            score=valid_score,
            labels=self.valid_set["labels"],
            weights=self.valid_set["weights"],
            plot_label="valid_set" + self.name,
        )

    def predict(self, test_set):
        """
        Params:
            test_set

        Functionality:
            this function can be used for predictions using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)
        self.saved_info = calculate_saved_info(self.model, self.holdout_set)
        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result
