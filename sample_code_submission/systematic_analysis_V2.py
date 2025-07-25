import numpy as np
from HiggsML.systematics import systematics
from model import Model
from HiggsML.datasets import download_dataset
from utils import histogram_dataset

data = download_dataset("blackSwan_data")

data.load_train_set()
data_set = data.get_train_set()


def tes_fitter(model, train_set, nbin=25):
    """
    Task 1 : Analysis TES Uncertainty
    1. Loop over different values of tes and make store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of tes and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given TES

    """
    signal_field = train_set[target == 1]
    background_field = train_set[target == 0]

    syst_set_signal = systematics(signal_field, tes=1)
    score_signal = model.predict(syst_set_signal["data"])
    histogram_nominal_signal, _ = np.histogram(
        score_signal, bins=nbin, range=(0, 1), weights=train_set["weights"]
    )

    syst_set_background = systematics(background_field, tes=1)
    score_background = model.predict(syst_set_background["data"])
    histogram_nominal_background, _ = np.histogram(
        score_background, bins=nbin, range=(0, 1), weights=train_set["weights"]
    )

    total_delta_S_signal = []
    total_delta_S_background = []
    for i in range(len(histogram_nominal_signal)):
        bin_nominal_signal = histogram_nominal_signal[i]
        bin_nominal_background = histogram_nominal_background[i]
        tes_range = np.linspace(0.9, 1.1, 10)
        delta_S_signal = []
        delta_S_background = []
        for tes in tes_range:

            syst_set_signal = systematics(signal_field, tes)
            score_signal = model.predict(syst_set_signal["data"])
            histogram_signal, _ = np.histogram(
                score_signal, bins=nbin, range=(0, 1), weights=train_set["weights"]
            )

            syst_set_signal = systematics(background_field, tes)
            score_signal = model.predict(syst_set_background["data"])
            histogram_background, _ = np.histogram(
                score_signal, bins=nbin, range=(0, 1), weights=train_set["weights"]
            )

            bin_signal = histogram_signal[i]
            bin_background = histogram_background[i]

            delta_signal = bin_signal - bin_nominal_signal
            delta_background = bin_background - bin_nominal_background

            delta_S_signal.append(delta_signal)
            delta_S_background.append(delta_background)
        total_delta_S_signal.append(delta_S_signal)
        total_delta_S_background.append(delta_S_background)

    # Write a function to loop over different values of tes and histogram and make fit function which transforms the histogram for any given TES

    def fit_function(array, tes):
        # Dummy fit function, replace with actual fitting procedure
        return array * tes

    return fit_function


def jes_fitter(
    model,
    train_set,
):
    """
    Task 1 : Analysis JES Uncertainty
    1. Loop over different values of jes and store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of JES and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given jes

    """
    syst_set = systematics(train_set, jes=1)
    score = model.predict(syst_set["data"])

    histogram = np.histogram(score, bins=25, range=(0, 1))

    # Write a function to loop over different values of jes and histogram and make fit function which transforms the histogram for any given JES

    def fit_function(array, jes):
        # Dummy fit function, replace with actual fitting procedure
        return array * jes

    return fit_function
