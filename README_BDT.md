# Higgs_collaboration_B
Collaboration repository for Higgs EL 1 B

# How to switch between standard training and automated hyperparameter optimization?

To force the retraining of the model, instead of loading an existing one, edit line 101 :

force_retrain = True

To use hyperparameter optimization (HPO) instead of standard training, you need to change "fit" to "fit_HPO" on line 375 in the model.py file:

self.model.fit_HPO(
    balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
)

# How to switch between model types
to change model type , go to boosteddecisiontree.py, line 81, in the __init__ function , 
change model_type to "xgb", "sklearn", or "lgbm"


