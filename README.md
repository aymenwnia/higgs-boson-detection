# Higgs Boson Detection with Boosted Decision Trees

This project is part of a larger Higgs boson detection initiative using simulated data from the ATLAS experiment at CERN. The primary goal is to accurately distinguish between a "signal" process (the decay of a Higgs boson into two tau leptons, $H \rightarrow \tau\tau$) and various "background" processes that look similar.

This repository specifically focuses on the implementation, training, and optimization of **Boosted Decision Tree (BDT)** models to tackle this classification problem.

---

##  My Contribution: BDT Model Analysis

My core contribution was to develop and evaluate a machine learning pipeline centered around Boosted Decision Trees. This involved:

1.  **Comparing BDT Implementations**: I benchmarked the performance of three popular and powerful gradient boosting libraries:
    * **XGBoost**: A highly optimized and widely used library known for its performance and accuracy.
    * **LightGBM**: A fast, distributed, high-performance gradient boosting framework.
    * **Scikit-learn**: Using its `GradientBoostingClassifier` as a robust baseline.

2.  **Hyperparameter Optimization (HPO)**: To move beyond baseline performance, I implemented a systematic approach to tune the models.
    * I integrated **`RandomizedSearchCV`** into the training pipeline. This allows for efficient exploration of a wide range of hyperparameters to find the optimal combination.
    * Users can easily switch between a standard training run (using default parameters) and an HPO run to see the impact of tuning.

3.  **In-Depth Performance Evaluation**: I focused on metrics crucial for particle physics:
    * **Statistical Significance ($Z$)**: The primary metric used to quantify a discovery. A higher significance indicates greater confidence that the signal is real and not a random fluctuation.
    * **Area Under the ROC Curve (AUC)**: A standard measure of a classifier's ability to distinguish between classes.

---

##  Key Findings & Results

The analysis demonstrated the critical role of both model choice and hyperparameter optimization in maximizing discovery potential.

### Baseline vs. Optimized Model

The initial training with a default XGBoost model established a solid baseline:
* **AUC**: 0.881
* **Significance ($Z$)**: 6.29

After applying `RandomizedSearchCV`, the model's performance improved substantially:
* **AUC**: 0.882 (a negligible change)
* **Significance ($Z$)**: **7.3** (a significant improvement!)

This key result shows that while AUC is a useful metric, **optimizing directly for significance is crucial for the physics goal.**

### Model Comparison

After running HPO for all three implementations, the final comparison yielded a clear winner:

| Model | Optimized Significance ($Z$) |
| :--- | :--- |
| **XGBoost** | **6.61** |
| LightGBM | 6.30 |
| Scikit-learn | 6.10 |

**Conclusion**: The **XGBoost classifier, after hyperparameter optimization, provided the highest statistical significance**, making it the best choice for this detection task.

---

##  How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/higgs-boson-detection.git](https://github.com/your-username/higgs-boson-detection.git)
    cd higgs-boson-detection
    ```

2.  **Install dependencies:**

3.  **Run the analysis:**
    * The project uses a pre-compiled dataset specified within the main script.
    * To run a standard training pass with XGBoost:
        ```python
        python train.py --model xgboost --hpo False
        ```
    * To run a full Hyperparameter Optimization with XGBoost:
        ```python
        python train.py --model xgboost --hpo True
        ```

---

##  Technologies Used

* Python 3.x
* Pandas & NumPy for data manipulation
* Matplotlib & Seaborn for plotting
* Scikit-learn for preprocessing and HPO (`RandomizedSearchCV`)
* XGBoost
* LightGBM
