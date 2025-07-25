from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import joblib
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    # delete the model saved in models folder before changing parameters to train and save a new model
    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]

        self.model.add(Dense(1024, input_dim=n_dim, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(8, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )

        self.model_path = os.path.join(
            os.path.dirname(__file__), "models/modelNN.keras"
        )

        self.scaler = StandardScaler()

    def save_model(self):
        self.model.save(self.model_path)
        joblib.dump(
            self.scaler, os.path.join(os.path.dirname(__file__), "models/scaler.pkl")
        )
        print("Model saved to models/scaler.pkl")

    def load_model(self):
        self.model = load_model(self.model_path)
        print("Model loaded from models/scaler.pkl")
        self.scaler = joblib.load(
            os.path.join(os.path.dirname(__file__), "models/scaler.pkl")
        )

    def fit(self, train_data, y_train, weights_train=None):
        if os.path.isfile(
            os.path.join(os.path.dirname(__file__), "models/modelNN.keras")
        ):
            self.load_model()
        else:
            self.scaler.fit_transform(train_data)
            X_train = self.scaler.transform(train_data)
            early_stop = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
            history = self.model.fit(
                X_train,
                y_train,
                sample_weight=weights_train,
                validation_split=0.2,
                epochs=80,
                batch_size=256,
                callbacks=[early_stop],
                verbose=1,
            )
            self._plot_loss(history)
            self.save_model()

    def predict(self, test_data):
        if "score" in test_data.columns:
            test_data = test_data.drop(columns=["score"])
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()

    def _plot_loss(self, history):
        plt.figure()
        plt.plot(history.history["loss"], label="Training loss")
        plt.plot(history.history["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()
