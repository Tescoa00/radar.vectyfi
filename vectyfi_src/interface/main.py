import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse


def preprocess() -> None:
    """
    - Download raw data from csv
    - Preprocess data (cleaning, feature engineering, etc.)
    """

    print("✅ preprocess() done \n")


def train(
        split_ratio: float = 0.02,
        learning_rate: float = 0.0005,
        batch_size: int = 256,
        patience: int = 2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)


    model = load_model()

    if model is None:
        model = initialize_model(input_shape=X_train_processed.shape[1:])

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(
        model, X_train_processed, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val)
    )

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)


    print("✅ train() done \n")

    return val_mae



def evaluate() -> float:
    """
    Evaluate the performance of the latest production model on processed data
    """

    print("✅ evaluate() done \n")

    return score


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    pred()
