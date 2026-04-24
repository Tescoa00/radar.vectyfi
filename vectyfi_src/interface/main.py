"""
Vectyfi Radar — ML Pipeline Entry Point
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Import our own modules ─────────────────────────────────────────────────
from vectyfi_src.ml.data_cleaning import clean_ted_data
from vectyfi_src.ml.preprocessing import format_features, build_pipeline

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path("model_test.pkl")
RANDOM_STATE = 42

# Raw input → cleaned output
RAW_DATA_PATH = "raw_data/export_CAN_2023_2018.csv"
CLEAN_DATA_PATH = "raw_data/balanced_cleaned_378k.tsv"


# ── Pipeline Steps ──────────────────────────────────────────────────────────

def clean(raw_path: str = RAW_DATA_PATH, clean_path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Run the full data-cleaning pipeline and return the clean dataframe."""
    print("\n⭐️ Use case: clean")

    if Path(clean_path).exists():
        df = pd.read_csv(clean_path, low_memory=False, sep="\t")
        print(f"✅ clean data set '{clean_path}' loaded\n")
    else:
        df = clean_ted_data(raw_path, clean_path)
        print("✅ clean() done\n")

    return df


def preprocess(df: pd.DataFrame):
    """Format features and split into X, y."""
    df = format_features(df)
    # with new data, there's no target 'TARGET_NOT_AWARDED' column
    X = df.drop(columns=['TARGET_NOT_AWARDED'], errors='ignore')
    y = df['TARGET_NOT_AWARDED'] if 'TARGET_NOT_AWARDED' in df.columns else None
    return X, y


def train(df: pd.DataFrame, split_ratio: float = 0.2, save: bool = True):
    """
    Preprocess, train an XGBoost pipeline, evaluate, and optionally save.

    Args:
        df:          The cleaned, balanced dataframe.
        split_ratio: Fraction of data reserved for the test set (default 20%).
        save:        If True, export the trained pipeline to MODEL_PATH as .pkl.
                     Set to False (--no-save) to skip persistence.
    """
    print("\n⭐️ Use case: train")

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=RANDOM_STATE, stratify=y
        )

    # Build the full sklearn Pipeline (ColumnTransformer + XGBClassifier)
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    # Evaluation
    model_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    model_acc = accuracy_score(y_test, pipeline.predict(X_test))

    print(f"Model AUC:      {model_auc:.4f}")
    print(f"Model Accuracy: {model_acc*100:.1f}%")

    # Conditionally save the pipeline pickle
    if save:
        save_model(pipeline)
    else:
        print("⏭️  --no-save flag: skipping .pkl export")

    print("✅ train() done\n")
    return model_auc, X_test, y_test


def evaluate(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Load the saved model and score it on a test set."""
    print("\n⭐️ Use case: evaluate")
    model = load_model()
    if model is None:
        print("⚠️  No saved model found — run without --no-save first.")
        return 0.0
    score = model.score(X_test, y_test)
    print(f"Score: {score:.2f}")
    print("✅ evaluate() done\n")
    return score


def pred(X_pred: pd.DataFrame = None, raw_df: pd.DataFrame = None) -> np.ndarray:
    """Make a prediction using the saved model."""
    print("\n⭐️ Use case: pred")
    model = load_model()
    if model is None:
        print("⚠️  No saved model found — run without --no-save first.")
        return np.array([])
    if X_pred is None:
        X_pred = raw_df.sample(1, random_state=None)
    X_pred, _ = preprocess(X_pred)
    y_pred = model.predict(X_pred)
    print("\n✅ prediction done: ", y_pred, "\n")
    return y_pred


# ── Model I/O ───────────────────────────────────────────────────────────────

def save_model(model) -> None:
    """Serialize the full sklearn Pipeline (preprocessor + XGBClassifier) to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved at {str(MODEL_PATH)}")


def load_model():
    """Load and return the Pipeline from MODEL_PATH, or None if missing."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded {str(MODEL_PATH)}")
    return model


# ── CLI Entry Point ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vectyfi Radar — ML training pipeline",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip data cleaning; load existing balanced CSV instead.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not export the trained pipeline to a .pkl file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Step 1: Data Cleaning ───────────────────────────────────────────
    # --skip-clean → load the existing balanced CSV directly
    # (default)   → re-run the full cleaning pipeline from raw data
    if args.skip_clean:
        print("\n⏭️  --skip-clean: loading existing balanced CSV...")
        df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False, sep="\t")
    else:
        df = clean()

    # ── Step 2: Train ───────────────────────────────────────────────────
    # --no-save → train & evaluate but don't write model_test.pkl
    model_auc, X_test, y_test = train(df, save=not args.no_save)

    # ── Step 3: Evaluate ────────────────────────────────────────────────
    # Only meaningful if a .pkl was saved (or already existed)
    if not args.no_save:
        evaluate(X_test, y_test)

    # ── Step 4: Predict on a random row ─────────────────────────────────
    if not args.no_save:
        pred(raw_df=df)
