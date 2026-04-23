import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from vectyfi_src.interface.main import preprocess, FEATURES

DATA_PATH  = Path("raw_data/balanced_cleaned_378k.csv")
MODEL_PATH = Path("ml/model.pkl")

BEST_PARAMS = {
    "n_estimators"      : 424,
    "max_depth"         : 9,
    "learning_rate"     : 0.1358,
    "colsample_bytree"  : 0.7367,
    "subsample"         : 0.6061,
    "min_child_weight"  : 6,
    "gamma"             : 0.0866,
    "reg_alpha"         : 0.8867,
    "reg_lambda"        : 1.0218,
}

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [col.split('__')[-1] for col in df.columns]
    df['INFO_ON_NON_AWARD'] = df['TARGET_NOT_AWARDED'].apply(lambda x: 'awarded' if x == 0 else 'not_awarded')
    print(f"Loaded: {len(df):,} rows")

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = XGBClassifier(
        **BEST_PARAMS,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Test AUC: {auc:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
