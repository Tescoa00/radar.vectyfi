import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from colorama import Fore, Style
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("model_test.pkl")
RANDOM_STATE = 42
DATA_PATH = "/Users/basile/code/Tescoa00/raw_data/export_CAN_2023_2018_balanced_500k.tsv"

FEATURES = [
    'B_MULTIPLE_CAE', 'B_EU_FUNDS', 'LOTS_NUMBER', 'TOP_TYPE', 'YEAR',
    'ISO_COUNTRY_CODE', 'TYPE_OF_CONTRACT', 'B_GPA', 'B_FRA_AGREEMENT',
    'CRIT_PRICE_WEIGHT', 'CRIT_CODE', 'B_ACCELERATED', 'CAE_TYPE', 'MAIN_ACTIVITY'
]
CAT_FEATURES     = ['TOP_TYPE', 'ISO_COUNTRY_CODE', 'TYPE_OF_CONTRACT', 'CAE_TYPE', 'MAIN_ACTIVITY']
BINARY_FEATURES  = ['B_MULTIPLE_CAE', 'B_EU_FUNDS', 'B_GPA', 'B_FRA_AGREEMENT', 'B_ACCELERATED']
NUMERIC_FEATURES = ['CRIT_PRICE_WEIGHT', 'LOTS_NUMBER', 'YEAR']

def preprocess(df: pd.DataFrame):
    """Préprocesse le df et retourne X, y"""
    df = df.copy()
    df['target'] = (df['INFO_ON_NON_AWARD'] == 'awarded').astype(int)

    df_model = df[FEATURES + ['target']].copy()

    for col in CAT_FEATURES:
        df_model[col] = df_model[col].fillna('missing').astype('category')
    for col in BINARY_FEATURES:
        df_model[col] = df_model[col].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    df_model['CRIT_CODE'] = df_model['CRIT_CODE'].map({'L': 0, 'M': 1}).fillna(0).astype(int)
    for col in NUMERIC_FEATURES:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)

    X = df_model[FEATURES]
    y = df_model['target']
    return X, y

def train(df: pd.DataFrame, split_ratio: float = 0.2) -> float:
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=RANDOM_STATE, stratify=y
    )
    model = XGBClassifier(
        n_estimators=100, max_depth=4,
        enable_categorical=True, tree_method='hist',
        random_state=RANDOM_STATE, n_jobs=-1,
        )
    model.fit(X_train, y_train)

    model_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    model_acc = accuracy_score(y_test, model.predict(X_test))

    # Évaluation
    print(f'Model AUC:      {model_auc:.4f}')
    print(f'Model Accuracy: {model_acc*100:.1f}%')

    # Export du pickle
    save_model(model)

    print("✅ train() done \n")
    return model_auc, X_test, y_test

def evaluate(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    model = load_model()
    score = model.score(X_test, y_test)
    print(f"Score: {score:.2f}")
    print("✅ evaluate() done \n")
    return score

def pred(X_pred: pd.DataFrame = None, raw_df: pd.DataFrame = None) -> np.ndarray:
    model = load_model()
    if X_pred is None:
        X_pred, _ = preprocess(raw_df.sample(1, random_state=None))
    y_pred = model.predict(X_pred)
    print("\n✅ prediction done: ", y_pred, "\n")
    return y_pred

def save_model(model) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved at {MODEL_PATH}")

def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded")
    return model

if __name__ == '__main__':
    raw_df = pd.read_csv(DATA_PATH, sep='\t', low_memory=False)
    model_auc, X_test, y_test = train(raw_df)
    evaluate(X_test, y_test)
    pred(raw_df=raw_df)
