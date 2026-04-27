import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import make_pipeline, Pipeline
import warnings

# Suppress minor warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# ── Feature Groups ──────────────────────────────────────────────────────────
BINARY_FLAGS = ['B_MULTIPLE_CAE', 'B_EU_FUNDS', 'B_FRA_AGREEMENT', 'B_GPA', 'B_ACCELERATED']
PASS_THROUGH_FEATURES = ['CRIT_PRICE_WEIGHT', 'LOTS_NUMBER', 'YEAR', 'CRIT_CODE'] + BINARY_FLAGS
OHE_FEATURES = ['TYPE_OF_CONTRACT', 'TOP_TYPE', 'CAE_TYPE']
TARGET_ENC_FEATURES = ['ISO_COUNTRY_CODE', 'MAIN_ACTIVITY']

RANDOM_STATE = 42
RAW_DATA_PATH='./raw_data/'

# ── Default XGBoost hyperparameters (no Optuna) ────────────────────────────
DEFAULT_XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'aucpr',
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
}


def format_features(df: pd.DataFrame) -> pd.DataFrame:
    #TODO binary features can actually all be encoded with OHE in conjunction with the categorical ones
    """Convert raw TED columns into ML-ready types (in-place safe)."""
    df = df.copy()

    # Map the procedural "B_" flags ("Y" = 1, everything else = 0) based on TED codebook [5, 6, 8]
    for col in BINARY_FLAGS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

    # Map CRIT_CODE based on TED rules: L (Lowest Price) = 1, M (Most economically advantageous) = 0 [7]
    if 'CRIT_CODE' in df.columns:
        df['CRIT_CODE'] = df['CRIT_CODE'].apply(lambda x: 1 if str(x).strip().upper() == 'L' else 0)

    return df

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer adapted to the columns actually present in X."""
    pt_cols = [c for c in PASS_THROUGH_FEATURES if c in X.columns]
    ohe_cols = [c for c in OHE_FEATURES if c in X.columns]
    tgt_cols = [c for c in TARGET_ENC_FEATURES if c in X.columns]

    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='mean')), pt_cols),
        (make_pipeline(SimpleImputer(strategy='most_frequent'),
                       OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary')),
         ohe_cols),
        (make_pipeline(SimpleImputer(strategy='most_frequent'),
                       TargetEncoder()),
         tgt_cols),
        remainder='drop'
        ).set_output(transform="pandas") # added newly for the SHAP analysis

    return preprocessor


def build_pipeline(X: pd.DataFrame, xgb_params: dict | None = None) -> Pipeline:
    """Return a full sklearn Pipeline (preprocessor + XGBClassifier)."""
    params = xgb_params or DEFAULT_XGB_PARAMS
    preprocessor = build_preprocessor(X)
    pipeline = make_pipeline(
        preprocessor,
        xgb.XGBClassifier(**params)
    )
    return pipeline


# ── Standalone runner (kept for backward compatibility) ─────────────────────
def run_ml_pipeline(input_filepath: str) -> None:
    """Load cleaned data, build pipeline, run 5-fold CV and print results."""
    print("1. Loading the balanced TED dataset...")
    df = pd.read_csv(input_filepath, low_memory=False, sep='\t')

    print("2. Converting binary TED variables...")
    df = format_features(df)

    # Separate features (X) from the target (y)
    X = df.drop(columns=['TARGET_NOT_AWARDED'])
    y = df['TARGET_NOT_AWARDED']

    print("3. Building leak-free pipeline with default hyperparameters...")
    pipeline = build_pipeline(X)

    print("4. Running 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    mean_auc = np.mean(scores['test_score'])

    print("\n" + "=" * 50)
    print("CROSS-VALIDATION FINISHED")
    print(f"Mean ROC-AUC Score: {mean_auc:.4f}")
    print(f"Hyperparameters used: {DEFAULT_XGB_PARAMS}")
    print("=" * 50)


# Execute the script
if __name__ == "__main__":
    run_ml_pipeline(RAW_DATA_PATH + 'balanced_cleaned_378k.tsv')