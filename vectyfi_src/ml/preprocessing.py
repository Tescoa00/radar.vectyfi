import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
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


def format_features(df: pd.DataFrame, split_ratio: float = 0.2):
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

    X_train, X_test, y_train, y_test = train_test_split_impute_missing(df)

    return X_train, X_test, y_train, y_test

def train_test_split_impute_missing(df: pd.DataFrame, split_ratio: float = 0.2):
    """train_test_split and impute missing values"""
    X = df.drop(columns=['TARGET_NOT_AWARDED'])
    y = df['TARGET_NOT_AWARDED']

    # train_test_split required for imputing missing values, otherwise data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=RANDOM_STATE, stratify=y,
    )

    preproc_num = make_pipeline(SimpleImputer(strategy="mean"))
    preproc_cat = make_pipeline(SimpleImputer(strategy="most_frequent"))

    preproc_transformer = make_column_transformer(
        (preproc_num, make_column_selector(dtype_include=["number"])),
        (preproc_cat, make_column_selector(dtype_include=["object"])),
        remainder="passthrough"
    ).set_output(transform="pandas")
    X_train_preproc = preproc_transformer.fit_transform(X_train)
    X_test_preproc = preproc_transformer.transform(X_test)
    # get original column names back
    X_train_preproc.columns = X_train.columns
    X_test_preproc.columns = X_test.columns

    return X_train_preproc, X_test_preproc, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer adapted to the columns actually present in X."""
    pt_feat = [c for c in PASS_THROUGH_FEATURES if c in X.columns]
    ohe_feat = [c for c in OHE_FEATURES if c in X.columns]
    tgt_feat = [c for c in TARGET_ENC_FEATURES if c in X.columns]

    # The ColumnTransformer applies the correct encoding rules to the correct columns simultaneously
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_bin', 'passthrough', pt_feat),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'), ohe_feat),
            ('target', TargetEncoder(), tgt_feat),
        ],
        remainder='drop',
    )
    return preprocessor


def build_pipeline(X: pd.DataFrame, xgb_params: dict | None = None) -> Pipeline:
    """Return a full sklearn Pipeline (preprocessor + XGBClassifier)."""
    params = xgb_params or DEFAULT_XGB_PARAMS
    preprocessor = build_preprocessor(X)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**params)),
    ])
    return pipeline


# ── Standalone runner (kept_feat for backward compatibility) ─────────────────────
def run_ml_pipeline(input_filepath: str) -> None:
    """Load cleaned data, build pipeline, run 5-fold CV and print results."""
    print("1. Loading the balanced TED dataset...")
    df = pd.read_csv(input_filepath, low_memory=False, sep='\t')

    print("2. Converting binary TED variables, train test split, and imputing missing values...")
    X_train, X_test, y_train, y_test = format_features(df) # X_test and y_test not used here

    print("3. Building leak-free pipeline with default hyperparameters...")
    pipeline = build_pipeline(X_train)

    print("4. Running 5-Fold Stratified Cross-Validation...")
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

    mean_auc = np.mean(scores['test_score'])

    print("\n" + "=" * 50)
    print("CROSS-VALIDATION FINISHED")
    print(f"TRAIN mean ROC-AUC Score: {mean_auc:.4f}")
    print(f"Hyperparameters used: {DEFAULT_XGB_PARAMS}")
    print("=" * 50)


# Execute the script
if __name__ == "__main__":
    run_ml_pipeline(RAW_DATA_PATH + 'balanced_cleaned_378k.tsv')