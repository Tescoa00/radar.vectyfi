import pandas as pd
import xgboost as xgb
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
import warnings

# Suppress minor warnings for cleaner terminal output
warnings.filterwarnings('ignore')

def run_ml_pipeline(input_filepath):
    print("1. Loading the balanced TED dataset...")
    df = pd.read_csv(input_filepath, low_memory=False)

    print("2. Formatting data types & converting binary TED variables...")
    # Force CRIT_PRICE_WEIGHT to be a pure number so it is not accidentally treated as text [7]
    if 'CRIT_PRICE_WEIGHT' in df.columns:
        df['CRIT_PRICE_WEIGHT'] = pd.to_numeric(df['CRIT_PRICE_WEIGHT'], errors='coerce')

    # Map the procedural "B_" flags ("Y" = 1, everything else = 0) based on TED codebook [5, 6, 8]
    binary_flags = ['B_MULTIPLE_CAE', 'B_EU_FUNDS', 'B_FRA_AGREEMENT', 'B_GPA', 'B_ACCELERATED']
    for col in binary_flags:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

    # Map CRIT_CODE based on TED rules: L (Lowest Price) = 1, M (Most economically advantageous) = 0 [7]
    if 'CRIT_CODE' in df.columns:
        df['CRIT_CODE'] = df['CRIT_CODE'].apply(lambda x: 1 if str(x).strip().upper() == 'L' else 0)

    # Separate features (X) from the target (y)
    X = df.drop(columns=['TARGET_NOT_AWARDED'])
    y = df['TARGET_NOT_AWARDED']

    print("3. Building the ColumnTransformer for Leak-Free Preprocessing...")
    # Group features strictly based on their mathematical nature and TED cardinality
    pass_through_features = ['CRIT_PRICE_WEIGHT', 'LOTS_NUMBER', 'YEAR', 'CRIT_CODE'] + binary_flags
    ohe_features = ['TYPE_OF_CONTRACT', 'TOP_TYPE', 'CAE_TYPE']
    target_features = ['ISO_COUNTRY_CODE', 'MAIN_ACTIVITY']

    # Ensure we only pass columns that exist in the dataframe to prevent crashes
    pass_through_features = [c for c in pass_through_features if c in X.columns]
    ohe_features = [c for c in ohe_features if c in X.columns]
    target_features = [c for c in target_features if c in X.columns]

    # The ColumnTransformer applies the correct encoding rules to the correct columns simultaneously
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_bin', 'passthrough', pass_through_features),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_features),
            ('target', TargetEncoder(), target_features) 
        ],
        remainder='drop'
    )

    print("4. Defining Optuna Objective Function for XGBoost...")
    def objective(trial):
        # Allow Optuna to test different parameter combinations
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'eval_metric': 'aucpr',
            'tree_method': 'hist',
            'random_state': 42
        }
        
        xgb_model = xgb.XGBClassifier(**param)
        
        # CRITICAL: Wrapping preprocessing and modeling in a Pipeline ensures Target Encoding 
        # is strictly calculated on training folds and applied to test folds, preventing data leaks.
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb_model)
        ])
        
        # Execute 5-Fold Stratified Cross Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Optuna will attempt to maximize this mean ROC-AUC score
        return np.mean(scores['test_score'])

    print("5. Initializing Bayesian Hyperparameter Optimization...")
    study = optuna.create_study(direction='maximize')
    # n_trials can be adjusted based on available computing power (e.g., 50 for a thorough search)
    study.optimize(objective, n_trials=50) 

    print("\n" + "="*50)
    print("OPTUNA OPTIMIZATION FINISHED")
    print(f"Best ROC-AUC Score: {study.best_value:.4f}")
    print("Best Hyperparameters Discovered:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)

# Execute the script
if __name__ == "__main__":
    run_ml_pipeline('raw_data/f_balanced_500k.csv')