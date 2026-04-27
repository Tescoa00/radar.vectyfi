import re
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def build_explainer(pipeline: Pipeline) -> shap.Explainer:
    """Build a SHAP TreeExplainer from the XGBClassifier inside the pipeline.
    Call once at startup — not per request.
    """
    xgb_model = pipeline[-1]
    return shap.Explainer(xgb_model)


def explain_instance(
    explainer: shap.Explainer,
    preprocessor,
    X_raw: pd.DataFrame,
) -> dict:
    """Preprocess X_raw and return SHAP base value + per-feature contributions.

    preprocessor must be model[0] with .set_output(transform="pandas") (included in preprocessing.py function 'build_preprocessor'),
    so that .transform() returns a named DataFrame.
    """
    X_transformed = preprocessor.transform(X_raw)

    shap_values = explainer(X_transformed)
    sv = shap_values[0]

    clean = re.compile(r"^[^_]+__")
    features = [clean.sub("", name) for name in X_transformed.columns.tolist()]
    values = X_transformed.iloc[0].tolist()
    contributions = sv.values.tolist()
    base_value = float(sv.base_values)

    return {
        "base_value": base_value,
        "shap_values": [
            {"feature": f, "value": round(v, 4), "shap": round(s, 4)}
            for f, v, s in zip(features, values, contributions)
        ],
    }
