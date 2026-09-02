"""
Vendor Reliability Scoring — Modeling
--------------------------------------
Loads the synthetic vendor dataset, trains models to PREDICT the
reliability_score from raw operational/financial/behavioral features
(as if the composite score were expensive/slow to compute by hand,
or as if we want to score NEW vendors before enough history exists
to calculate it directly), and inspects which features matter most.

Models compared:
  1. Linear Regression   (interpretable baseline)
  2. Random Forest       (captures non-linearity/interactions)
  3. Gradient Boosting   (usually strongest performer)

Outputs:
  - Model performance comparison (R^2, MAE)
  - Feature importance plot (which signals drive reliability)
  - Saved trained model + feature list for the Streamlit app
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = "vendor_data.csv"

NUMERIC_FEATURES = [
    "years_as_vendor",
    "num_orders_completed",
    "avg_order_value_usd",
    "on_time_delivery_rate",
    "defect_rejection_rate",
    "lead_time_variance_days",
    "price_competitiveness",
    "payment_terms_compliance",
    "rfq_response_time_hours",
    "communication_responsiveness_score",
    "price_volatility_across_quotes",
    "dispute_frequency_per_year",
    "has_iso_certification",
]
CATEGORICAL_FEATURES = ["category", "country"]
TARGET = "reliability_score"

BEHAVIORAL_FEATURES = {
    "rfq_response_time_hours",
    "communication_responsiveness_score",
    "price_volatility_across_quotes",
    "dispute_frequency_per_year",
}


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(name, pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:20s}  R^2 = {r2:.3f}   MAE = {mae:.2f} pts")
    return r2, mae


def get_feature_names(pipe):
    preprocessor = pipe.named_steps["preprocess"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


def plot_feature_importance(pipe, feature_names, top_n=15, out_path="feature_importance.png"):
    importances = pipe.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    names = [feature_names[i] for i in order]
    vals = importances[order]
    colors = ["#d97706" if n in BEHAVIORAL_FEATURES else "#2563eb" for n in names]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(names))[::-1], vals, color=colors)
    plt.yticks(range(len(names))[::-1], names)
    plt.xlabel("Feature importance")
    plt.title("What drives the predicted reliability score?\n(orange = behavioral feature)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved feature importance plot -> {out_path}")


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # 1. Linear baseline
    lin_pipe = build_pipeline(LinearRegression())
    lin_pipe.fit(X_train, y_train)
    results["Linear Regression"] = evaluate("Linear Regression", lin_pipe, X_test, y_test)

    # 2. Random Forest
    rf_pipe = build_pipeline(RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42))
    rf_pipe.fit(X_train, y_train)
    results["Random Forest"] = evaluate("Random Forest", rf_pipe, X_test, y_test)

    # 3. Gradient Boosting
    gb_pipe = build_pipeline(GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42))
    gb_pipe.fit(X_train, y_train)
    results["Gradient Boosting"] = evaluate("Gradient Boosting", gb_pipe, X_test, y_test)

    # Pick best model by R^2 for the demo app
    best_name = max(results, key=lambda k: results[k][0])
    best_pipe = {"Linear Regression": lin_pipe, "Random Forest": rf_pipe, "Gradient Boosting": gb_pipe}[best_name]
    print(f"\nBest model: {best_name}")

    # Feature importance (tree-based models only)
    if hasattr(best_pipe.named_steps["model"], "feature_importances_"):
        feature_names = get_feature_names(best_pipe)
        plot_feature_importance(best_pipe, feature_names)

    joblib.dump(best_pipe, "reliability_model.pkl")
    print("Saved trained model -> reliability_model.pkl")


if __name__ == "__main__":
    main()
