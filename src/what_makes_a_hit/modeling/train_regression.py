"""Regression framing: predict raw 0-100 popularity from audio features.

Compares a linear regression baseline against an XGBoost gradient boosting
regressor, plus a "predict the training mean" floor so the linear model's
own value-add is visible, not just the gradient boosting model's.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

import config

CATEGORICAL_FEATURES = ["key"]
NUMERIC_FEATURES = [f for f in config.AUDIO_FEATURES if f not in CATEGORICAL_FEATURES]


def build_linear_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])


def build_gbm_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    train_df = pd.read_parquet(config.TRAIN_PARQUET)
    test_df = pd.read_parquet(config.TEST_PARQUET)

    X_train, y_train = train_df[config.AUDIO_FEATURES], train_df[config.TARGET_COLUMN]
    X_test, y_test = test_df[config.AUDIO_FEATURES], test_df[config.TARGET_COLUMN]

    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_metrics = evaluate(y_test, baseline_pred)

    linear_model = build_linear_pipeline()
    linear_model.fit(X_train, y_train)
    linear_metrics = evaluate(y_test, linear_model.predict(X_test))

    gbm_model = build_gbm_regressor()
    gbm_model.fit(X_train, y_train)
    gbm_metrics = evaluate(y_test, gbm_model.predict(X_test))

    results = {
        "baseline_mean": baseline_metrics,
        "linear_regression": linear_metrics,
        "xgboost_regressor": gbm_metrics,
        "gbm_beats_linear": gbm_metrics["rmse"] < linear_metrics["rmse"],
        "linear_beats_baseline": linear_metrics["rmse"] < baseline_metrics["rmse"],
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(linear_model, config.LINEAR_MODEL_PATH)
    joblib.dump(gbm_model, config.GBM_REGRESSOR_PATH)
    config.REGRESSION_RESULTS_JSON.write_text(json.dumps(results, indent=2))

    print("Regression results (held-out test set):")
    for name, metrics in [
        ("Baseline (predict train mean)", baseline_metrics),
        ("Linear regression", linear_metrics),
        ("XGBoost regressor", gbm_metrics),
    ]:
        print(f"  {name:<32} RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}")


if __name__ == "__main__":
    main()
