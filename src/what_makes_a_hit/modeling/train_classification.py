"""Classification framing: is this track a "hit" (top 20% popularity)?

The hit threshold is the 80th percentile of `popularity` computed on the
*training* fold only, then applied as a fixed cutoff to both train and test
-- computing it from the full dataset (train+test) would leak test-set label
information into how the label itself is defined.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import config

CATEGORICAL_FEATURES = ["key"]
NUMERIC_FEATURES = [f for f in config.AUDIO_FEATURES if f not in CATEGORICAL_FEATURES]


def build_logistic_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000))])


def build_gbm_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
    )


def evaluate(y_true, y_proba) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "positive_rate": float(np.mean(y_true)),
    }


def main() -> None:
    train_df = pd.read_parquet(config.TRAIN_PARQUET)
    test_df = pd.read_parquet(config.TEST_PARQUET)

    hit_threshold = float(train_df[config.TARGET_COLUMN].quantile(config.HIT_PERCENTILE))

    y_train = (train_df[config.TARGET_COLUMN] >= hit_threshold).astype(int)
    y_test = (test_df[config.TARGET_COLUMN] >= hit_threshold).astype(int)
    X_train, X_test = train_df[config.AUDIO_FEATURES], test_df[config.AUDIO_FEATURES]

    logistic_model = build_logistic_pipeline()
    logistic_model.fit(X_train, y_train)
    logistic_proba = logistic_model.predict_proba(X_test)[:, 1]
    logistic_metrics = evaluate(y_test, logistic_proba)

    gbm_model = build_gbm_classifier()
    gbm_model.fit(X_train, y_train)
    gbm_proba = gbm_model.predict_proba(X_test)[:, 1]
    gbm_metrics = evaluate(y_test, gbm_proba)

    gbm_precision, gbm_recall, _ = precision_recall_curve(y_test, gbm_proba)
    log_precision, log_recall, _ = precision_recall_curve(y_test, logistic_proba)
    pr_curve_df = pd.concat(
        [
            pd.DataFrame({"model": "xgboost_classifier", "precision": gbm_precision, "recall": gbm_recall}),
            pd.DataFrame({"model": "logistic_regression", "precision": log_precision, "recall": log_recall}),
        ],
        ignore_index=True,
    )

    results = {
        "hit_threshold_popularity": hit_threshold,
        "hit_percentile": config.HIT_PERCENTILE,
        "logistic_regression": logistic_metrics,
        "xgboost_classifier": gbm_metrics,
        "gbm_beats_logistic": gbm_metrics["auc"] > logistic_metrics["auc"],
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_model, config.LOGISTIC_MODEL_PATH)
    joblib.dump(gbm_model, config.GBM_CLASSIFIER_PATH)
    config.CLASSIFICATION_RESULTS_JSON.write_text(json.dumps(results, indent=2))
    pr_curve_df.to_parquet(config.PR_CURVE_PARQUET, index=False)

    print(f"Hit threshold (top {(1 - config.HIT_PERCENTILE) * 100:.0f}% by train popularity): >= {hit_threshold:.1f}")
    print("Classification results (held-out test set):")
    for name, metrics in [("Logistic regression", logistic_metrics), ("XGBoost classifier", gbm_metrics)]:
        print(f"  {name:<24} AUC={metrics['auc']:.3f}  AvgPrecision={metrics['average_precision']:.3f}")


if __name__ == "__main__":
    main()
