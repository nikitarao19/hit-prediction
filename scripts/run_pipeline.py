#!/usr/bin/env python3
"""Run the full what-makes-a-hit pipeline end-to-end and print a summary.

    python scripts/run_pipeline.py

Steps: clean raw data -> build features + split -> train regression models
-> train classification models -> compute SHAP -> genre breakdown.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src.what_makes_a_hit.analysis import genre_breakdown
from src.what_makes_a_hit.data import build_features, load_data
from src.what_makes_a_hit.modeling import interpret, train_classification, train_regression


def step(name: str, fn) -> float:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    start = time.time()
    fn()
    elapsed = time.time() - start
    print(f"[{elapsed:.1f}s]")
    return elapsed


def main() -> None:
    total_start = time.time()

    step("1/6  Loading + cleaning raw data", load_data.main)
    step("2/6  Building features + genre-stratified split", build_features.main)
    step("3/6  Training regression models (linear vs. XGBoost)", train_regression.main)
    step("4/6  Training classification models (logistic vs. XGBoost)", train_classification.main)
    step("5/6  Computing SHAP values", interpret.main)
    step("6/6  Genre breakdown", genre_breakdown.main)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")

    reg = json.loads(config.REGRESSION_RESULTS_JSON.read_text())
    clf = json.loads(config.CLASSIFICATION_RESULTS_JSON.read_text())

    print("\nRegression (RMSE / MAE, held-out test set):")
    print(f"  Baseline (mean)     RMSE={reg['baseline_mean']['rmse']:.3f}  MAE={reg['baseline_mean']['mae']:.3f}")
    print(f"  Linear regression   RMSE={reg['linear_regression']['rmse']:.3f}  MAE={reg['linear_regression']['mae']:.3f}")
    print(f"  XGBoost regressor   RMSE={reg['xgboost_regressor']['rmse']:.3f}  MAE={reg['xgboost_regressor']['mae']:.3f}")

    print(f"\nClassification (hit = top {(1 - clf['hit_percentile']) * 100:.0f}%, popularity >= {clf['hit_threshold_popularity']:.1f}):")
    print(f"  Logistic regression AUC={clf['logistic_regression']['auc']:.3f}  AP={clf['logistic_regression']['average_precision']:.3f}")
    print(f"  XGBoost classifier  AUC={clf['xgboost_classifier']['auc']:.3f}  AP={clf['xgboost_classifier']['average_precision']:.3f}")

    shap_df = pd.read_parquet(config.SHAP_VALUES_PARQUET)
    top_shap = shap_df.abs().mean().sort_values(ascending=False).head(5)
    print("\nTop 5 SHAP features (regression model):")
    for feature, value in top_shap.items():
        print(f"  {feature:<20} {value:.3f}")

    print(f"\nTotal pipeline time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
