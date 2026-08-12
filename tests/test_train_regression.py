"""Sanity tests for the regression models: finite metrics, and the gradient
boosting model actually beating a "predict the mean" baseline on a dataset
with a real (if noisy) signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from what_makes_a_hit.modeling.train_regression import build_gbm_regressor, build_linear_pipeline, evaluate

RNG = np.random.default_rng(43)


def _synthetic_xy(n: int = 800):
    """popularity is a noisy function of danceability + energy, so a model
    that actually learns from the features should beat the mean baseline."""
    danceability = RNG.uniform(0, 1, n)
    energy = RNG.uniform(0, 1, n)
    key = RNG.integers(0, 12, n)
    noise = RNG.normal(0, 5, n)
    popularity = np.clip(20 + 50 * danceability + 20 * energy + noise, 0, 100)

    X = pd.DataFrame(
        {
            "danceability": danceability,
            "energy": energy,
            "key": key,
            "loudness": RNG.uniform(-20, 0, n),
            "mode": RNG.integers(0, 2, n),
            "speechiness": RNG.uniform(0, 1, n),
            "acousticness": RNG.uniform(0, 1, n),
            "instrumentalness": RNG.uniform(0, 1, n),
            "liveness": RNG.uniform(0, 1, n),
            "valence": RNG.uniform(0, 1, n),
            "tempo": RNG.uniform(60, 180, n),
            "time_signature": np.full(n, 4),
            "duration_ms": RNG.uniform(120_000, 300_000, n),
            "explicit": RNG.integers(0, 2, n),
        }
    )
    y = pd.Series(popularity, name="popularity")
    return X, y


def test_evaluate_computes_correct_rmse_and_mae():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])

    metrics = evaluate(y_true, y_pred)

    assert np.isclose(metrics["mae"], 2.75)
    assert np.isclose(metrics["rmse"], np.sqrt((4 + 4 + 9 + 16) / 4))


def test_linear_and_gbm_models_produce_finite_predictions():
    X, y = _synthetic_xy()
    X_train, y_train = X.iloc[:600], y.iloc[:600]
    X_test = X.iloc[600:]

    linear_model = build_linear_pipeline()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)

    gbm_model = build_gbm_regressor()
    gbm_model.fit(X_train, y_train)
    gbm_pred = gbm_model.predict(X_test)

    assert np.all(np.isfinite(linear_pred))
    assert np.all(np.isfinite(gbm_pred))


def test_models_beat_mean_baseline_on_a_dataset_with_real_signal():
    X, y = _synthetic_xy()
    X_train, y_train = X.iloc[:600], y.iloc[:600]
    X_test, y_test = X.iloc[600:], y.iloc[600:]

    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_rmse = evaluate(y_test, baseline_pred)["rmse"]

    linear_model = build_linear_pipeline()
    linear_model.fit(X_train, y_train)
    linear_rmse = evaluate(y_test, linear_model.predict(X_test))["rmse"]

    gbm_model = build_gbm_regressor()
    gbm_model.fit(X_train, y_train)
    gbm_rmse = evaluate(y_test, gbm_model.predict(X_test))["rmse"]

    assert linear_rmse < baseline_rmse
    assert gbm_rmse < baseline_rmse
