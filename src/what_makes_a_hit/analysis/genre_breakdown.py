"""Does 'what makes a hit' differ by genre?

For a handful of deliberately varied focus genres (mainstream pop/hip-hop,
lyrics-driven, instrumental-heavy classical, dance-oriented edm, ...),
compares (a) how well the single global XGBoost regressor performs within
just that genre's tracks, and (b) which audio features drive predicted
popularity within that genre, via genre-scoped SHAP values from the same
fitted model (no per-genre refit -- the question is whether one global
model's *reasoning* shifts by genre, not whether a bespoke model would fit
each genre better).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config


def main() -> None:
    test_df = pd.read_parquet(config.TEST_PARQUET)
    model = joblib.load(config.GBM_REGRESSOR_PATH)
    explainer = shap.TreeExplainer(model)

    rows = []
    genre_top_features = {}

    for genre in config.FOCUS_GENRES:
        genre_df = test_df[test_df[config.GENRE_COLUMN] == genre]
        if genre_df.empty:
            continue

        X_genre = genre_df[config.AUDIO_FEATURES]
        y_genre = genre_df[config.TARGET_COLUMN]
        y_pred = model.predict(X_genre)

        rmse = float(np.sqrt(mean_squared_error(y_genre, y_pred)))
        mae = float(mean_absolute_error(y_genre, y_pred))

        shap_values = explainer.shap_values(X_genre)
        shap_df = pd.DataFrame(shap_values, columns=config.AUDIO_FEATURES)
        top_features = shap_df.abs().mean().sort_values(ascending=False)

        rows.append(
            {
                "genre": genre,
                "n_test": len(genre_df),
                "rmse": rmse,
                "mae": mae,
                "mean_popularity": float(y_genre.mean()),
                "top_feature": top_features.index[0],
            }
        )
        genre_top_features[genre] = {k: float(v) for k, v in top_features.items()}

    breakdown_df = pd.DataFrame(rows).sort_values("mean_popularity", ascending=False)
    breakdown_df.to_csv(config.GENRE_BREAKDOWN_CSV, index=False)
    config.GENRE_SHAP_JSON.write_text(json.dumps(genre_top_features, indent=2))

    print("Genre breakdown (global XGBoost model, evaluated within each genre's test rows):")
    print(breakdown_df.to_string(index=False))


if __name__ == "__main__":
    main()
