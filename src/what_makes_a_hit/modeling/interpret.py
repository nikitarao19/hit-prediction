"""SHAP interpretability for the XGBoost popularity regressor.

Computed once here (not live in the dashboard) so the deployed Streamlit app
doesn't need the `shap` package or its C-extension build at all -- it just
reads the two parquet files this script writes and plots them directly.
"""

from __future__ import annotations

import joblib
import pandas as pd
import shap

import config


def main() -> None:
    test_df = pd.read_parquet(config.TEST_PARQUET)
    model = joblib.load(config.GBM_REGRESSOR_PATH)

    sample_n = min(config.SHAP_SAMPLE_SIZE, len(test_df))
    sample_df = test_df.sample(n=sample_n, random_state=config.RANDOM_SEED).reset_index(drop=True)
    X_sample = sample_df[config.AUDIO_FEATURES]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_df = pd.DataFrame(shap_values, columns=config.AUDIO_FEATURES)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shap_df.to_parquet(config.SHAP_VALUES_PARQUET, index=False)
    X_sample.assign(track_genre=sample_df[config.GENRE_COLUMN]).to_parquet(config.SHAP_SAMPLE_PARQUET, index=False)

    top_features = shap_df.abs().mean().sort_values(ascending=False)
    print(f"SHAP values computed on {sample_n:,} held-out tracks.")
    print("Top features by mean |SHAP value|:")
    for feature, value in top_features.items():
        print(f"  {feature:<20} {value:.3f}")


if __name__ == "__main__":
    main()
