"""Feature selection and genre-stratified train/test split.

Encoding and scaling (one-hot for `key`, standardization for the linear
models) intentionally happen *inside* each model's training pipeline
(`modeling/train_regression.py`, `modeling/train_classification.py`) rather
than here, so those transforms are always fit on the training fold only --
fitting a scaler on the full dataset before splitting would leak test-set
statistics into training.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

import config

ID_COLUMNS = ["track_id", "track_name", "artists", "album_name"]


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Slim the cleaned dataset down to id columns + model inputs + target."""
    out = df[ID_COLUMNS + config.AUDIO_FEATURES + [config.GENRE_COLUMN, config.TARGET_COLUMN]].copy()
    out["explicit"] = out["explicit"].astype(int)
    return out


def genre_stratified_split(
    df: pd.DataFrame, test_size: float = config.TEST_SIZE, random_state: int = config.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[config.GENRE_COLUMN],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[config.AUDIO_FEATURES]
    y = df[config.TARGET_COLUMN]
    return X, y


def main() -> None:
    df = pd.read_parquet(config.TRACKS_CLEAN_PARQUET)
    model_df = select_model_columns(df)
    train_df, test_df = genre_stratified_split(model_df)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(config.TRAIN_PARQUET, index=False)
    test_df.to_parquet(config.TEST_PARQUET, index=False)

    train_genre_share = train_df[config.GENRE_COLUMN].value_counts(normalize=True).sort_index()
    test_genre_share = test_df[config.GENRE_COLUMN].value_counts(normalize=True).sort_index()
    max_genre_share_gap = (train_genre_share - test_genre_share).abs().max()

    print(f"Train rows: {len(train_df):,}  Test rows: {len(test_df):,}")
    print(f"Max |train genre share - test genre share| across {train_genre_share.shape[0]} genres: {max_genre_share_gap:.4f}")


if __name__ == "__main__":
    main()
