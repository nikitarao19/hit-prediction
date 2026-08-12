"""Sanity tests for feature selection and the genre-stratified split."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from what_makes_a_hit.data.build_features import genre_stratified_split, get_xy, select_model_columns

RNG = np.random.default_rng(43)


def _synthetic_clean_df(n_per_genre: int = 50) -> pd.DataFrame:
    genres = ["pop", "hip-hop", "classical"]
    rows = []
    for genre in genres:
        for i in range(n_per_genre):
            rows.append(
                {
                    "track_id": f"{genre}_{i}",
                    "track_name": f"Track {i}",
                    "artists": "Artist",
                    "album_name": "Album",
                    "track_genre": genre,
                    "popularity": RNG.integers(0, 101),
                    "duration_ms": 200_000,
                    "explicit": bool(i % 2),
                    "danceability": RNG.uniform(0, 1),
                    "energy": RNG.uniform(0, 1),
                    "key": int(RNG.integers(0, 12)),
                    "loudness": RNG.uniform(-20, 0),
                    "mode": int(RNG.integers(0, 2)),
                    "speechiness": RNG.uniform(0, 1),
                    "acousticness": RNG.uniform(0, 1),
                    "instrumentalness": RNG.uniform(0, 1),
                    "liveness": RNG.uniform(0, 1),
                    "valence": RNG.uniform(0, 1),
                    "tempo": RNG.uniform(60, 180),
                    "time_signature": 4,
                }
            )
    return pd.DataFrame(rows)


def test_select_model_columns_casts_explicit_to_int_and_keeps_expected_columns():
    df = _synthetic_clean_df(n_per_genre=5)

    out = select_model_columns(df)

    expected_columns = set(["track_id", "track_name", "artists", "album_name"] + config.AUDIO_FEATURES + [config.GENRE_COLUMN, config.TARGET_COLUMN])
    assert set(out.columns) == expected_columns
    assert out["explicit"].dtype.kind in "iu"
    assert set(out["explicit"].unique()) <= {0, 1}


def test_genre_stratified_split_preserves_genre_proportions():
    df = select_model_columns(_synthetic_clean_df(n_per_genre=100))

    train_df, test_df = genre_stratified_split(df, test_size=0.2, random_state=43)

    assert len(train_df) + len(test_df) == len(df)

    train_share = train_df["track_genre"].value_counts(normalize=True)
    test_share = test_df["track_genre"].value_counts(normalize=True)
    for genre in df["track_genre"].unique():
        assert abs(train_share[genre] - test_share[genre]) < 0.02


def test_genre_stratified_split_has_no_overlapping_tracks():
    df = select_model_columns(_synthetic_clean_df(n_per_genre=30))

    train_df, test_df = genre_stratified_split(df, test_size=0.2, random_state=43)

    assert set(train_df["track_id"]) & set(test_df["track_id"]) == set()


def test_get_xy_returns_feature_matrix_and_target_series():
    df = select_model_columns(_synthetic_clean_df(n_per_genre=5))

    X, y = get_xy(df)

    assert list(X.columns) == config.AUDIO_FEATURES
    assert y.name == config.TARGET_COLUMN
    assert len(X) == len(y) == len(df)
