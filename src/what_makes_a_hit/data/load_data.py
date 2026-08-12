"""Load and clean the raw Kaggle Spotify tracks dataset.

Known quirks of this dataset, handled here:
  - A stray `Unnamed: 0` column (a leftover pandas index from however the
    dataset author exported it) -- dropped.
  - The same `track_id` can appear multiple times with an otherwise
    *identical* feature row, only the `track_genre` label differs. This
    happens because the dataset was assembled by querying Spotify's API
    per-genre, so a track that appears on more than one genre's charts gets
    one row per genre. Left as-is, a duplicated track could land in both the
    train and test split with an (almost) exactly memorized feature vector,
    which would quietly inflate test-set performance. We deduplicate on
    `track_id` (keep first) to remove that leakage risk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

import config


@dataclass
class CleaningReport:
    raw_rows: int
    raw_unique_track_ids: int
    dropped_invalid_duration_or_missing_meta: int
    dropped_invalid_popularity: int
    dropped_duplicate_track_id: int
    final_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


NUMERIC_COLUMNS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]


def load_raw(csv_path=None) -> pd.DataFrame:
    csv_path = csv_path or config.RAW_CSV
    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def clean_tracks(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    raw_rows = len(df)
    raw_unique_track_ids = df["track_id"].nunique()

    df = df.copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["explicit"] = df["explicit"].astype(bool)

    valid_meta = df["duration_ms"].notna() & (df["duration_ms"] > 0) & df["track_name"].notna() & df["artists"].notna()
    dropped_invalid_duration_or_missing_meta = int((~valid_meta).sum())
    df = df[valid_meta]

    valid_popularity = df["popularity"].between(0, 100)
    dropped_invalid_popularity = int((~valid_popularity).sum())
    df = df[valid_popularity]

    before_dedupe = len(df)
    df = df.drop_duplicates(subset=["track_id"], keep="first")
    dropped_duplicate_track_id = before_dedupe - len(df)

    df = df.reset_index(drop=True)

    report = CleaningReport(
        raw_rows=raw_rows,
        raw_unique_track_ids=raw_unique_track_ids,
        dropped_invalid_duration_or_missing_meta=dropped_invalid_duration_or_missing_meta,
        dropped_invalid_popularity=dropped_invalid_popularity,
        dropped_duplicate_track_id=dropped_duplicate_track_id,
        final_rows=len(df),
    )
    return df, report


def load_and_clean(csv_path=None) -> tuple[pd.DataFrame, CleaningReport]:
    return clean_tracks(load_raw(csv_path))


def main() -> None:
    df, report = load_and_clean()
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.TRACKS_CLEAN_PARQUET, index=False)

    report_path = config.DATA_PROCESSED_DIR / "cleaning_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))

    print("Cleaning report:")
    for key, value in report.to_dict().items():
        print(f"  {key}: {value:,}")
    print(f"Wrote {len(df):,} clean rows to {config.TRACKS_CLEAN_PARQUET}")


if __name__ == "__main__":
    main()
