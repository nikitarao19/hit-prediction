"""Central configuration for what-makes-a-hit."""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_CSV = DATA_RAW_DIR / "dataset.csv"

TRACKS_CLEAN_PARQUET = DATA_PROCESSED_DIR / "tracks_clean.parquet"
TRAIN_PARQUET = DATA_PROCESSED_DIR / "train.parquet"
TEST_PARQUET = DATA_PROCESSED_DIR / "test.parquet"

REGRESSION_RESULTS_JSON = DATA_PROCESSED_DIR / "regression_results.json"
CLASSIFICATION_RESULTS_JSON = DATA_PROCESSED_DIR / "classification_results.json"
PR_CURVE_PARQUET = DATA_PROCESSED_DIR / "pr_curve.parquet"

SHAP_VALUES_PARQUET = DATA_PROCESSED_DIR / "shap_values_regression.parquet"
SHAP_SAMPLE_PARQUET = DATA_PROCESSED_DIR / "shap_sample_features.parquet"

GENRE_BREAKDOWN_CSV = DATA_PROCESSED_DIR / "genre_breakdown.csv"
GENRE_SHAP_JSON = DATA_PROCESSED_DIR / "genre_top_features.json"

MODEL_DIR = DATA_PROCESSED_DIR / "models"
LINEAR_MODEL_PATH = MODEL_DIR / "linear_regression.joblib"
GBM_REGRESSOR_PATH = MODEL_DIR / "xgb_regressor.joblib"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
GBM_CLASSIFIER_PATH = MODEL_DIR / "xgb_classifier.joblib"

# --- Target / features ---------------------------------------------------
TARGET_COLUMN = "popularity"

AUDIO_FEATURES = [
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
    "duration_ms",
    "explicit",
]

GENRE_COLUMN = "track_genre"

# Sliders in the dashboard feature explorer only expose the features an
# audio-engineer / A&R lens actually cares about; key/mode/time_signature
# are categorical-ish and less interesting to twiddle live.
DASHBOARD_SLIDER_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Genres used for the genre-vs-genre breakdown analysis: a deliberately
# varied mix (mainstream pop/hip-hop, a lyrics-driven genre, an
# instrumental-heavy genre, and dance/electronic) so "what makes a hit"
# has a real chance of looking different across rows.
FOCUS_GENRES = ["pop", "hip-hop", "classical", "edm", "acoustic", "metal"]

# --- Split / modeling params ----------------------------------------------
TEST_SIZE = 0.2
RANDOM_SEED = 43

# "Hit" = top 20% most popular tracks, threshold computed from the training
# set only so the test set can't leak into the label definition.
HIT_PERCENTILE = 0.80

# SHAP summary computed on a random subsample of the test set for speed;
# large enough to be a stable read on global feature importance.
SHAP_SAMPLE_SIZE = 2000
