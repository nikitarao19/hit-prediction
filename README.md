# What Makes a Hit? — Predicting Spotify Track Popularity from Audio Features

Spotify assigns every track a popularity score (0-100, weighted toward recent
plays). This project asks how much of that score can be predicted from a
track's audio characteristics alone — danceability, energy, tempo, valence,
and similar features — using a real dataset of **89,740 unique tracks across
113 genres**. Two model families are compared honestly against a naive
baseline, with SHAP values used to show which audio features actually matter
most, and the analysis is upfront about what audio features *can't* explain.

**[Explore the interactive dashboard →](#dashboard)** (feature-slider hit
predictor, SHAP explorer, genre breakdown, model scoreboard)

## Results at a glance

| Framing | Model | Metric | Score |
|---|---|---|---|
| Regression (predict 0-100) | Baseline (predict the mean) | RMSE / MAE | 20.68 / 17.33 |
| Regression | Linear regression | RMSE / MAE | 20.28 / 16.74 |
| Regression | **XGBoost** | RMSE / MAE | **18.70 / 14.93** |
| Classification (top-20% "hit") | Logistic regression | AUC / Avg. Precision | 0.637 / 0.298 |
| Classification | **XGBoost** | AUC / Avg. Precision | **0.717 / 0.387** |

XGBoost beats the linear baseline in both framings, and the linear model
itself beats a "just predict the average" floor — so there's a real,
if modest, learnable signal in audio features alone. The top SHAP drivers of
predicted popularity are **instrumentalness, valence, acousticness,
danceability,** and **duration** — see [Interpretability](#interpretability)
below.

## Honest limitation

**Audio features alone cannot fully explain real-world popularity.** An RMSE
of ~18.7 points on a 0-100 scale, and a classification AUC of 0.72, both say
the same thing: audio characteristics carry real signal but leave most of the
variance unexplained. Artist fame, marketing spend, and playlist placement
aren't in this dataset, and they plausibly explain a meaningful share of what
makes a track popular in the real world. This project measures how much of
popularity *audio DNA alone* can predict — not a complete model of what makes
a song a hit.

Two more caveats worth stating plainly:
- **This is a static snapshot, not a live feed.** Popularity reflects
  whatever Spotify's algorithm computed when this dataset was collected, not
  real-time streams — a track's true current popularity may have moved on.
- **Genre labels are Spotify's own categorization**, assembled by querying
  their API per-genre, not an objective ground truth about a track's musical
  identity.

## Data

[Kaggle: "Spotify Tracks Dataset"](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
by maharshipandya — ~114,000 tracks, 125 genres, one static CSV. Pulled via
the Kaggle CLI (`kaggle datasets download -d maharshipandya/-spotify-tracks-dataset`),
no scraping or live polling required — the entire modeling pipeline could be
built and evaluated immediately, with no data-accumulation wait.

**Cleaning** (`src/what_makes_a_hit/data/load_data.py`), row counts before/after:

| Step | Rows |
|---|---|
| Raw file | 114,000 |
| Dropped: invalid duration / missing track name or artist | −1 |
| Dropped: duplicate `track_id` | −24,259 |
| **Final, clean, unique tracks** | **89,740** |

The duplicate-`track_id` drop is the interesting one and isn't mentioned in
Kaggle's dataset description: this dataset was assembled by querying
Spotify's API **per genre**, so a track that charts in more than one genre's
list gets one row per genre — same `track_id`, same audio features, only the
`track_genre` label differs (confirmed by checking: popularity has a
standard deviation of ~0 within duplicated-`track_id` groups). Left in, a
duplicated track could land in both the train and test split with an
almost-exactly-memorized feature row, quietly inflating test-set performance.
Deduplicating on `track_id` (keep first) removes that leakage risk before the
split ever happens.

## Methodology

- **Two framings on purpose**: a **regression** framing predicting the raw
  0-100 popularity score (RMSE/MAE), and a secondary **classification**
  framing — "is this track in the top 20% most popular?" (AUC / average
  precision) — to demonstrate both evaluation families cleanly rather than
  reporting a single accuracy number.
- **Split**: random 80/20, **stratified by `track_genre`** so both splits get
  proportional genre representation (max train/test genre-share gap after
  stratification: **0.0000**, i.e. essentially exact). No event-level
  grouping/leakage concern otherwise — each row is an independent track.
- **Hit threshold**: the 80th percentile of `popularity`, computed on the
  **training set only** (52.0 in this run) and then applied as a fixed cutoff
  to both splits — computing it from the full dataset would leak test-set
  label information into the label's own definition.
- **Models**: linear regression vs. XGBoost for the regression framing;
  logistic regression vs. XGBoost for the classification framing. `key` is
  one-hot encoded and numeric features are standardized for the two linear
  models (fit on train only); XGBoost trains on the raw feature values
  directly.
- **Interpretability**: SHAP `TreeExplainer` on the XGBoost regressor,
  computed on a 2,000-track held-out sample.
- **Genre breakdown**: the same global XGBoost model, evaluated and
  SHAP-explained separately within each of six deliberately varied focus
  genres (pop, hip-hop, classical, edm, acoustic, metal) to see whether "what
  makes a hit" shifts by genre.

## Interpretability

Top 5 features by mean |SHAP value| (XGBoost regressor, 2,000 held-out
tracks): **instrumentalness, valence, acousticness, danceability,
duration_ms**. Instrumental tracks tend to predict lower popularity (mainstream
listening skews toward vocal tracks); danceable, energetic, positive-valence
tracks predict higher — both directionally intuitive, a useful sanity check
that the model learned something real rather than noise.

## Does "what makes a hit" differ by genre?

Yes, clearly. Evaluating the same global model within each genre's held-out
tracks:

| Genre | Test tracks | RMSE | Mean popularity | Top SHAP feature |
|---|---|---|---|---|
| metal | 46 | 26.4 | 56.8 | instrumentalness |
| edm | 139 | 30.3 | 41.6 | valence |
| pop | 83 | 33.8 | 41.1 | valence |
| hip-hop | 168 | 29.0 | 41.0 | danceability |
| acoustic | 200 | 18.5 | 40.9 | valence |
| classical | 173 | 20.0 | 11.9 | acousticness |

Two things stand out. First, the **top driver genuinely changes**:
danceability leads hip-hop, acousticness dominates classical (unsurprising,
but good confirmation the model isn't just repeating its global ranking
verbatim), and valence shows up repeatedly for mood-driven genres.
Second, **pop has the highest RMSE of any focus genre** (33.8) despite
being one of the most-streamed genres in the dataset — a reasonable read is
that pop popularity is disproportionately driven by exactly the
off-dataset factors flagged above (artist fame, marketing, playlist
placement), so audio features alone explain the least about it.

## Dashboard

A dark, editorial, music-player-styled Streamlit app
(`src/what_makes_a_hit/dashboard/app.py`) — designed to be legible to someone
who has never seen a SHAP value:

- **01 · Studio** — shape a track with grouped dials (feel / texture / bones),
  start from a genre's typical sound, or load a real track and **listen to it
  in-page** via Spotify's embed player — with an honesty readout comparing the
  model's audio-only guess against Spotify's actual score. A "now playing"
  card renders the prediction as a seek-bar with hit odds, a radar overlays
  your track against the median top-20% hit, "chase the hit" lets the model
  greedily nudge your dials toward a higher score, and "sounds like" surfaces
  the nearest real tracks by audio distance — all playable.
- **02 · What matters** — plain-language SHAP takeaways (directions computed
  from the data, not asserted), the global importance chart, and a per-track
  beeswarm with a how-to-read-it caption.
- **03 · Genres** — popularity and model error by genre, a genre × feature
  SHAP heatmap, and the most popular track per focus genre, playable in-page.
- **04 · Under the hood** — the honest scoreboard in plain english (what ±15
  points means, what AUC 0.72 means), full model comparisons,
  precision-recall curves, predicted-vs-actual scatter, and data notes.

Run it locally:

```bash
streamlit run src/what_makes_a_hit/dashboard/app.py
```

## Project structure

```
what-makes-a-hit/
├── config.py                          # feature list, split params, paths
├── data/
│   ├── raw/                           # downloaded Kaggle CSV (gitignored)
│   └── processed/                     # cleaned data, models, metrics, SHAP values
├── src/what_makes_a_hit/
│   ├── data/
│   │   ├── load_data.py               # load + clean raw CSV
│   │   └── build_features.py          # feature selection, genre-stratified split
│   ├── modeling/
│   │   ├── train_regression.py        # linear regression vs. XGBoost
│   │   ├── train_classification.py    # logistic regression vs. XGBoost
│   │   └── interpret.py               # SHAP values for the regressor
│   ├── analysis/
│   │   └── genre_breakdown.py         # per-genre performance + SHAP
│   └── dashboard/
│       ├── app.py                     # Streamlit dashboard
│       └── theme.py                   # shared color tokens / chart styling
├── scripts/
│   └── run_pipeline.py                # run the full pipeline end-to-end
└── tests/
```

## Running it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Kaggle credentials must already be configured at ~/.kaggle/kaggle.json
mkdir -p data/raw
cd data/raw && kaggle datasets download -d maharshipandya/-spotify-tracks-dataset \
  && unzip -o ./*spotify-tracks-dataset.zip && cd ../..

python scripts/run_pipeline.py     # runs the full pipeline, prints summary metrics
pytest tests/                      # sanity tests
streamlit run src/what_makes_a_hit/dashboard/app.py
```

## Interview pitch

I built a model predicting Spotify track popularity from audio features
alone — danceability, energy, tempo, and similar characteristics — across a
real dataset of about 90,000 unique tracks spanning 113 genres. I compared a
linear baseline against a gradient boosting model with proper held-out
evaluation, and used SHAP values to show which features actually drive the
prediction rather than treating the model as a black box. I'm upfront that
audio features alone can't fully explain real-world popularity — artist fame
and marketing matter too — but the project quantifies how much of it audio
characteristics alone can predict, and which ones matter most, and shows that
answer shifts meaningfully by genre.
