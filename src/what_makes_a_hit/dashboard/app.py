"""What Makes a Hit? -- interactive Streamlit dashboard.

Loads precomputed pipeline artifacts (models, SHAP values, genre breakdown)
plus the cleaned track table, and never re-runs training or SHAP itself --
that keeps the deployed app fast and dependency-light (see requirements.txt).
Run with: streamlit run src/what_makes_a_hit/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Streamlit Cloud runs this file directly with only requirements.txt
# installed (no `pip install -e .`), so `what_makes_a_hit` isn't
# necessarily on sys.path as an installed package -- add this file's own
# directory (for `theme`) and the project root (for the root-level
# `config` module) explicitly rather than relying on that.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import config
import theme

NOTE_NAMES = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]

st.set_page_config(page_title="What Makes a Hit?", page_icon="\U0001f3a7", layout="wide", initial_sidebar_state="expanded")


# ============================================================ styling ====
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 15% 0%, #1a0f24 0%, #0d0d0d 38%, #0d0d0d 100%);
        }
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}

        .hero-title {
            font-size: 3.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0;
            background: linear-gradient(90deg, #1DB954 0%, #3987e5 45%, #9085e9 75%, #d55181 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            color: #c3c2b7;
            font-size: 1.05rem;
            max-width: 760px;
            margin-top: 0.3rem;
        }

        .eq { display: flex; align-items: flex-end; gap: 4px; height: 28px; margin: 0.6rem 0 1.6rem 0; }
        .eq span {
            display: block; width: 4px; border-radius: 2px;
            background: linear-gradient(180deg, #1DB954, #3987e5);
            animation: eq-bounce 1.1s ease-in-out infinite;
        }
        .eq span:nth-child(1) { height: 30%; animation-delay: -1.0s; }
        .eq span:nth-child(2) { height: 65%; animation-delay: -0.8s; }
        .eq span:nth-child(3) { height: 100%; animation-delay: -0.6s; }
        .eq span:nth-child(4) { height: 45%; animation-delay: -0.4s; }
        .eq span:nth-child(5) { height: 80%; animation-delay: -0.9s; }
        .eq span:nth-child(6) { height: 55%; animation-delay: -0.2s; }
        .eq span:nth-child(7) { height: 90%; animation-delay: -0.5s; }
        .eq span:nth-child(8) { height: 35%; animation-delay: -0.1s; }
        .eq span:nth-child(9) { height: 70%; animation-delay: -0.7s; }
        .eq span:nth-child(10) { height: 40%; animation-delay: -0.3s; }
        @keyframes eq-bounce {
            0%, 100% { transform: scaleY(0.35); opacity: 0.75; }
            50% { transform: scaleY(1); opacity: 1; }
        }

        .hit-card {
            background: #1a1a19;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
        }
        .stat-row { display: flex; gap: 0.9rem; flex-wrap: wrap; margin: 0.4rem 0 1.8rem 0; }
        .stat-tile {
            flex: 1 1 200px;
            background: #1a1a19;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.9rem 1.1rem;
        }
        .stat-tile .label { color: #898781; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
        .stat-tile .value { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin-top: 0.15rem; }
        .stat-tile .sub { color: #c3c2b7; font-size: 0.82rem; margin-top: 0.1rem; }

        .track-row {
            display: flex; align-items: center; gap: 0.8rem;
            padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .track-row:last-child { border-bottom: none; }
        .track-rank { color: #898781; font-weight: 700; width: 1.4rem; }
        .track-meta { flex: 1; min-width: 0; }
        .track-name { color: #ffffff; font-weight: 600; font-size: 0.92rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .track-artist { color: #898781; font-size: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .genre-pill {
            display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 600; color: #0d0d0d; white-space: nowrap;
        }
        .pop-badge { color: #c3c2b7; font-size: 0.85rem; font-variant-numeric: tabular-nums; width: 2.4rem; text-align: right; }

        .hit-badge {
            display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
            font-weight: 700; font-size: 0.95rem;
        }

        [data-testid="stMetricValue"] { color: #ffffff; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); }
        .stTabs [data-baseweb="tab"] { color: #898781; font-weight: 600; }
        .stTabs [aria-selected="true"] { color: #1DB954 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def equalizer_html() -> str:
    return '<div class="eq">' + "".join("<span></span>" for _ in range(10)) + "</div>"


def stat_tile(label: str, value: str, sub: str = "") -> str:
    return f"""<div class="stat-tile"><div class="label">{label}</div>
        <div class="value">{value}</div><div class="sub">{sub}</div></div>"""


# ============================================================ data i/o ====
REQUIRED_ARTIFACTS = [
    config.TRACKS_CLEAN_PARQUET,
    config.TEST_PARQUET,
    config.REGRESSION_RESULTS_JSON,
    config.CLASSIFICATION_RESULTS_JSON,
    config.SHAP_VALUES_PARQUET,
    config.GENRE_BREAKDOWN_CSV,
    config.LINEAR_MODEL_PATH,
    config.GBM_REGRESSOR_PATH,
    config.GBM_CLASSIFIER_PATH,
]


@st.cache_data
def load_tracks() -> pd.DataFrame:
    return pd.read_parquet(config.TRACKS_CLEAN_PARQUET)


@st.cache_data
def load_test() -> pd.DataFrame:
    return pd.read_parquet(config.TEST_PARQUET)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data
def load_shap() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(config.SHAP_VALUES_PARQUET), pd.read_parquet(config.SHAP_SAMPLE_PARQUET)


@st.cache_data
def load_genre_breakdown() -> pd.DataFrame:
    return pd.read_csv(config.GENRE_BREAKDOWN_CSV)


@st.cache_data
def load_pr_curve() -> pd.DataFrame:
    return pd.read_parquet(config.PR_CURVE_PARQUET)


@st.cache_resource
def load_models() -> dict:
    return {
        "linear": joblib.load(config.LINEAR_MODEL_PATH),
        "xgb_reg": joblib.load(config.GBM_REGRESSOR_PATH),
        "logistic": joblib.load(config.LOGISTIC_MODEL_PATH),
        "xgb_clf": joblib.load(config.GBM_CLASSIFIER_PATH),
    }


@st.cache_resource
def build_nn_index(tracks_df: pd.DataFrame) -> tuple[NearestNeighbors, StandardScaler]:
    # Over-fetch neighbors: this dataset has occasional re-released tracks
    # (same track_name/artists, different track_id) that would otherwise
    # show up as visually duplicate rows in the "sounds like" list.
    scaler = StandardScaler().fit(tracks_df[config.AUDIO_FEATURES])
    nn = NearestNeighbors(n_neighbors=20).fit(scaler.transform(tracks_df[config.AUDIO_FEATURES]))
    return nn, scaler


# ============================================================ state ====
def apply_genre_preset(tracks_df: pd.DataFrame) -> None:
    genre = st.session_state["genre_preset"]
    if genre == "Custom":
        return
    genre_rows = tracks_df[tracks_df[config.GENRE_COLUMN] == genre]
    medians = genre_rows[config.DASHBOARD_SLIDER_FEATURES].median()
    for feat in config.DASHBOARD_SLIDER_FEATURES:
        st.session_state[f"slider_{feat}"] = float(round(medians[feat], 3))
    st.session_state["slider_key"] = int(genre_rows["key"].mode().iloc[0])
    st.session_state["mode_choice"] = "Major" if genre_rows["mode"].mode().iloc[0] == 1 else "Minor"
    st.session_state["time_signature_choice"] = int(genre_rows["time_signature"].mode().iloc[0]) if genre_rows["time_signature"].mode().iloc[0] in (3, 4, 5) else 4
    st.session_state["duration_minutes"] = round(float(genre_rows["duration_ms"].median()) / 60000, 2)


def surprise_me(tracks_df: pd.DataFrame) -> None:
    row = tracks_df.sample(1, random_state=np.random.randint(0, 1_000_000)).iloc[0]
    for feat in config.DASHBOARD_SLIDER_FEATURES:
        st.session_state[f"slider_{feat}"] = float(round(row[feat], 3))
    st.session_state["slider_key"] = int(row["key"])
    st.session_state["mode_choice"] = "Major" if row["mode"] == 1 else "Minor"
    st.session_state["time_signature_choice"] = int(row["time_signature"]) if row["time_signature"] in (3, 4, 5) else 4
    st.session_state["duration_minutes"] = round(float(row["duration_ms"]) / 60000, 2)
    st.session_state["explicit_choice"] = bool(row["explicit"])
    st.session_state["genre_preset"] = "Custom"
    st.session_state["surprise_track"] = f"{row['track_name']} — {row['artists']}"


def init_state(tracks_df: pd.DataFrame) -> None:
    overall_median = tracks_df[config.DASHBOARD_SLIDER_FEATURES].median()
    for feat in config.DASHBOARD_SLIDER_FEATURES:
        st.session_state.setdefault(f"slider_{feat}", float(round(overall_median[feat], 3)))
    st.session_state.setdefault("slider_key", 0)
    st.session_state.setdefault("mode_choice", "Major")
    st.session_state.setdefault("time_signature_choice", 4)
    st.session_state.setdefault("explicit_choice", False)
    st.session_state.setdefault("duration_minutes", 3.5)
    st.session_state.setdefault("genre_preset", "Custom")


def current_input_row() -> pd.DataFrame:
    values = {feat: st.session_state[f"slider_{feat}"] for feat in config.DASHBOARD_SLIDER_FEATURES}
    values["key"] = st.session_state["slider_key"]
    values["mode"] = 1 if st.session_state["mode_choice"] == "Major" else 0
    values["time_signature"] = st.session_state["time_signature_choice"]
    values["duration_ms"] = int(st.session_state["duration_minutes"] * 60000)
    values["explicit"] = int(st.session_state["explicit_choice"])
    return pd.DataFrame([values])[config.AUDIO_FEATURES]


# ============================================================ tabs ====
def render_predictor_tab(tracks_df: pd.DataFrame, models: dict) -> None:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### \U0001f39b️ Produce a track")
        genre_options = ["Custom"] + config.FOCUS_GENRES
        st.selectbox(
            "Start from a genre's typical profile",
            options=genre_options,
            key="genre_preset",
            on_change=apply_genre_preset,
            args=(tracks_df,),
        )
        st.button("\U0001f3b2 Surprise me (load a real track)", on_click=surprise_me, args=(tracks_df,))
        if st.session_state.get("surprise_track"):
            st.caption(f"Loaded: **{st.session_state['surprise_track']}**")

        st.slider("Danceability", 0.0, 1.0, step=0.01, key="slider_danceability")
        st.slider("Energy", 0.0, 1.0, step=0.01, key="slider_energy")
        st.slider("Valence (musical positivity)", 0.0, 1.0, step=0.01, key="slider_valence")
        st.slider("Acousticness", 0.0, 1.0, step=0.01, key="slider_acousticness")
        st.slider("Instrumentalness", 0.0, 1.0, step=0.01, key="slider_instrumentalness")
        st.slider("Speechiness", 0.0, 1.0, step=0.01, key="slider_speechiness")
        st.slider("Liveness", 0.0, 1.0, step=0.01, key="slider_liveness")
        st.slider("Loudness (dB)", -45.0, 3.0, step=0.5, key="slider_loudness")
        st.slider("Tempo (BPM)", 40.0, 220.0, step=1.0, key="slider_tempo")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox("Key", options=list(range(12)), format_func=lambda k: NOTE_NAMES[k], key="slider_key")
        with c2:
            st.radio("Mode", options=["Major", "Minor"], key="mode_choice", horizontal=True)
        with c3:
            st.selectbox("Time signature", options=[3, 4, 5], format_func=lambda v: f"{v}/4", key="time_signature_choice")
        d1, d2 = st.columns(2)
        with d1:
            st.slider("Duration (minutes)", 1.0, 8.0, step=0.25, key="duration_minutes")
        with d2:
            st.toggle("Explicit", key="explicit_choice")

    input_row = current_input_row()
    pred_xgb = float(np.clip(models["xgb_reg"].predict(input_row)[0], 0, 100))
    pred_linear = float(np.clip(models["linear"].predict(input_row)[0], 0, 100))
    hit_proba = float(models["xgb_clf"].predict_proba(input_row)[0][1])

    with right:
        st.markdown("#### \U0001f3af Predicted popularity")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred_xgb,
                domain={"x": [0.08, 0.92], "y": [0, 1]},
                number={"suffix": " / 100", "font": {"color": theme.PRIMARY_INK, "size": 42}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": theme.MUTED_INK, "tickfont": {"color": theme.MUTED_INK}},
                    "bar": {"color": theme.BRAND_GREEN, "thickness": 0.35},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 20], "color": "rgba(208,59,59,0.18)"},
                        {"range": [20, 52], "color": "rgba(250,178,25,0.14)"},
                        {"range": [52, 100], "color": "rgba(12,163,12,0.14)"},
                    ],
                },
            )
        )
        gauge = theme.apply_theme(gauge, height=260)
        gauge.update_layout(margin=dict(l=40, r=40, t=30, b=10))
        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

        delta = pred_xgb - pred_linear
        st.caption(f"XGBoost model · linear baseline would say **{pred_linear:.1f}** ({'+' if delta >= 0 else ''}{delta:.1f})")

        if hit_proba >= 0.5:
            badge_color, badge_bg, label = theme.STATUS_GOOD, "rgba(12,163,12,0.18)", "\U0001f525 Likely a hit"
        elif hit_proba >= 0.2:
            badge_color, badge_bg, label = theme.STATUS_WARNING, "rgba(250,178,25,0.18)", "\U0001f914 Could go either way"
        else:
            badge_color, badge_bg, label = theme.STATUS_CRITICAL, "rgba(208,59,59,0.18)", "❄️ Unlikely to chart"
        st.markdown(
            f'<span class="hit-badge" style="color:{badge_color}; background:{badge_bg};">{label} — {hit_proba * 100:.0f}% chance of top-20% popularity</span>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### \U0001f3a7 Sounds like")
        nn, scaler = build_nn_index(tracks_df)
        distances, indices = nn.kneighbors(scaler.transform(input_row))
        neighbors = tracks_df.iloc[indices[0]].drop_duplicates(subset=["track_name", "artists"]).head(5)

        rows_html = ""
        for rank, (_, row) in enumerate(neighbors.iterrows(), start=1):
            color = theme.GENRE_COLORS.get(row[config.GENRE_COLUMN], theme.MUTED_INK)
            rows_html += f"""<div class="track-row">
                <div class="track-rank">{rank}</div>
                <div class="track-meta">
                    <div class="track-name">{row['track_name']}</div>
                    <div class="track-artist">{row['artists']}</div>
                </div>
                <span class="genre-pill" style="background:{color};">{row[config.GENRE_COLUMN]}</span>
                <span class="pop-badge">{int(row['popularity'])}</span>
            </div>"""
        st.markdown(f'<div class="hit-card">{rows_html}</div>', unsafe_allow_html=True)


def render_shap_tab(shap_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    st.markdown("#### Which audio features actually move the prediction?")
    st.caption(
        "SHAP values for the XGBoost regressor on 2,000 held-out tracks. Each dot is one track; its horizontal "
        "position is how much that feature pushed the predicted popularity up or down for that specific track."
    )

    importance = shap_df.abs().mean().sort_values(ascending=False)

    bar = go.Figure(go.Bar(x=importance.values[::-1], y=importance.index[::-1], orientation="h", marker_color=theme.BRAND_GREEN))
    bar.update_layout(title="Global importance (mean |SHAP value|)", xaxis_title="mean |SHAP value|")
    st.plotly_chart(theme.apply_theme(bar, height=420, show_legend=False), width="stretch")

    features_sorted = importance.index.tolist()
    rng = np.random.default_rng(config.RANDOM_SEED)
    xs, ys, colors = [], [], []
    for i, feat in enumerate(features_sorted):
        vals = shap_df[feat].to_numpy()
        raw = sample_df[feat].to_numpy(dtype=float)
        span = raw.max() - raw.min()
        norm = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
        y_pos = len(features_sorted) - 1 - i
        jitter = rng.uniform(-0.35, 0.35, size=len(vals))
        xs.append(vals)
        ys.append(np.full(len(vals), y_pos) + jitter)
        colors.append(norm)

    beeswarm = go.Figure(
        go.Scatter(
            x=np.concatenate(xs),
            y=np.concatenate(ys),
            mode="markers",
            marker=dict(
                size=5,
                color=np.concatenate(colors),
                colorscale=theme.SHAP_COLORSCALE,
                showscale=True,
                colorbar=dict(title="Feature<br>value", tickvals=[0, 1], ticktext=["Low", "High"], thickness=14, len=0.6),
                opacity=0.6,
                line=dict(width=0),
            ),
        )
    )
    beeswarm.update_yaxes(tickmode="array", tickvals=list(range(len(features_sorted))), ticktext=list(reversed(features_sorted)))
    beeswarm.update_layout(title="Feature impact per track", xaxis_title="SHAP value (impact on predicted popularity)")
    beeswarm.add_vline(x=0, line_width=1, line_color=theme.AXIS_LINE)
    st.plotly_chart(theme.apply_theme(beeswarm, height=520, show_legend=False), width="stretch")


def render_genre_tab(genre_breakdown_df: pd.DataFrame, genre_shap: dict) -> None:
    st.markdown("#### Does 'what makes a hit' change by genre?")
    st.caption("Same global XGBoost model, evaluated and explained separately within each genre's held-out tracks.")

    col1, col2 = st.columns(2)
    genres = genre_breakdown_df["genre"].tolist()
    colors = [theme.GENRE_COLORS.get(g, theme.MUTED_INK) for g in genres]

    with col1:
        fig = go.Figure(go.Bar(x=genres, y=genre_breakdown_df["mean_popularity"], marker_color=colors))
        fig.update_layout(title="Average popularity by genre", yaxis_title="popularity (0-100)")
        st.plotly_chart(theme.apply_theme(fig, height=380, show_legend=False), width="stretch")

    with col2:
        fig = go.Figure(go.Bar(x=genres, y=genre_breakdown_df["rmse"], marker_color=colors))
        fig.update_layout(title="Model error (RMSE) by genre", yaxis_title="RMSE")
        st.plotly_chart(theme.apply_theme(fig, height=380, show_legend=False), width="stretch")

    st.markdown("##### Top SHAP feature by genre")
    chip_html = ""
    for _, row in genre_breakdown_df.iterrows():
        color = theme.GENRE_COLORS.get(row["genre"], theme.MUTED_INK)
        chip_html += f"""<div class="stat-tile" style="flex:1 1 150px;">
            <div class="label" style="color:{color};">{row['genre']}</div>
            <div class="value" style="font-size:1.1rem;">{row['top_feature']}</div>
            <div class="sub">n={int(row['n_test'])} test tracks</div></div>"""
    st.markdown(f'<div class="stat-row">{chip_html}</div>', unsafe_allow_html=True)

    all_features = importance_order = config.AUDIO_FEATURES
    ordered_features = sorted(all_features, key=lambda f: -np.mean([genre_shap[g].get(f, 0) for g in genres]))
    z = [[genre_shap[g].get(f, 0) for f in ordered_features] for g in genres]
    heatmap = go.Figure(
        go.Heatmap(
            z=z,
            x=ordered_features,
            y=genres,
            colorscale=theme.SEQUENTIAL_GREEN,
            colorbar=dict(title="mean |SHAP|", thickness=14),
        )
    )
    heatmap.update_layout(title="Feature importance heatmap by genre")
    st.plotly_chart(theme.apply_theme(heatmap, height=380, show_legend=False), width="stretch")


def render_scoreboard_tab(reg_results: dict, clf_results: dict, pr_curve_df: pd.DataFrame, test_df: pd.DataFrame, models: dict) -> None:
    st.markdown("#### Regression: predicting the raw 0-100 popularity score")
    reg_models = [
        ("baseline", "Baseline (mean)", reg_results["baseline_mean"]),
        ("linear_regression", "Linear regression", reg_results["linear_regression"]),
        ("xgboost_regressor", "XGBoost", reg_results["xgboost_regressor"]),
    ]
    fig = go.Figure()
    for key, label, metrics in reg_models:
        fig.add_trace(go.Bar(name=label, x=["RMSE", "MAE"], y=[metrics["rmse"], metrics["mae"]], marker_color=theme.MODEL_COLORS[key]))
    fig.update_layout(barmode="group", title="Held-out test set error (lower is better)", yaxis_title="popularity points")
    st.plotly_chart(theme.apply_theme(fig, height=380), width="stretch")

    st.markdown("#### Classification: is this track in the top 20%?")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        clf_models = [
            ("logistic_regression", "Logistic regression", clf_results["logistic_regression"]),
            ("xgboost_classifier", "XGBoost", clf_results["xgboost_classifier"]),
        ]
        fig = go.Figure()
        for key, label, metrics in clf_models:
            fig.add_trace(go.Bar(name=label, x=["AUC", "Avg. Precision"], y=[metrics["auc"], metrics["average_precision"]], marker_color=theme.MODEL_COLORS[key]))
        fig.update_layout(barmode="group", title="Held-out test set (higher is better)", yaxis=dict(range=[0, 1]))
        st.plotly_chart(theme.apply_theme(fig, height=380), width="stretch")

    with col2:
        fig = go.Figure()
        for model_key, label in [("logistic_regression", "Logistic regression"), ("xgboost_classifier", "XGBoost")]:
            curve = pr_curve_df[pr_curve_df["model"] == model_key].sort_values("recall")
            fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines", name=label, line=dict(color=theme.MODEL_COLORS[model_key], width=2.5)))
        fig.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision", yaxis=dict(range=[0, 1]))
        st.plotly_chart(theme.apply_theme(fig, height=380), width="stretch")

    st.markdown("#### Predicted vs. actual popularity (XGBoost regressor)")
    sample = test_df.sample(n=min(1500, len(test_df)), random_state=config.RANDOM_SEED)
    preds = models["xgb_reg"].predict(sample[config.AUDIO_FEATURES])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample[config.TARGET_COLUMN], y=preds, mode="markers", name="Test tracks", marker=dict(color=theme.BRAND_GREEN, size=5, opacity=0.35)))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect prediction (y = x)", line=dict(color=theme.MUTED_INK, dash="dash", width=1.5)))
    fig.update_layout(xaxis_title="Actual popularity", yaxis_title="Predicted popularity")
    st.plotly_chart(theme.apply_theme(fig, height=440), width="stretch")


# ============================================================ main ====
def main() -> None:
    missing = [p for p in REQUIRED_ARTIFACTS if not Path(p).exists()]
    if missing:
        st.error("Pipeline artifacts are missing. Run `python scripts/run_pipeline.py` first.")
        st.code("\n".join(str(p) for p in missing))
        st.stop()

    inject_css()

    tracks_df = load_tracks()
    test_df = load_test()
    reg_results = load_json(config.REGRESSION_RESULTS_JSON)
    clf_results = load_json(config.CLASSIFICATION_RESULTS_JSON)
    cleaning_report = load_json(config.DATA_PROCESSED_DIR / "cleaning_report.json")
    shap_df, sample_df = load_shap()
    genre_breakdown_df = load_genre_breakdown()
    genre_shap = load_json(config.GENRE_SHAP_JSON)
    pr_curve_df = load_pr_curve()
    models = load_models()

    init_state(tracks_df)

    with st.sidebar:
        st.markdown("### \U0001f3a7 What Makes a Hit?")
        st.caption("Predicting Spotify track popularity from audio features alone.")
        st.markdown("---")
        st.markdown(
            f"**{cleaning_report['final_rows']:,}** unique tracks \n"
            f"**{tracks_df[config.GENRE_COLUMN].nunique()}** genres \n"
            f"Source: Kaggle -- maharshipandya/spotify-tracks-dataset"
        )
        with st.expander("About the data & limitations"):
            st.markdown(
                f"""
Raw file: **{cleaning_report['raw_rows']:,}** rows.

- Dropped {cleaning_report['dropped_invalid_duration_or_missing_meta']} row(s) with invalid duration / missing metadata
- Dropped {cleaning_report['dropped_duplicate_track_id']:,} duplicate rows (same track re-queried under multiple genres, identical audio features)
- **{cleaning_report['final_rows']:,}** clean, unique tracks used for modeling

**Honest limitation:** audio features alone can't fully explain real-world
popularity -- artist fame, marketing spend, and playlist placement aren't in
this dataset and plausibly explain a meaningful share of it. This project
measures how much *audio characteristics alone* can predict, not a complete
model of what makes a song a hit.

This dataset is also a **static snapshot**, not a live feed -- popularity
reflects whenever it was collected, not real-time streams. Genre labels are
Spotify's own categorization, not an objective ground truth.
                """
            )

    st.markdown('<div class="hero-title">WHAT MAKES A HIT?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Predicting Spotify popularity from audio DNA alone -- danceability, '
        "energy, valence, and more -- across ~90K real tracks spanning 113 genres. Two models, "
        "honestly benchmarked against a naive baseline, explained with SHAP.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(equalizer_html(), unsafe_allow_html=True)

    rmse_improvement = (1 - reg_results["xgboost_regressor"]["rmse"] / reg_results["baseline_mean"]["rmse"]) * 100
    stats_html = (
        stat_tile("Tracks modeled", f"{cleaning_report['final_rows']:,}", "after de-duplication")
        + stat_tile("Genres", f"{tracks_df[config.GENRE_COLUMN].nunique()}", "pop to classical to metal")
        + stat_tile("XGBoost RMSE", f"{reg_results['xgboost_regressor']['rmse']:.1f} pts", f"{rmse_improvement:.0f}% better than baseline")
        + stat_tile("Hit-detection AUC", f"{clf_results['xgboost_classifier']['auc']:.2f}", "top 20% popularity, held-out")
    )
    st.markdown(f'<div class="stat-row">{stats_html}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["\U0001f39b️ Hit Predictor", "\U0001f9ec What Matters", "\U0001f3b8 By Genre", "\U0001f4ca Scoreboard"])
    with tab1:
        render_predictor_tab(tracks_df, models)
    with tab2:
        render_shap_tab(shap_df, sample_df)
    with tab3:
        render_genre_tab(genre_breakdown_df, genre_shap)
    with tab4:
        render_scoreboard_tab(reg_results, clf_results, pr_curve_df, test_df, models)


if __name__ == "__main__":
    main()
