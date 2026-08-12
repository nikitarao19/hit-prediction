"""What Makes a Hit? -- interactive Streamlit dashboard.

Loads precomputed pipeline artifacts (models, SHAP values, genre breakdown)
plus the cleaned track table, and never re-runs training or SHAP itself --
that keeps the deployed app fast and dependency-light (see requirements.txt).
Listening is handled by Spotify's public embed player (open.spotify.com/embed),
which needs no API credentials -- the dataset's track_id column is enough.
Run with: streamlit run src/what_makes_a_hit/dashboard/app.py
"""

from __future__ import annotations

import json
import math
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

SLIDER_BOUNDS = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "speechiness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "loudness": (-45.0, 3.0),
    "tempo": (40.0, 220.0),
}
CHASE_STEPS = {"loudness": 2.0, "tempo": 8.0}
CHASE_DEFAULT_STEP = 0.08

SLIDER_HELP = {
    "danceability": "How suited the track is for dancing -- steady beat, strong groove.",
    "energy": "Perceived intensity: loud, fast, busy.",
    "valence": "Musical positivity. High = sounds happy, low = sounds moody.",
    "acousticness": "Confidence the track is acoustic rather than electronic.",
    "instrumentalness": "Likelihood the track has no vocals.",
    "speechiness": "How much spoken word is in the mix.",
    "liveness": "Probability it was recorded in front of an audience.",
    "loudness": "Overall mastering level in decibels.",
    "tempo": "Speed, in beats per minute.",
}

st.set_page_config(page_title="What makes a hit?", page_icon="\U0001f3a7", layout="wide", initial_sidebar_state="collapsed")


# ============================================================ styling ====
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        .stApp {
            background: #0d0d0d;
            font-family: 'Space Grotesk', system-ui, sans-serif;
        }
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 3rem; padding-bottom: 4rem; max-width: 1180px;}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            font-family: 'Space Grotesk', system-ui, sans-serif;
            letter-spacing: -0.02em;
        }
        .stApp p, .stApp li, .stApp label { font-family: 'Space Grotesk', system-ui, sans-serif; }

        .mono {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #898781;
        }
        .mono-green { color: #1DB954; }

        .headline {
            font-size: 3.4rem;
            font-weight: 700;
            letter-spacing: -0.045em;
            line-height: 1.02;
            color: #ffffff;
            margin: 0.5rem 0 0.9rem 0;
        }
        .headline em { font-style: normal; color: #1DB954; }
        .lede {
            color: #c7c5bd;
            font-size: 1.06rem;
            line-height: 1.55;
            max-width: 620px;
            margin-bottom: 1.4rem;
        }

        .eq { display: inline-flex; align-items: flex-end; gap: 3px; height: 22px; margin-left: 4px; }
        .eq span {
            display: block; width: 3px; border-radius: 1.5px; background: #1DB954;
            animation: eq-bounce 1.2s ease-in-out infinite;
        }
        .eq span:nth-child(1) { height: 35%; animation-delay: -1.0s; opacity:.9; }
        .eq span:nth-child(2) { height: 70%; animation-delay: -0.7s; opacity:.7; }
        .eq span:nth-child(3) { height: 100%; animation-delay: -0.4s; opacity:.9; }
        .eq span:nth-child(4) { height: 50%; animation-delay: -0.9s; opacity:.6; }
        .eq span:nth-child(5) { height: 85%; animation-delay: -0.2s; opacity:.8; }
        .eq span:nth-child(6) { height: 40%; animation-delay: -0.6s; opacity:.7; }
        @keyframes eq-bounce {
            0%, 100% { transform: scaleY(0.3); }
            50% { transform: scaleY(1); }
        }

        .statline {
            display: flex; flex-wrap: wrap;
            border-top: 1px solid rgba(255,255,255,0.10);
            border-bottom: 1px solid rgba(255,255,255,0.10);
            margin: 0.4rem 0 1.6rem 0;
        }
        .statline > div {
            flex: 1 1 180px; padding: 0.85rem 1.4rem 0.85rem 0;
        }
        .statline > div + div { border-left: 1px solid rgba(255,255,255,0.10); padding-left: 1.4rem; }
        .statline .k { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.13em; text-transform: uppercase; color: #898781; }
        .statline .v { color: #ffffff; font-size: 1.35rem; font-weight: 600; margin-top: 0.2rem; letter-spacing: -0.01em; }
        .statline .v small { color: #898781; font-size: 0.78rem; font-weight: 400; margin-left: 0.35rem; }

        .steps { display: flex; gap: 0; flex-wrap: wrap; margin: 0.2rem 0 1.6rem 0; }
        .steps > div { flex: 1 1 220px; padding: 0.2rem 1.4rem 0.2rem 0; }
        .steps > div + div { border-left: 1px solid rgba(255,255,255,0.10); padding-left: 1.4rem; }
        .steps .n { font-family: 'IBM Plex Mono', monospace; color: #1DB954; font-size: 0.72rem; letter-spacing: 0.12em; }
        .steps .t { color: #c7c5bd; font-size: 0.9rem; line-height: 1.45; margin-top: 0.25rem; }

        .grouplabel {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
            color: #898781; margin: 1.3rem 0 0.1rem 0;
            display: flex; align-items: center; gap: 0.6rem;
        }
        .grouplabel::after { content: ""; flex: 1; height: 1px; background: rgba(255,255,255,0.08); }

        .np-card {
            background: #141413;
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 14px;
            padding: 1.15rem 1.25rem 1.3rem 1.25rem;
        }
        .np-head { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.1rem; }
        .np-art {
            width: 74px; height: 74px; border-radius: 9px; flex: none;
            background: #0d0d0d; border: 1px solid rgba(255,255,255,0.07);
            display: flex; align-items: flex-end; justify-content: center;
            gap: 2px; padding: 10px 8px; overflow: hidden;
        }
        .np-art span { display: block; width: 3px; border-radius: 1.5px; background: #1DB954; }
        .np-title { color: #fff; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em; }
        .np-artist { color: #898781; font-size: 0.85rem; margin-top: 0.15rem; }

        .seek-num { color: #ffffff; font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; }
        .seek-den { color: #898781; font-size: 0.85rem; font-weight: 400; letter-spacing: 0; margin-left: 0.45rem; }
        .seek-track { position: relative; height: 5px; border-radius: 3px; background: rgba(255,255,255,0.12); margin: 1.05rem 0 0.4rem 0; }
        .seek-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: #1DB954; }
        .seek-knob {
            position: absolute; top: 50%; width: 11px; height: 11px; border-radius: 50%;
            background: #ffffff; transform: translate(-50%, -50%);
        }
        .seek-ends { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #898781; }
        .seek-note { color: #898781; font-size: 0.82rem; margin-top: 0.7rem; }
        .seek-note b { color: #c7c5bd; font-weight: 500; }

        .verdict { display: flex; align-items: center; gap: 0.55rem; margin-top: 0.9rem; color: #c7c5bd; font-size: 0.92rem; }
        .verdict .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
        .verdict b { color: #ffffff; font-weight: 600; }

        .match-cap { display: flex; align-items: center; gap: 0.6rem; margin: 0.9rem 0 0.35rem 0; }
        .pill {
            display: inline-block; padding: 0.08rem 0.55rem; border-radius: 999px;
            font-size: 0.7rem; font-weight: 600; color: #0d0d0d; white-space: nowrap;
        }
        .mono-dim { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #898781; letter-spacing: 0.05em; }

        .insight { display: flex; gap: 1rem; padding: 0.75rem 0; border-bottom: 1px solid rgba(255,255,255,0.07); }
        .insight:last-child { border-bottom: none; }
        .insight .n { font-family: 'IBM Plex Mono', monospace; color: #1DB954; font-size: 0.75rem; padding-top: 0.2rem; flex: none; }
        .insight .f { color: #ffffff; font-weight: 600; font-size: 0.95rem; min-width: 150px; flex: none; }
        .insight .s { color: #c7c5bd; font-size: 0.92rem; line-height: 1.45; }

        .explain { border-left: 2px solid #1DB954; padding: 0.15rem 0 0.15rem 1rem; margin: 0.6rem 0 1.2rem 0; }
        .explain .q { color: #ffffff; font-weight: 600; font-size: 0.98rem; }
        .explain .a { color: #c7c5bd; font-size: 0.92rem; line-height: 1.5; margin-top: 0.25rem; max-width: 640px; }

        .footer-line {
            border-top: 1px solid rgba(255,255,255,0.10);
            margin-top: 3rem; padding-top: 1rem;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
            color: #898781; letter-spacing: 0.04em; line-height: 1.8;
        }
        .footer-line a { color: #c7c5bd; text-decoration: none; border-bottom: 1px solid rgba(255,255,255,0.2); }

        .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid rgba(255,255,255,0.10); }
        .stTabs [data-baseweb="tab"] {
            color: #898781; font-weight: 500;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.06em;
        }
        .stTabs [aria-selected="true"] { color: #ffffff !important; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #1DB954; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_block(markup: str) -> None:
    """st.markdown for HTML-only blocks, hardened against markdown's
    indented-code-block rule: a whitespace-only line followed by a line
    indented 4+ spaces (easy to produce with triple-quoted f-strings)
    renders as literal escaped code. Stripping indentation and blank
    lines makes the injected HTML immune to that."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def equalizer_html() -> str:
    return '<span class="eq">' + "".join("<span></span>" for _ in range(6)) + "</span>"


def waveform_art_html(feats: dict, n_bars: int = 14) -> str:
    """A deterministic little 'album art' waveform drawn from the current
    dial settings, so the untitled track always has a face."""
    energy, dance, valence = feats["energy"], feats["danceability"], feats["valence"]
    spans = ""
    for i in range(n_bars):
        base = math.sin(i * (0.55 + 0.9 * dance) + valence * 6.28)
        wobble = 0.5 * math.sin(i * 1.7 + energy * 3.14)
        h = 0.2 + 0.8 * abs(0.65 * base + 0.35 * wobble) * (0.35 + 0.65 * energy)
        opacity = 0.4 + 0.6 * h
        spans += f'<span style="height:{h * 100:.0f}%;opacity:{opacity:.2f};"></span>'
    return f'<div class="np-art">{spans}</div>'


def seek_bar_html(pred: float, linear_pred: float) -> str:
    p = max(0.0, min(100.0, pred))
    return f"""
    <div class="seek-num">{p:.0f}<span class="seek-den">/ 100 predicted popularity</span></div>
    <div class="seek-track"><div class="seek-fill" style="width:{p:.1f}%"></div><div class="seek-knob" style="left:{p:.1f}%"></div></div>
    <div class="seek-ends"><span>0</span><span>100</span></div>
    <div class="seek-note">A plain linear model would say <b>{linear_pred:.0f}</b>. The dataset average is <b>33</b>.</div>
    """


def verdict_html(hit_proba: float) -> str:
    pct = hit_proba * 100
    if hit_proba >= 0.5:
        color, text = theme.STATUS_GOOD, f"<b>Hit territory.</b> {pct:.0f}% odds of landing in the top 20%."
    elif hit_proba >= 0.2:
        color, text = theme.STATUS_WARNING, f"<b>In the hunt.</b> {pct:.0f}% odds of landing in the top 20%."
    else:
        color, text = theme.STATUS_CRITICAL, f"<b>A long shot.</b> {pct:.0f}% odds of landing in the top 20%."
    return f'<div class="verdict"><span class="dot" style="background:{color}"></span><span>{text}</span></div>'


def spotify_player(track_id: str, height: int = 80) -> None:
    # Rendered via st.markdown rather than st.components/st.iframe: markdown
    # keeps the player in the main document (no sandbox-in-sandbox) and lets
    # us set the allow attributes Spotify's player needs for playback.
    st.markdown(
        f'<iframe style="border-radius:12px;display:block;" '
        f'src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" '
        f'width="100%" height="{height}" frameBorder="0" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
        f'loading="lazy"></iframe>',
        unsafe_allow_html=True,
    )


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

RADAR_FEATURES = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"]


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


@st.cache_data
def hit_profile(hit_threshold: float) -> pd.Series:
    """Median audio profile of tracks at or above the hit threshold."""
    df = load_tracks()
    return df.loc[df[config.TARGET_COLUMN] >= hit_threshold, RADAR_FEATURES].median()


@st.cache_data
def genre_anthems() -> pd.DataFrame:
    """The single most popular track in each focus genre -- the genre's face."""
    df = load_tracks()
    df = df[df[config.GENRE_COLUMN].isin(config.FOCUS_GENRES)]
    idx = df.groupby(config.GENRE_COLUMN)[config.TARGET_COLUMN].idxmax()
    return df.loc[idx].set_index(config.GENRE_COLUMN).loc[[g for g in config.FOCUS_GENRES]]


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
def clear_loaded_track() -> None:
    st.session_state["surprise_track"] = None


def apply_genre_preset(tracks_df: pd.DataFrame) -> None:
    genre = st.session_state["genre_preset"]
    st.session_state["surprise_track"] = None
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
    st.session_state["surprise_track"] = {
        "track_id": row["track_id"],
        "name": row["track_name"],
        "artists": row["artists"],
        "popularity": int(row["popularity"]),
        "genre": row[config.GENRE_COLUMN],
    }


def chase_the_hit() -> None:
    """Greedy dial-nudging: repeatedly try small moves on each slider and
    keep whichever single move raises the XGBoost prediction the most."""
    model = load_models()["xgb_reg"]
    st.session_state["surprise_track"] = None
    st.session_state["genre_preset"] = "Custom"
    for _ in range(4):
        base_row = current_input_row()
        base_pred = float(model.predict(base_row)[0])
        candidates, meta = [], []
        for feat in config.DASHBOARD_SLIDER_FEATURES:
            lo, hi = SLIDER_BOUNDS[feat]
            step = CHASE_STEPS.get(feat, CHASE_DEFAULT_STEP)
            for delta in (-step, step):
                value = float(np.clip(st.session_state[f"slider_{feat}"] + delta, lo, hi))
                row = base_row.copy()
                row[feat] = value
                candidates.append(row)
                meta.append((feat, value))
        preds = model.predict(pd.concat(candidates, ignore_index=True))
        best = int(np.argmax(preds))
        if preds[best] <= base_pred + 1e-6:
            break
        feat, value = meta[best]
        if feat == "tempo":
            value = round(value)
        elif feat == "loudness":
            value = round(value * 2) / 2
        else:
            value = round(value, 3)
        st.session_state[f"slider_{feat}"] = float(value)


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
    st.session_state.setdefault("surprise_track", None)


def current_input_row() -> pd.DataFrame:
    values = {feat: st.session_state[f"slider_{feat}"] for feat in config.DASHBOARD_SLIDER_FEATURES}
    values["key"] = st.session_state["slider_key"]
    values["mode"] = 1 if st.session_state["mode_choice"] == "Major" else 0
    values["time_signature"] = st.session_state["time_signature_choice"]
    values["duration_ms"] = int(st.session_state["duration_minutes"] * 60000)
    values["explicit"] = int(st.session_state["explicit_choice"])
    return pd.DataFrame([values])[config.AUDIO_FEATURES]


# ============================================================ studio ====
def render_slider(feat: str, label: str, lo: float, hi: float, step: float, fmt: str | None = None) -> None:
    st.slider(label, lo, hi, step=step, key=f"slider_{feat}", help=SLIDER_HELP[feat], format=fmt, on_change=clear_loaded_track)


def render_studio(tracks_df: pd.DataFrame, models: dict) -> None:
    html_block(
        """
        <div class="steps">
            <div><div class="n">01</div><div class="t">The model studied the audio DNA of 89,740 real tracks -- never the artist, never the marketing.</div></div>
            <div><div class="n">02</div><div class="t">You shape a track with the dials below, load a genre's typical sound, or pull up a real song.</div></div>
            <div><div class="n">03</div><div class="t">It predicts the track's Spotify popularity live -- and the other tabs show its reasoning.</div></div>
        </div>
        """
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        c1, c2 = st.columns([1.4, 1], gap="small")
        with c1:
            st.selectbox(
                "Start from a genre's typical sound",
                options=["Custom"] + config.FOCUS_GENRES,
                key="genre_preset",
                on_change=apply_genre_preset,
                args=(tracks_df,),
            )
        with c2:
            st.markdown('<div style="height:1.72rem"></div>', unsafe_allow_html=True)
            st.button("Load a real track", key="btn_surprise", on_click=surprise_me, args=(tracks_df,), width="stretch")

        st.markdown('<div class="grouplabel">The feel</div>', unsafe_allow_html=True)
        render_slider("danceability", "Danceability", 0.0, 1.0, 0.01)
        render_slider("energy", "Energy", 0.0, 1.0, 0.01)
        render_slider("valence", "Valence", 0.0, 1.0, 0.01)

        st.markdown('<div class="grouplabel">The texture</div>', unsafe_allow_html=True)
        render_slider("acousticness", "Acousticness", 0.0, 1.0, 0.01)
        render_slider("instrumentalness", "Instrumentalness", 0.0, 1.0, 0.01)
        render_slider("speechiness", "Speechiness", 0.0, 1.0, 0.01)
        render_slider("liveness", "Liveness", 0.0, 1.0, 0.01)

        st.markdown('<div class="grouplabel">The bones</div>', unsafe_allow_html=True)
        render_slider("loudness", "Loudness (dB)", -45.0, 3.0, 0.5, fmt="%.1f")
        render_slider("tempo", "Tempo (BPM)", 40.0, 220.0, 1.0, fmt="%.0f")

        b1, b2, b3 = st.columns(3)
        with b1:
            st.selectbox("Key", options=list(range(12)), format_func=lambda k: NOTE_NAMES[k], key="slider_key")
        with b2:
            st.radio("Mode", options=["Major", "Minor"], key="mode_choice", horizontal=True)
        with b3:
            st.selectbox("Time signature", options=[3, 4, 5], format_func=lambda v: f"{v}/4", key="time_signature_choice")
        d1, d2 = st.columns(2)
        with d1:
            st.slider("Duration (minutes)", 1.0, 8.0, step=0.25, key="duration_minutes")
        with d2:
            st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
            st.toggle("Explicit lyrics", key="explicit_choice")

    input_row = current_input_row()
    pred_xgb = float(np.clip(models["xgb_reg"].predict(input_row)[0], 0, 100))
    pred_linear = float(np.clip(models["linear"].predict(input_row)[0], 0, 100))
    hit_proba = float(models["xgb_clf"].predict_proba(input_row)[0][1])

    loaded = st.session_state.get("surprise_track")

    with right:
        if loaded:
            genre_color = theme.GENRE_COLORS.get(loaded["genre"], theme.MUTED_INK)
            html_block(
                f"""<div class="match-cap"><span class="mono">Now playing</span>
                <span class="pill" style="background:{genre_color};">{loaded["genre"]}</span></div>"""
            )
            spotify_player(loaded["track_id"], height=152)
            gap = pred_xgb - loaded["popularity"]
            direction = "generous" if gap > 0 else "harsh"
            html_block(
                f"""<div class="seek-note" style="margin-bottom:0.8rem;">Spotify's actual score for this track is
                <b>{loaded["popularity"]}</b>. Hearing the audio alone, the model says <b>{pred_xgb:.0f}</b> --
                {"about right" if abs(gap) <= 8 else f"{abs(gap):.0f} points too {direction}"}.
                Whatever the gap, that's fame, marketing, and playlists at work: everything the model can't hear.</div>"""
            )
        else:
            html_block(
                f"""
                <div class="np-card" style="margin-bottom:1rem;">
                    <div class="np-head">
                        {waveform_art_html({f: st.session_state[f"slider_{f}"] for f in ("energy", "danceability", "valence")})}
                        <div>
                            <div class="mono" style="margin-bottom:0.3rem;">Now playing</div>
                            <div class="np-title">Untitled track</div>
                            <div class="np-artist">dialed in by you</div>
                        </div>
                    </div>
                    {seek_bar_html(pred_xgb, pred_linear)}
                    {verdict_html(hit_proba)}
                </div>
                """
            )

        if loaded:
            html_block(
                f"""
                <div class="np-card" style="margin-bottom:1rem;">
                    {seek_bar_html(pred_xgb, pred_linear)}
                    {verdict_html(hit_proba)}
                </div>
                """
            )

        st.button(
            "Chase the hit",
            key="btn_chase",
            on_click=chase_the_hit,
            help="Let the model nudge your dials toward a higher predicted score, one small move at a time.",
        )

        st.markdown('<div class="grouplabel" style="margin-top:1.6rem;">Your track vs. a typical hit</div>', unsafe_allow_html=True)
        radar = build_radar(input_row)
        st.plotly_chart(radar, width="stretch", config={"displayModeBar": False})

        st.markdown('<div class="grouplabel">Sounds like</div>', unsafe_allow_html=True)
        st.caption("The closest real tracks to your dial settings, by audio-feature distance. Press play.")
        nn, scaler = build_nn_index(tracks_df)
        _, indices = nn.kneighbors(scaler.transform(input_row))
        neighbors = tracks_df.iloc[indices[0]].drop_duplicates(subset=["track_name", "artists"]).head(3)
        for _, row in neighbors.iterrows():
            genre_color = theme.GENRE_COLORS.get(row[config.GENRE_COLUMN], theme.MUTED_INK)
            html_block(
                f"""<div class="match-cap"><span class="pill" style="background:{genre_color};">{row[config.GENRE_COLUMN]}</span>
                <span class="mono-dim">popularity {int(row["popularity"])}</span></div>"""
            )
            spotify_player(row["track_id"], height=80)


def build_radar(input_row: pd.DataFrame) -> go.Figure:
    clf_results = load_json(config.CLASSIFICATION_RESULTS_JSON)
    profile = hit_profile(clf_results["hit_threshold_popularity"])
    labels = [f.capitalize() for f in RADAR_FEATURES]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[float(profile[f]) for f in RADAR_FEATURES] + [float(profile[RADAR_FEATURES[0]])],
            theta=labels + [labels[0]],
            fill="toself",
            name="Typical top-20% hit",
            line=dict(color=theme.BRAND_GREEN, width=1.5),
            fillcolor="rgba(29,185,84,0.13)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[float(input_row[f].iloc[0]) for f in RADAR_FEATURES] + [float(input_row[RADAR_FEATURES[0]].iloc[0])],
            theta=labels + [labels[0]],
            name="Your track",
            line=dict(color="#3987e5", width=2.5),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], gridcolor=theme.GRIDLINE, tickfont=dict(color=theme.MUTED_INK, size=9), showline=False),
            angularaxis=dict(gridcolor=theme.GRIDLINE, tickfont=dict(color=theme.SECONDARY_INK, size=11), linecolor=theme.AXIS_LINE),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=theme.FONT_FAMILY, color=theme.SECONDARY_INK, size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=theme.SECONDARY_INK, size=11), orientation="h", yanchor="bottom", y=-0.22),
        margin=dict(l=45, r=45, t=30, b=10),
        height=330,
    )
    return fig


# ============================================================ what matters ====
FEATURE_STORIES = {
    "instrumentalness": {
        "down": "No vocals, no mercy. Tracks without singing are the model's single biggest red flag.",
        "up": "Unusually for mainstream data, instrumental tracks score higher here.",
    },
    "valence": {
        "down": "Moodier beats happier: lower musical positivity actually nudges predictions up.",
        "up": "Happier-sounding tracks tend to score higher.",
    },
    "acousticness": {
        "down": "Polished studio production beats raw acoustic texture, on average.",
        "up": "Acoustic texture reads as a plus to the model.",
    },
    "danceability": {
        "up": "Groove pays. The more danceable the track, the higher the model's guess.",
        "down": "Surprisingly, danceability drags predictions down in this data.",
    },
    "duration_ms": {
        "down": "Long tracks lose people. Shorter runtimes predict better.",
        "up": "Longer tracks predict better in this data.",
    },
    "energy": {
        "down": "Past a point, sheer intensity works against a track.",
        "up": "More intensity, more popularity, on average.",
    },
    "loudness": {
        "up": "Louder masters predict better -- the loudness war has a winner.",
        "down": "Quieter masters predict better here.",
    },
    "speechiness": {
        "down": "Heavy spoken-word content is a penalty outside of rap's sweet spot.",
        "up": "More spoken word, higher predictions, on average.",
    },
}


def shap_insights(shap_df: pd.DataFrame, sample_df: pd.DataFrame, top_n: int = 5) -> list[tuple[str, str]]:
    """Turn the SHAP table into plain-language one-liners, with each
    feature's direction measured from the data rather than assumed."""
    importance = shap_df.abs().mean().sort_values(ascending=False)
    lines = []
    for feat in importance.index[:top_n]:
        corr = float(np.corrcoef(sample_df[feat].astype(float), shap_df[feat])[0, 1])
        direction = "up" if corr >= 0 else "down"
        story = FEATURE_STORIES.get(feat, {}).get(
            direction,
            f"Higher {feat.replace('_', ' ')} tends to push predictions {'up' if direction == 'up' else 'down'}.",
        )
        lines.append((feat, story))
    return lines


def render_what_matters(shap_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    st.markdown("#### Which dials actually move the needle?")
    st.caption(
        "SHAP values, computed on 2,000 held-out tracks: for every prediction, each feature's exact share of "
        "the credit or blame. This is the model showing its work, not a black box with a score."
    )

    lines = shap_insights(shap_df, sample_df)
    insights_html = "".join(
        f'<div class="insight"><div class="n">{i + 1:02d}</div><div class="f">{feat.replace("_", " ")}</div><div class="s">{story}</div></div>'
        for i, (feat, story) in enumerate(lines)
    )
    st.markdown(insights_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    importance = shap_df.abs().mean().sort_values(ascending=False)

    bar = go.Figure(go.Bar(x=importance.values[::-1], y=[f.replace("_", " ") for f in importance.index[::-1]], orientation="h", marker_color=theme.BRAND_GREEN))
    bar.update_layout(title="Average influence on a prediction (mean |SHAP|, popularity points)", xaxis_title="popularity points")
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
    beeswarm.update_yaxes(tickmode="array", tickvals=list(range(len(features_sorted))), ticktext=[f.replace("_", " ") for f in reversed(features_sorted)])
    beeswarm.update_layout(
        title="Every dot is one real track",
        xaxis_title="pull on that track's predicted popularity (SHAP value)",
    )
    beeswarm.add_vline(x=0, line_width=1, line_color=theme.AXIS_LINE)
    st.plotly_chart(theme.apply_theme(beeswarm, height=520, show_legend=False), width="stretch")
    st.caption(
        "How to read it: dots to the right of the line pushed that track's prediction up, dots to the left pushed it down. "
        "Color is the feature's value on that track -- red = high, blue = low. Red on the left, as with instrumentalness, "
        "means a high value hurts."
    )


# ============================================================ genres ====
def render_genres(genre_breakdown_df: pd.DataFrame, genre_shap: dict) -> None:
    st.markdown("#### Does the recipe change by genre?")
    st.caption("Same single model, evaluated and explained separately within each genre's held-out tracks. Short answer: yes.")

    genres = genre_breakdown_df["genre"].tolist()
    colors = [theme.GENRE_COLORS.get(g, theme.MUTED_INK) for g in genres]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=genres, y=genre_breakdown_df["mean_popularity"], marker_color=colors))
        fig.update_layout(title="How popular each genre runs", yaxis_title="average popularity (0-100)")
        st.plotly_chart(theme.apply_theme(fig, height=360, show_legend=False), width="stretch")
    with col2:
        fig = go.Figure(go.Bar(x=genres, y=genre_breakdown_df["rmse"], marker_color=colors))
        fig.update_layout(title="Where the model struggles (higher = harder to predict)", yaxis_title="typical error, popularity points")
        st.plotly_chart(theme.apply_theme(fig, height=360, show_legend=False), width="stretch")

    html_block(
        """<div class="explain"><div class="q">Why is pop the hardest genre for an audio-only model?</div>
        <div class="a">Pop has the biggest prediction errors of any focus genre here -- a reasonable read is that pop
        popularity depends most on exactly what this model can't hear: star power, marketing, and playlist
        placement. Classical and acoustic, where the sound itself carries more of the signal, are far easier.</div></div>"""
    )

    ordered_features = sorted(config.AUDIO_FEATURES, key=lambda f: -np.mean([genre_shap[g].get(f, 0) for g in genres]))
    z = [[genre_shap[g].get(f, 0) for f in ordered_features] for g in genres]
    heatmap = go.Figure(
        go.Heatmap(
            z=z,
            x=[f.replace("_", " ") for f in ordered_features],
            y=genres,
            colorscale=theme.SEQUENTIAL_GREEN,
            colorbar=dict(title="influence", thickness=14),
        )
    )
    heatmap.update_layout(title="What each genre's predictions hinge on (mean |SHAP|)")
    st.plotly_chart(theme.apply_theme(heatmap, height=380, show_legend=False), width="stretch")
    st.caption(
        "Each row is a genre, each column an audio feature; brighter = that feature matters more to the model "
        "inside that genre. Danceability leads hip-hop, acousticness dominates classical."
    )

    st.markdown('<div class="grouplabel" style="margin-top:1.6rem;">The sound of each genre</div>', unsafe_allow_html=True)
    st.caption("The most popular track per genre in this dataset. Press play to hear what each genre's ceiling sounds like.")
    anthems = genre_anthems()
    cols = st.columns(3)
    for i, (genre, row) in enumerate(anthems.iterrows()):
        with cols[i % 3]:
            genre_color = theme.GENRE_COLORS.get(genre, theme.MUTED_INK)
            html_block(
                f"""<div class="match-cap"><span class="pill" style="background:{genre_color};">{genre}</span>
                <span class="mono-dim">popularity {int(row["popularity"])}</span></div>"""
            )
            spotify_player(row["track_id"], height=80)


# ============================================================ under the hood ====
def render_under_the_hood(reg_results: dict, clf_results: dict, pr_curve_df: pd.DataFrame, test_df: pd.DataFrame, models: dict, cleaning_report: dict) -> None:
    mae = reg_results["xgboost_regressor"]["mae"]
    base_mae = reg_results["baseline_mean"]["mae"]
    auc = clf_results["xgboost_classifier"]["auc"]

    st.markdown("#### The honest scoreboard")
    html_block(
        f"""
        <div class="explain"><div class="q">How wrong is it, typically?</div>
        <div class="a">Off by about <b>{mae:.0f} points</b> on the 0-100 scale, on tracks it never saw in training.
        Sounds rough -- but guessing the dataset average for every song misses by {base_mae:.1f}, and a plain linear
        model by {reg_results["linear_regression"]["mae"]:.1f}. The audio carries real signal; it just doesn't carry all of it.</div></div>

        <div class="explain"><div class="q">Can it spot a hit?</div>
        <div class="a">Asked "will this land in the top 20%?", it scores an AUC of <b>{auc:.2f}</b> --
        where 0.5 is a coin flip and 1.0 is clairvoyance. Useful, far from magic.</div></div>

        <div class="explain"><div class="q">What can't it hear?</div>
        <div class="a">Artist fame, marketing budgets, and playlist placement aren't in the data -- and they
        plausibly decide more of a song's fate than its waveform does. This project measures how far the sound
        alone gets you, and is upfront that the answer is "part of the way."</div></div>
        """
    )

    st.markdown("#### Predicting the exact score (0-100)")
    reg_models = [
        ("baseline", "Guess the average", reg_results["baseline_mean"]),
        ("linear_regression", "Linear regression", reg_results["linear_regression"]),
        ("xgboost_regressor", "XGBoost", reg_results["xgboost_regressor"]),
    ]
    fig = go.Figure()
    for key, label, metrics in reg_models:
        fig.add_trace(go.Bar(name=label, x=["RMSE", "MAE"], y=[metrics["rmse"], metrics["mae"]], marker_color=theme.MODEL_COLORS[key]))
    fig.update_layout(barmode="group", title="Error on 17,948 unseen tracks (lower is better)", yaxis_title="popularity points")
    st.plotly_chart(theme.apply_theme(fig, height=360), width="stretch")

    st.markdown("#### Calling the top 20%")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        clf_models = [
            ("logistic_regression", "Logistic regression", clf_results["logistic_regression"]),
            ("xgboost_classifier", "XGBoost", clf_results["xgboost_classifier"]),
        ]
        fig = go.Figure()
        for key, label, metrics in clf_models:
            fig.add_trace(go.Bar(name=label, x=["AUC", "Avg. precision"], y=[metrics["auc"], metrics["average_precision"]], marker_color=theme.MODEL_COLORS[key]))
        fig.update_layout(barmode="group", title="Held-out test set (higher is better)", yaxis=dict(range=[0, 1]))
        st.plotly_chart(theme.apply_theme(fig, height=360), width="stretch")
    with col2:
        fig = go.Figure()
        for model_key, label in [("logistic_regression", "Logistic regression"), ("xgboost_classifier", "XGBoost")]:
            curve = pr_curve_df[pr_curve_df["model"] == model_key].sort_values("recall")
            fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines", name=label, line=dict(color=theme.MODEL_COLORS[model_key], width=2.5)))
        fig.update_layout(title="Precision-recall tradeoff", xaxis_title="recall (hits found)", yaxis_title="precision (calls that were right)", yaxis=dict(range=[0, 1]))
        st.plotly_chart(theme.apply_theme(fig, height=360), width="stretch")

    st.markdown("#### Every prediction, laid bare")
    sample = test_df.sample(n=min(1500, len(test_df)), random_state=config.RANDOM_SEED)
    preds = models["xgb_reg"].predict(sample[config.AUDIO_FEATURES])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample[config.TARGET_COLUMN], y=preds, mode="markers", name="Unseen tracks", marker=dict(color=theme.BRAND_GREEN, size=5, opacity=0.35)))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect prediction", line=dict(color=theme.MUTED_INK, dash="dash", width=1.5)))
    fig.update_layout(xaxis_title="what Spotify actually scored it", yaxis_title="what the model guessed")
    st.plotly_chart(theme.apply_theme(fig, height=420), width="stretch")
    st.caption(
        "The model hedges: it rarely dares to predict above ~60, because the audio alone almost never justifies "
        "certainty that a track is a smash. The vertical stripe at zero is a quirk of the dataset -- thousands of "
        "tracks with a popularity of exactly 0."
    )

    with st.expander("Data notes & cleaning"):
        st.markdown(
            f"""
- Source: [Kaggle — maharshipandya/spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (ODbL), a static snapshot.
- Raw file: **{cleaning_report["raw_rows"]:,}** rows. Dropped {cleaning_report["dropped_invalid_duration_or_missing_meta"]} broken row(s) and
  **{cleaning_report["dropped_duplicate_track_id"]:,} duplicates** (the same track re-listed under multiple genres — left in, those
  would leak between train and test and quietly inflate every score). **{cleaning_report["final_rows"]:,}** unique tracks modeled.
- 80/20 train/test split, stratified by genre. The "hit" cutoff (top 20% ⇒ popularity ≥ {clf_results["hit_threshold_popularity"]:.0f}) was computed
  from training data only.
- Popularity is Spotify's own 0-100 score, weighted toward recent plays, as of when the dataset was collected — not live.
- Genre labels are Spotify's categorization, not an objective truth.
            """
        )


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

    st.markdown('<div class="mono mono-green">Sound vs. fame — an honest model</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="headline">What makes a <em>hit?</em>{equalizer_html()}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lede">A model that predicts a song\'s Spotify popularity from its sound alone — '
        "no artist names, no marketing, no playlists. Shape a track, press play on its nearest "
        "real-world cousins, and see exactly what the model is thinking.</div>",
        unsafe_allow_html=True,
    )

    mae = reg_results["xgboost_regressor"]["mae"]
    auc = clf_results["xgboost_classifier"]["auc"]
    html_block(
        f"""
        <div class="statline">
            <div><div class="k">Tracks studied</div><div class="v">{cleaning_report["final_rows"]:,}</div></div>
            <div><div class="k">Genres</div><div class="v">{tracks_df[config.GENRE_COLUMN].nunique()}</div></div>
            <div><div class="k">Typical miss</div><div class="v">±{mae:.0f}<small>of 100 points</small></div></div>
            <div><div class="k">Hit detection</div><div class="v">{auc:.2f}<small>AUC · 0.5 = coin flip</small></div></div>
        </div>
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(["01 · Studio", "02 · What matters", "03 · Genres", "04 · Under the hood"])
    with tab1:
        render_studio(tracks_df, models)
    with tab2:
        render_what_matters(shap_df, sample_df)
    with tab3:
        render_genres(genre_breakdown_df, genre_shap)
    with tab4:
        render_under_the_hood(reg_results, clf_results, pr_curve_df, test_df, models, cleaning_report)

    html_block(
        """
        <div class="footer-line">
            Data: Kaggle · maharshipandya / spotify-tracks-dataset (ODbL) &nbsp;·&nbsp;
            Popularity is Spotify's own 0–100 score &nbsp;·&nbsp;
            <a href="https://github.com/nikitarao19/hit-prediction">Code on GitHub</a>
        </div>
        """
    )


if __name__ == "__main__":
    main()
