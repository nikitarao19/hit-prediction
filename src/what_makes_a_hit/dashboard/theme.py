"""Color tokens and Plotly styling shared across the dashboard.

The palette below was run through the dataviz skill's CVD/contrast
validator against this app's dark surface (`node validate_palette.js
"#16a34a,#3987e5,#d55181,#c98500,#9085e9,#d95926" --mode dark --surface
"#1a1a19"` -> all checks pass) before being wired in here, rather than
picked by eye.
"""

from __future__ import annotations

import plotly.graph_objects as go

# --- Surfaces / ink (dark theme, deliberately single-mode -- this app does
# not offer a light mode, matching a music-player aesthetic) --------------
PAGE_BG = "#0d0d0d"
CARD_BG = "#1a1a19"
PRIMARY_INK = "#ffffff"
SECONDARY_INK = "#c3c2b7"
MUTED_INK = "#898781"
GRIDLINE = "#2c2c2a"
AXIS_LINE = "#383835"

# --- Brand -----------------------------------------------------------------
BRAND_GREEN = "#1DB954"

# --- Validated categorical palette (6 slots, dark surface #1a1a19) --------
GENRE_COLORS = {
    "pop": "#16a34a",
    "hip-hop": "#3987e5",
    "classical": "#d55181",
    "edm": "#c98500",
    "acoustic": "#9085e9",
    "metal": "#d95926",
}
CATEGORICAL_ORDER = list(GENRE_COLORS.values())

# --- Status palette (fixed, dark-surface contrast-checked) ----------------
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

# --- Model-quality progression (muted -> best), reused across regression &
# classification comparisons so "which model is better" always reads the
# same way -------------------------------------------------------------
MODEL_COLORS = {
    "baseline": MUTED_INK,
    "linear_regression": "#3987e5",
    "logistic_regression": "#3987e5",
    "xgboost_regressor": BRAND_GREEN,
    "xgboost_classifier": BRAND_GREEN,
}

# --- SHAP low->high feature-value colorscale: the domain-conventional
# blue (low) -> red (high) reading, built from the two pre-validated
# dark-surface categorical endpoints rather than a fresh hue pick. ---------
SHAP_COLORSCALE = [[0.0, "#3987e5"], [1.0, "#e66767"]]

# --- Single-hue sequential ramp (brand green, low->high magnitude) for the
# genre x feature importance heatmap. Monotonically increasing lightness
# and chroma from a surface-adjacent step up to the brand accent. ---------
SEQUENTIAL_GREEN = [[0.0, "#17301f"], [0.5, "#1f7a3d"], [1.0, "#22c55e"]]

# Space Grotesk is loaded via a Google Fonts @import in the dashboard's
# injected CSS; plotly text renders in the same document, so it inherits
# the webfont with a system-ui fallback while it loads.
FONT_FAMILY = "'Space Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO_FAMILY = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace"


def apply_theme(fig: go.Figure, height: int | None = None, show_legend: bool | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=SECONDARY_INK, size=13),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SECONDARY_INK)),
        margin=dict(l=10, r=10, t=48, b=10),
    )
    # Setting title_font unconditionally makes Plotly.js render the literal
    # string "undefined" on figures with no title (e.g. the gauge) -- only
    # style it when a title is actually present.
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(family=FONT_FAMILY, color=PRIMARY_INK, size=16))
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=AXIS_LINE, linecolor=AXIS_LINE, color=MUTED_INK)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=AXIS_LINE, linecolor=AXIS_LINE, color=MUTED_INK)
    if height:
        fig.update_layout(height=height)
    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)
    return fig
