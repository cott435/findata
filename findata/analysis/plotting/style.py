"""Shared visual encoding for multi-layer / multi-ticker overlays.

Used by both the interactive explorers and the static (matplotlib) savers:

  COLOR      = layer (each transform / series family gets its own hue)
  SHADE+DASH = ticker (dark -> light across the sample; the DARKEST shades get
               the most spaced-out dashing so overlapping dark lines stay
               tellable apart)
"""
from __future__ import annotations

LAYER_PALETTE = ("#4c72b0", "#d64545", "#2a9d8f", "#e39c37", "#7b52a3",
                 "#5c8a4e", "#b05c88", "#556b7a", "#8a6642", "#3f7f93")
_DASH_PATTERNS = ([14, 8], [9, 6], [6, 4], [3, 3], "solid")   # spaced -> solid


def _blend(hex_color: str, other: str, frac: float) -> str:
    a, b = int(hex_color[1:], 16), int(other[1:], 16)
    mixed = [round((a >> s & 255) + ((b >> s & 255) - (a >> s & 255)) * frac)
             for s in (16, 8, 0)]
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def series_style(layer_idx: int, ticker_idx: int, n_tickers: int) -> tuple[str, object]:
    """(color, line_dash) for one (layer, ticker) pair."""
    base = LAYER_PALETTE[layer_idx % len(LAYER_PALETTE)]
    t = ticker_idx / max(n_tickers - 1, 1)          # 0 = darkest .. 1 = lightest
    color = _blend(base, "#000000", 0.35 * (1 - t)) if t < 0.5 else \
        _blend(base, "#ffffff", 0.55 * (2 * t - 1))
    dash = _DASH_PATTERNS[min(int(t * len(_DASH_PATTERNS)), len(_DASH_PATTERNS) - 1)]
    return color, dash


def _mpl_dash(dash) -> object:
    return "-" if dash == "solid" else (0, tuple(dash))
