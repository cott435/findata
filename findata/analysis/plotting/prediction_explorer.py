"""Interactive forecast-vs-realized browser for quantile predictions.

Input contract — one tidy long frame (what forecasting's scripts/test.py
writes to ``<run_dir>/test/predictions.parquet``):

    ticker    str
    date      decision date (forecast origin; predictions cover t+1 .. t+H)
    h         horizon in trading days (1..H)
    pred_qXX  predicted CUMULATIVE log return over h, one column per quantile
              (e.g. pred_q05 .. pred_q95) — always stored in raw return space
    realized  realized cumulative log return over the same window
    sigma     trailing vol at the decision date (for the vol-norm display)

Display transforms (all derived on the fly from the raw columns):
    scale   raw = cumulative log return | h-norm = / sqrt(h) | vol-norm = / (sigma*sqrt(h))
    view    cumulative, or per-step (the h-1 -> h increment). Differencing
            quantile curves is exact for the median but only approximate for
            outer quantiles (quantiles don't subtract) — treat per-step bands
            as a visual guide.
"""
from __future__ import annotations

import re

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from holoviews import opts

from findata.analysis.plotting.base import (ExplorerBase, SessionSync,
                                            minimap_opts)
from findata.analysis.plotting.style import LAYER_PALETTE

SCALES = ("raw", "h-norm", "vol-norm")
VIEWS = ("cumulative", "per-step")


def quantile_col(q: float) -> str:
    return f"pred_q{int(round(q * 100)):02d}"


def pred_quantiles(preds: pd.DataFrame) -> list[float]:
    """Quantile levels inferred from the pred_qXX columns."""
    qs = [int(m.group(1)) / 100 for c in preds.columns
          for m in [re.fullmatch(r"pred_q(\d{2})", c)] if m]
    return sorted(qs)


def band_pairs(quantiles) -> dict[str, tuple[float, float]]:
    """Symmetric (lo, hi) pairs keyed by coverage label, widest first
    (e.g. {'90%': (0.05, 0.95), '80%': (0.10, 0.90), '50%': (0.25, 0.75)})."""
    qs = set(round(q, 4) for q in quantiles)
    pairs = [(q, round(1 - q, 4)) for q in sorted(qs) if q < 0.5 and round(1 - q, 4) in qs]
    return {f"{round((hi - lo) * 100)}%": (lo, hi)
            for lo, hi in sorted(pairs, key=lambda p: p[0])}


def to_per_step(frame: pd.DataFrame, value_cols) -> pd.DataFrame:
    """Cumulative-over-h columns -> per-step increments within each
    (ticker, date) forecast; the h=1 row keeps its (already 1-step) value."""
    out = frame.sort_values(["ticker", "date", "h"]).copy()
    g = out.groupby(["ticker", "date"])[list(value_cols)]
    out[list(value_cols)] = g.diff().fillna(out[list(value_cols)])
    return out


def apply_scale(frame: pd.DataFrame, value_cols, scale: str) -> pd.DataFrame:
    """Divide the raw-return-space columns by the chosen scale denominator."""
    if scale == "raw":
        return frame
    out = frame.copy()
    root_h = np.sqrt(out["h"].to_numpy(dtype=float))
    denom = root_h if scale == "h-norm" else out["sigma"].to_numpy(dtype=float) * root_h
    denom = np.where(denom > 0, denom, np.nan)
    out[list(value_cols)] = out[list(value_cols)].div(denom, axis=0)
    return out


class PredictionExplorer(ExplorerBase):
    """Scroll through forecast origins per ticker; compare predicted quantile
    bands against the realized return under switchable scale/view.

    preds      the tidy frame described in the module docstring
    quantiles  optional explicit levels (default: inferred from columns)
    """

    def __init__(self, preds: pd.DataFrame, quantiles=None, *, width: int = 1250,
                 height: int = 430, error_height: int = 170,
                 minimap_height: int = 90, selector_width: int = 300):
        preds = preds.copy()
        preds["ticker"] = preds["ticker"].astype(str)
        preds["date"] = pd.to_datetime(preds["date"])
        self._preds = preds.sort_values(["ticker", "date", "h"]).reset_index(drop=True)
        self._tickers = sorted(preds["ticker"].unique())
        self._horizons = sorted(int(h) for h in preds["h"].unique())
        self._quantiles = list(quantiles) if quantiles is not None \
            else pred_quantiles(preds)
        if 0.5 not in [round(q, 4) for q in self._quantiles]:
            raise ValueError("predictions need a median column (pred_q50)")
        self._bands = band_pairs(self._quantiles)
        self._value_cols = [quantile_col(q) for q in self._quantiles] + ["realized"]
        self._width, self._height = width, height
        self._error_height, self._minimap_height = error_height, minimap_height
        self._selector_width = selector_width
        self._frames: dict = {}

    # -- data ------------------------------------------------------------------
    def _ticker_frame(self, ticker: str, view: str) -> pd.DataFrame:
        key = (ticker, view)
        if key not in self._frames:
            df = self._preds[self._preds["ticker"] == ticker]
            if view == "per-step":
                df = to_per_step(df, self._value_cols)
            self._frames[key] = df
        return self._frames[key]

    def _prepare(self, ticker: str, h: int, scale: str, view: str) -> pd.DataFrame:
        df = self._ticker_frame(ticker, view)
        sub = df[df["h"] == h]
        return apply_scale(sub, self._value_cols, scale)

    # -- overlays ----------------------------------------------------------------
    def _main_overlay(self, ticker, h, scale, view, bands) -> hv.Overlay:
        sub = self._prepare(ticker, int(h), scale, view)
        d = sub["date"].to_numpy()
        vdim = hv.Dimension("value_pred", label="prediction")
        vdim_hi = hv.Dimension("value_pred_hi", label="prediction_hi")
        elems = []
        for i, label in enumerate(sorted(self._bands, key=lambda L: -int(L[:-1]))):
            if label not in bands:
                continue
            lo_q, hi_q = self._bands[label]
            elems.append(
                hv.Area((d, sub[quantile_col(lo_q)].to_numpy(dtype=float),
                         sub[quantile_col(hi_q)].to_numpy(dtype=float)),
                        kdims=["date"], vdims=[vdim, vdim_hi], label=label)
                .opts(alpha=0.14 + 0.05 * i, color=LAYER_PALETTE[0], line_alpha=0))
        elems.append(hv.Curve((d, sub[quantile_col(0.5)].to_numpy(dtype=float)),
                              "date", vdim, label="median")
                     .opts(color=LAYER_PALETTE[0], line_width=2.0))
        elems.append(hv.Curve((d, sub["realized"].to_numpy(dtype=float)),
                              "date", vdim, label="realized")
                     .opts(color="black", line_width=1.5))
        title = f"{ticker} — predicted vs realized at h={h} ({view}, {scale})"
        return hv.Overlay(elems).opts(title=title)

    def _error_overlay(self, ticker, h, scale, view) -> hv.Overlay:
        sub = self._prepare(ticker, int(h), scale, view)
        vdim = hv.Dimension("value_err", label="median - realized")
        err = sub[quantile_col(0.5)].to_numpy(dtype=float) \
            - sub["realized"].to_numpy(dtype=float)
        return hv.Overlay([
            hv.Curve((sub["date"].to_numpy(), err), "date", vdim, label="error")
            .opts(color=LAYER_PALETTE[1], line_width=1.2),
            hv.HLine(0).opts(color="black", line_width=0.7, alpha=0.5),
        ]).opts(title="median error")

    def _minimap_curve(self, ticker, h) -> hv.Curve:
        sub = self._ticker_frame(ticker, "cumulative")
        sub = sub[sub["h"] == int(h)]
        return hv.Curve((sub["date"].to_numpy(),
                         sub["realized"].to_numpy(dtype=float)), "date", "value")

    # -- per-session app -----------------------------------------------------------
    def _make_app(self) -> pn.Row:
        sync = SessionSync()
        w = self._selector_width
        ticker_w = pn.widgets.Select(name="ticker", options=self._tickers, width=w)
        h_w = pn.widgets.Select(name="horizon (days)", options=self._horizons,
                                value=self._horizons[-1], width=w)
        scale_w = pn.widgets.RadioButtonGroup(options=list(SCALES), value="raw",
                                              width=w)
        view_w = pn.widgets.RadioButtonGroup(options=list(VIEWS), value="cumulative",
                                             width=w)
        band_labels = sorted(self._bands, key=lambda L: -int(L[:-1]))
        bands_w = pn.widgets.CheckButtonGroup(options=band_labels, value=band_labels,
                                              button_type="primary", width=w)

        curve_opts = opts.Curve(tools=["hover"], muted_alpha=0.1)
        overlay_opts = dict(width=self._width, autorange="y", show_grid=True,
                            legend_position="right",
                            legend_opts={"click_policy": "mute"},
                            active_tools=["pan", "wheel_zoom"],
                            hooks=[sync.capture_x_range])

        main = hv.DynamicMap(pn.bind(self._main_overlay, ticker=ticker_w, h=h_w,
                                     scale=scale_w, view=view_w, bands=bands_w))
        error = hv.DynamicMap(pn.bind(self._error_overlay, ticker=ticker_w, h=h_w,
                                      scale=scale_w, view=view_w))
        minimap = hv.DynamicMap(pn.bind(self._minimap_curve, ticker=ticker_w,
                                        h=h_w)).opts(
            minimap_opts(width=self._width, height=self._minimap_height,
                         sync=sync, framewise=True))

        layout = hv.Layout([
            main.opts(curve_opts, opts.Overlay(height=self._height, **overlay_opts)),
            error.opts(curve_opts,
                       opts.Overlay(height=self._error_height, **overlay_opts)),
            minimap,
        ]).cols(1)

        controls = pn.Column(
            pn.pane.Markdown(
                "### Forecast explorer\n"
                "scale: raw = cum log return · h-norm = /sqrt(h) · "
                "vol-norm = /(sigma*sqrt(h))\n\n"
                "per-step = h-1 to h increment (approximate for outer quantiles)\n\n"
                "legend click mutes; drag the minimap to scrub time"),
            ticker_w, h_w,
            pn.pane.Markdown("**scale**"), scale_w,
            pn.pane.Markdown("**view**"), view_w,
            pn.pane.Markdown("**quantile bands**"), bands_w,
        )
        return pn.Row(controls, layout)
