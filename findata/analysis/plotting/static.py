"""Static (matplotlib) savers.

Layer/mode savers share the interactive explorers' visual encoding
(color = layer, shade/dash = ticker; see style.py). The prediction savers
consume the same tidy frame as PredictionExplorer (see prediction_explorer.py
for the schema) so scripts can emit both static PNGs and the interactive app
from one parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from findata.analysis.plotting.prediction_explorer import (apply_scale, band_pairs,
                                                           pred_quantiles,
                                                           quantile_col)
from findata.analysis.plotting.style import (LAYER_PALETTE, _blend, _mpl_dash,
                                             series_style)


def _to_dt_index(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame
    return frame.set_axis(pd.to_datetime(frame.index))


def _agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# transform-layer / mode savers (wide date x ticker frames)
# ---------------------------------------------------------------------------

def save_layer_plot(layers: dict, tickers, path, *, cumulative: bool = True,
                    title: str | None = None, figsize=(15, 7.5)):
    """Overlay of (transform layer x ticker) series with the shared encoding.
    Legend shows one hue swatch per layer + one gray shade/dash swatch per ticker."""
    plt = _agg()
    from matplotlib.lines import Line2D

    tickers = list(tickers)
    fig, ax = plt.subplots(figsize=figsize)
    for li, (layer, frame) in enumerate(layers.items()):
        frame = _to_dt_index(frame)
        for tj, tk in enumerate(tickers):
            if tk not in frame.columns:
                continue
            color, dash = series_style(li, tj, len(tickers))
            s = frame[tk].dropna()
            s = s.cumsum() if cumulative else s
            ax.plot(s.index, s.to_numpy(dtype=float), color=color,
                    ls=_mpl_dash(dash), lw=1.2)
    handles = [Line2D([0], [0], color=LAYER_PALETTE[li % len(LAYER_PALETTE)],
                      lw=2.6, label=layer) for li, layer in enumerate(layers)]
    handles += [Line2D([0], [0], color=series_style(0, tj, len(tickers))[0] if len(layers) == 1
                       else _blend("#444444", "#ffffff", 0.55 * tj / max(len(tickers) - 1, 1)),
                       ls=_mpl_dash(series_style(0, tj, len(tickers))[1]),
                       lw=1.4, label=tk) for tj, tk in enumerate(tickers)]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper left")
    ax.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax.set_title(title or "transform layers (color = transform, shade/dash = ticker)")
    ax.set_ylabel("cumulative value" if cumulative else "value")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved: {path}")


def save_modes_plot(modes: pd.DataFrame, sector: str, residual: pd.DataFrame,
                    member_tickers, path, *, cumulative: bool = True, figsize=(15, 7)):
    """One sector's reward anatomy: global mode vs the sector's own mode vs the
    two compounded, over the residual paths of that sector's sampled tickers."""
    plt = _agg()

    modes = _to_dt_index(modes)
    residual = _to_dt_index(residual)
    prep = (lambda s: s.dropna().cumsum()) if cumulative else (lambda s: s.dropna())

    fig, ax = plt.subplots(figsize=figsize)
    for tj, tk in enumerate(member_tickers):
        if tk in residual.columns:
            s = prep(residual[tk])
            ax.plot(s.index, s.to_numpy(dtype=float), color="#b8b8b8",
                    lw=0.9, ls=_mpl_dash(series_style(0, tj, len(member_tickers))[1]),
                    label="residuals" if tj == 0 else None)
    g = prep(modes["global"])
    ax.plot(g.index, g.to_numpy(dtype=float), color="black", lw=2.2, label="global mode")
    key = f"sector_{sector}"
    if key in modes.columns:
        s = prep(modes[key])
        ax.plot(s.index, s.to_numpy(dtype=float), color="#d64545", lw=2.0,
                label=f"{sector} mode")
        comp = prep(modes["global"] + modes[key])
        ax.plot(comp.index, comp.to_numpy(dtype=float), color="#7b52a3", lw=2.0,
                ls=(0, (8, 4)), label=f"global + {sector} mode")
    ax.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_title(f"{sector}: global vs sector mode vs compound "
                 f"({'cumulative' if cumulative else 'daily'}), member residuals in gray")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved: {path}")


# ---------------------------------------------------------------------------
# prediction savers (tidy predictions frame; see prediction_explorer schema)
# ---------------------------------------------------------------------------

def save_fan_chart(preds: pd.DataFrame, ticker: str, path, *, date=None,
                   figsize=(7, 4)):
    """Quantile fan across horizons from one forecast origin (default: the
    ticker's last decision date) vs the realized path."""
    plt = _agg()
    sub = preds[preds["ticker"] == ticker]
    if sub.empty:
        return
    date = sub["date"].max() if date is None else date
    sub = sub[sub["date"] == date].sort_values("h")
    qs = pred_quantiles(sub)
    x = sub["h"].to_numpy()
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, (lo, hi)) in enumerate(band_pairs(qs).items()):
        ax.fill_between(x, sub[quantile_col(lo)], sub[quantile_col(hi)],
                        alpha=0.16 + 0.08 * i, color=LAYER_PALETTE[0],
                        label=f"{label} interval")
    ax.plot(x, sub[quantile_col(0.5)], lw=2, color=LAYER_PALETTE[0], label="median")
    ax.plot(x, sub["realized"], "o-", color="black", label="realized")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"{ticker} — forecast from {pd.Timestamp(date).date()}")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Cumulative log return")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_prediction_bands(preds: pd.DataFrame, ticker: str, h: int, path, *,
                          scale: str = "raw", figsize=(15, 6)):
    """Time series of the h-day forecast bands vs realized for one ticker."""
    plt = _agg()
    sub = preds[(preds["ticker"] == ticker) & (preds["h"] == h)].sort_values("date")
    if sub.empty:
        return
    qs = pred_quantiles(sub)
    value_cols = [quantile_col(q) for q in qs] + ["realized"]
    sub = apply_scale(sub, value_cols, scale)
    d = pd.to_datetime(sub["date"])
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, (lo, hi)) in enumerate(band_pairs(qs).items()):
        ax.fill_between(d, sub[quantile_col(lo)], sub[quantile_col(hi)],
                        alpha=0.15 + 0.07 * i, color=LAYER_PALETTE[0],
                        label=f"{label} interval", lw=0)
    ax.plot(d, sub[quantile_col(0.5)], lw=1.8, color=LAYER_PALETTE[0], label="median")
    ax.plot(d, sub["realized"], lw=1.2, color="black", label="realized")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"{ticker} — h={h} forecast vs realized ({scale})")
    ax.set_ylabel({"raw": "cum log return", "h-norm": "cum log return / sqrt(h)",
                   "vol-norm": "cum log return / (sigma*sqrt(h))"}[scale])
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_calibration(preds: pd.DataFrame, path, *, figsize=(8, 5)):
    """Empirical P(realized <= predicted q) per quantile, one line per horizon.
    A calibrated model tracks the diagonal."""
    plt = _agg()
    qs = pred_quantiles(preds)
    fig, ax = plt.subplots(figsize=figsize)
    for hi, h in enumerate(sorted(preds["h"].unique())):
        sub = preds[preds["h"] == h]
        ok = sub["realized"].notna()
        emp = [float((sub.loc[ok, "realized"] <= sub.loc[ok, quantile_col(q)]).mean())
               for q in qs]
        ax.plot(qs, emp, "o-", lw=1.4, label=f"h={h}",
                color=LAYER_PALETTE[hi % len(LAYER_PALETTE)])
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="perfect")
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Quantile calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_pred_scatter(preds: pd.DataFrame, h: int, path, *, figsize=(6, 6)):
    """Median prediction vs realized at one horizon, with the Spearman rank
    correlation (a pooled analogue of the daily rank IC)."""
    plt = _agg()
    sub = preds[preds["h"] == h].dropna(subset=["realized"])
    if sub.empty:
        return
    x = sub[quantile_col(0.5)]
    y = sub["realized"]
    rho = float(x.corr(y, method="spearman"))
    lim = float(np.nanmax(np.abs(np.concatenate([x.to_numpy(), y.to_numpy()])))) * 1.05
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=8, alpha=0.25, color=LAYER_PALETTE[0])
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, ls="--")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("predicted median (cum log return)")
    ax.set_ylabel("realized")
    ax.set_title(f"h={h}: median vs realized (spearman={rho:.3f}, n={len(sub)})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
