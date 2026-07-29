"""Element builders for the workbench tabs (pure functions -> hv elements).

Callers supply unique y-dimension names (bokeh links axes whose dimension
labels match; only the shared 'date' dimension should link).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn

from findata.analysis.plotting.style import LAYER_PALETTE

UP_COLOR = '#2a9d8f'
DOWN_COLOR = '#d64545'

CANDLE_MAX_BARS = 4000


def candlestick(frame: pd.DataFrame, vdim: hv.Dimension) -> hv.Overlay:
    """OHLC candles (wick Segments + body Rectangles); degrades to a close
    line above CANDLE_MAX_BARS for render weight."""
    df = frame[['open', 'high', 'low', 'close']].dropna()
    if df.empty or len(df) > CANDLE_MAX_BARS:
        series = frame['close'].dropna()
        curve = hv.Curve((series.index, series.to_numpy(float)), 'date', vdim,
                         label='close').opts(color=LAYER_PALETTE[0])
        return hv.Overlay([curve])
    idx = pd.to_datetime(df.index)
    spacing = np.median(np.diff(idx.values)).astype('timedelta64[ns]')
    half = pd.Timedelta(spacing * 0.3)
    up = (df['close'] >= df['open']).to_numpy()
    colors = np.where(up, UP_COLOR, DOWN_COLOR)

    wicks = hv.Segments(
        {'date': idx, 'y0': df['low'].to_numpy(float),
         'date2': idx, 'y1': df['high'].to_numpy(float)},
        kdims=['date', 'y0', 'date2', 'y1']).opts(color='#666666', line_width=1)
    bodies = hv.Rectangles(
        {'x0': idx - half, 'y0': np.minimum(df['open'], df['close']).to_numpy(float),
         'x1': idx + half, 'y1': np.maximum(df['open'], df['close']).to_numpy(float),
         'dir': colors},
        kdims=['x0', 'y0', 'x1', 'y1'], vdims=['dir']).opts(
        color='dir', line_color='dir')
    return hv.Overlay([wicks, bodies])


def series_overlay(series_map: dict, vdim: hv.Dimension, palette=LAYER_PALETTE,
                   dashes=None) -> hv.NdOverlay:
    """Named curves on a common date axis (stable type for DynamicMaps)."""
    curves = {}
    for i, (name, series) in enumerate(series_map.items()):
        s = series.dropna()
        curve = hv.Curve((s.index, s.to_numpy(float)), 'date', vdim).opts(
            color=palette[i % len(palette)])
        if dashes and name in dashes:
            curve = curve.opts(line_dash=dashes[name])
        curves[name] = curve
    if not curves:
        curves['(none)'] = hv.Curve([], 'date', vdim)
    return hv.NdOverlay(curves, kdims='series')


def mp_spectrum_elements(spec: dict, vdim_name: str) -> hv.Overlay:
    """Eigenvalue density vs the Marchenko-Pastur law, lambda_+ marked."""
    eigvals = np.asarray(spec['eigvals'])
    q, sigma2, lam_plus = spec['q'], spec['sigma2'], spec['lam_plus']
    lam_minus = sigma2 * (1 - 1 / np.sqrt(q)) ** 2

    bulk = eigvals[eigvals <= lam_plus * 1.5]
    freq, edges = np.histogram(bulk, bins=60, density=True)
    hist = hv.Histogram((edges, freq), kdims='eigenvalue',
                        vdims=vdim_name).opts(fill_alpha=0.5, color=LAYER_PALETTE[0])

    grid = np.linspace(max(lam_minus, 1e-9), lam_plus, 200)
    density = q / (2 * np.pi * sigma2 * grid) * np.sqrt(
        np.clip((lam_plus - grid) * (grid - lam_minus), 0, None))
    mp = hv.Curve((grid, density), 'eigenvalue', vdim_name).opts(
        color=LAYER_PALETTE[1], line_width=2)
    marker = hv.VLine(lam_plus).opts(color=LAYER_PALETTE[1], line_dash='dashed')
    return (hist * mp * marker).opts(
        hv.opts.Histogram(title=f'eigenvalues vs MP (n_signal={spec["n_signal"]}, '
                                f'top share={spec["top_eig_share"]:.1%})'))


def heatmap(df: pd.DataFrame, kdims: list, vdim: str, title: str = '',
            fmt: str = '%.3f', width: int = 600, height: int = 320) -> hv.HeatMap:
    """Wide frame -> labeled heatmap (rows x columns)."""
    long = df.reset_index().melt(id_vars=df.index.name or 'index')
    long.columns = [kdims[0], kdims[1], vdim]
    hm = hv.HeatMap(long, kdims=kdims, vdims=vdim)
    labels = hv.Labels(long.dropna(), kdims=kdims, vdims=vdim).opts(
        text_font_size='8pt', text_color='black')
    return (hm.opts(width=width, height=height, colorbar=True, cmap='RdBu_r',
                    symmetric=True, tools=['hover'], title=title,
                    xrotation=45) * labels)


def cumulative_ic(daily_ic: pd.Series, vdim: hv.Dimension) -> hv.Overlay:
    cum = daily_ic.dropna().cumsum()
    idx = pd.to_datetime(cum.index)
    curve = hv.Curve((idx, cum.to_numpy(float)), 'date', vdim).opts(
        color=LAYER_PALETTE[0], line_width=2)
    daily = hv.Scatter((pd.to_datetime(daily_ic.dropna().index),
                        daily_ic.dropna().cumsum().to_numpy(float)), 'date', vdim).opts(
        color=LAYER_PALETTE[0], alpha=0.15, size=2)
    zero = hv.HLine(0).opts(color='#888888', line_dash='dotted')
    return curve * daily * zero


def table(df: pd.DataFrame, width: int = 640, height: int = 320,
          fmt: dict = None):
    """Sortable table; Tabulator when available, DataFrame pane otherwise."""
    try:
        return pn.widgets.Tabulator(df, disabled=True, width=width, height=height,
                                    formatters=fmt or {})
    except Exception:
        return pn.pane.DataFrame(df, width=width, height=height)
