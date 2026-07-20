"""Transform / mode-decomposition explorer.

Interaction model borrows from Tableau dashboards: the legend is a filter
(click to mute), quick-filter selectors live in a left rail, a 'focus'
switcher steps through transforms one at a time (highlight action), and a
minimap drives the shared zoom across linked panels. Visual encoding
(color = layer, shade/dash = ticker) is shared with the static savers — see
:mod:`findata.analysis.plotting.style`.
"""
from __future__ import annotations

import holoviews as hv
import pandas as pd
import panel as pn
from holoviews import opts

from findata.analysis.plotting.base import (ExplorerBase, SessionSync,
                                            _to_dt_index, minimap_opts)
from findata.analysis.plotting.style import LAYER_PALETTE, series_style


class TransformExplorer(ExplorerBase):
    """Interactive browser for how transforms alter per-ticker reward paths.

    layers   {layer_name: wide date x ticker frame} — transforms and/or mode
             components (e.g. market_structure.transform_layers() merged with
             decompose_modes()['panels'])
    modes    optional date x mode frame (decompose_modes()['modes']): the
             global market signal and per-sector modes, viewable separately or
             compounded (sector shown as global+sector)
    tickers  display sample (default: the first layer's columns)

    Widgets: ticker CrossSelector, layer toggle buttons, a 'focus' radio that
    steps through one transform at a time, cumulative/daily view, and — when
    modes are given — a mode selector plus a compound-with-global checkbox.
    Legend clicks mute individual curves; the minimap zooms all panels.
    """

    def __init__(self, layers: dict, modes: pd.DataFrame | None = None,
                 tickers=None, *, cumulative: bool = True, width: int = 1250,
                 height: int = 430, modes_height: int = 260,
                 minimap_height: int = 90, selector_width: int = 380):
        self._layers = {name: _to_dt_index(f) for name, f in layers.items()}
        self._layer_names = list(self._layers)
        first = next(iter(self._layers.values()))
        self._tickers = list(tickers) if tickers is not None else list(first.columns)
        self._modes = _to_dt_index(modes) if modes is not None else None
        self._mode_names = list(self._modes.columns) if self._modes is not None else []
        self._cumulative = cumulative
        self._width, self._height = width, height
        self._modes_height, self._minimap_height = modes_height, minimap_height
        self._selector_width = selector_width

    # -- overlays ---------------------------------------------------------
    @staticmethod
    def _prep(s: pd.Series, cumulative: bool) -> pd.Series:
        s = s.dropna()
        return s.cumsum() if cumulative else s

    def _main_overlay(self, layers_sel, tickers_sel, view, focus) -> hv.Overlay:
        cumulative = view == "cumulative"
        show = [focus] if focus != "All" else list(layers_sel)
        vdim = hv.Dimension("value_main", label="series")
        curves = []
        for li, layer in enumerate(self._layer_names):   # stable hue per layer
            if layer not in show:
                continue
            frame = self._layers[layer]
            for tj, tk in enumerate(self._tickers):      # stable shade per ticker
                if tk not in tickers_sel or tk not in frame.columns:
                    continue
                color, dash = series_style(li, tj, len(self._tickers))
                s = self._prep(frame[tk], cumulative)
                curves.append(
                    hv.Curve((s.index, s.to_numpy(dtype=float)), "date", vdim,
                             label=f"{layer} · {tk}")
                    .opts(color=color, line_dash=dash,
                          line_width=1.9 if dash != "solid" else 1.4))
        if not curves:
            curves = [hv.Curve([], "date", vdim, label="(none)")]
        title = f"ticker series by transform ({view})"
        return hv.Overlay(curves).opts(title=title)

    def _modes_overlay(self, modes_sel, view, compound) -> hv.Overlay:
        cumulative = view == "cumulative"
        vdim = hv.Dimension("value_modes", label="mode")
        g = self._modes["global"] if "global" in self._modes else None
        curves = []
        for mi, m in enumerate(self._mode_names):
            if m not in modes_sel:
                continue
            s, label = self._modes[m], m
            if compound and m.startswith("sector_") and g is not None:
                s, label = s + g, f"global+{m}"
            s = self._prep(s, cumulative)
            curves.append(hv.Curve((s.index, s.to_numpy(dtype=float)), "date", vdim,
                                   label=label)
                          .opts(color=LAYER_PALETTE[mi % len(LAYER_PALETTE)],
                                line_width=2.0))
        if not curves:
            curves = [hv.Curve([], "date", vdim, label="(none)")]
        return hv.Overlay(curves).opts(
            title="mode series (global / sector)"
                  + (" — sectors compounded with global" if compound else ""))

    def _minimap_series(self) -> pd.Series:
        if self._modes is not None and "global" in self._modes:
            return self._modes["global"].dropna().cumsum()
        first = next(iter(self._layers.values()))
        return first[self._tickers].mean(axis=1).dropna().cumsum()

    # -- per-session app ----------------------------------------------------
    def _make_app(self) -> pn.Row:
        sync = SessionSync()
        w = self._selector_width
        ticker_w = pn.widgets.CrossSelector(
            name="tickers", options=self._tickers,
            value=self._tickers[:min(6, len(self._tickers))], width=w)
        layer_w = pn.widgets.CheckButtonGroup(
            name="transforms", options=self._layer_names,
            value=self._layer_names[:2], button_type="primary", width=w)
        focus_w = pn.widgets.RadioButtonGroup(
            name="focus", options=["All"] + self._layer_names, value="All",
            button_type="success", width=w)
        view_w = pn.widgets.RadioButtonGroup(
            options=["cumulative", "daily"],
            value="cumulative" if self._cumulative else "daily", width=w)

        curve_opts = opts.Curve(tools=["hover"], muted_alpha=0.08)
        overlay_opts = dict(width=self._width, autorange="y", show_grid=True,
                            legend_position="right",
                            legend_opts={"click_policy": "mute"},
                            active_tools=["pan", "wheel_zoom"],
                            hooks=[sync.capture_x_range])

        main = hv.DynamicMap(pn.bind(self._main_overlay, layers_sel=layer_w,
                                     tickers_sel=ticker_w, view=view_w,
                                     focus=focus_w))
        panels = [main.opts(curve_opts,
                            opts.Overlay(height=self._height, **overlay_opts))]

        controls = [pn.pane.Markdown("### Transforms\ncolor = transform · "
                                     "shade/dash = ticker (darkest = most spaced dashes)\n"
                                     "legend click mutes a curve"),
                    view_w, layer_w,
                    pn.pane.Markdown("**focus (step through transforms)**"), focus_w,
                    ticker_w]

        if self._modes is not None:
            mode_w = pn.widgets.CrossSelector(
                name="modes", options=self._mode_names,
                value=[m for m in ("global",) if m in self._mode_names], width=w)
            compound_w = pn.widgets.Checkbox(
                name="compound sector modes with global", value=False)
            modes_map = hv.DynamicMap(pn.bind(self._modes_overlay, modes_sel=mode_w,
                                              view=view_w, compound=compound_w))
            panels.append(modes_map.opts(
                curve_opts, opts.Overlay(height=self._modes_height, **overlay_opts)))
            controls += [pn.pane.Markdown("### Modes"), compound_w, mode_w]

        mm = self._minimap_series()
        minimap = hv.Curve((mm.index, mm.to_numpy(dtype=float)), "date", "value").opts(
            minimap_opts(width=self._width, height=self._minimap_height, sync=sync))

        layout = hv.Layout(panels + [minimap]).cols(1)
        return pn.Row(pn.Column(*controls), layout)
