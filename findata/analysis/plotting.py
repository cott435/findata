"""Interactive feature browser (HoloViews/Bokeh via Panel).

FeatureExplorer renders one linked panel per feature group with a shared,
zoomable x-axis:

- curves are added/removed live from CrossSelector widgets WITHOUT resetting
  the current zoom: each panel is a DynamicMap patched in place, never rebuilt
- every panel's y-axis autoscales to the data inside the visible x-window
  (``autorange='y'``); pan/zoom on any panel rescales all of them
- a minimap at the bottom drives the shared x-range through a Bokeh RangeTool
- optional per-series normalization (z-score / min-max / demean, computed over
  the full sample) and a ticker dropdown when given a (ticker, date) MultiIndex
- click a legend entry to mute its curve; hover shows date/value

Implementation notes
--------------------
Widgets and DynamicMaps are constructed PER SESSION (``serve`` passes a
factory to ``pn.serve``). Serving shared instances breaks live sessions:
when any other session is destroyed (page reload, stray HTTP request),
holoviews' cleanup strips the stream subscriptions that push plot updates.

Panels share their x-axis through the common 'date' dimension, while each
panel gets a unique y-dimension name AND label — bokeh links axes whose
dimension labels match, so identical labels would link the y-ranges too.
"""
from __future__ import annotations

import holoviews as hv
import pandas as pd
import panel as pn
from bokeh.models import CustomJS, GlyphRenderer, RangeTool
from holoviews import opts

hv.extension('bokeh')
pn.extension()

NORMALIZATIONS = ("raw", "z-score", "min-max", "demean")

# Re-emit the RangesUpdate event after a data patch settles so holoviews'
# autorange recomputes the y-window. Without this the y-range goes stale when
# existing curves are swapped in place (e.g. on a ticker change) because the
# autorange callback races bokeh's spatial index rebuild.
_RETRIGGER_JS = """
setTimeout(function() {
    var view = Bokeh.index.find_one(fig);
    if (view != null) { view.trigger_ranges_update_event(); }
}, 60);
"""


def _normalize(s: pd.Series, how: str) -> pd.Series:
    if how == "z-score":
        std = s.std()
        return (s - s.mean()) / (std or 1.0)
    if how == "min-max":
        span = s.max() - s.min()
        return (s - s.min()) / (span or 1.0)
    if how == "demean":
        return s - s.mean()
    return s


class FeatureExplorer:
    """Linked multi-panel time-series browser.

    df          wide frame indexed by date, or by (ticker, date) — a MultiIndex
                adds a ticker dropdown that re-slices every panel
    structure   {panel_title: spec} where spec is either a plain list of
                columns, or {'options': [...], 'value': [...]} to pre-select
    ticker      initial ticker shown when df has a (ticker, date) MultiIndex
    """

    def __init__(self, df: pd.DataFrame, structure: dict, *, width: int = 1300,
                 height: int = 250, minimap_feature: str | None = None,
                 minimap_height: int = 90, normalization: bool = True,
                 ticker: str | None = None, selector_width: int = 420):
        self._df = df
        self._structure = {
            name: (dict(spec) if isinstance(spec, dict) else {"options": list(spec)})
            for name, spec in structure.items()
        }
        self._frames: dict = {}
        self._width = width
        self._height = height
        self._minimap_feature = minimap_feature
        self._minimap_height = minimap_height
        self._normalization = normalization
        self._selector_width = selector_width
        self._tickers = (list(dict.fromkeys(df.index.get_level_values(0)))
                         if isinstance(df.index, pd.MultiIndex) else None)
        self._default_ticker = ticker

    # -- data access (shared across sessions, read-only) ----------------------
    def _frame(self, ticker) -> pd.DataFrame:
        if ticker not in self._frames:
            frame = self._df.xs(ticker, level=0) if ticker is not None else self._df
            if not isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.set_axis(pd.to_datetime(frame.index))
            self._frames[ticker] = frame
        return self._frames[ticker]

    def _overlay(self, name, vdim, features, norm, ticker) -> hv.NdOverlay:
        frame = self._frame(ticker)
        curves = {}
        for f in features:
            if f not in frame.columns:
                continue
            s = _normalize(frame[f].dropna(), norm)
            curves[f] = hv.Curve((s.index, s.to_numpy(dtype=float)), "date", vdim)
        if not curves:  # a DynamicMap must always return the same type
            curves["(none)"] = hv.Curve([], "date", vdim)
        return hv.NdOverlay(curves, kdims="feature").opts(title=str(name))

    def _minimap_curve(self, ticker) -> hv.Curve:
        frame = self._frame(ticker)
        col = self._minimap_feature or frame.columns[0]
        s = frame[col].dropna()
        return hv.Curve((s.index, s.to_numpy(dtype=float)), "date", "value")

    # -- per-session app construction ------------------------------------------
    def _make_app(self) -> pn.Row:
        session = {"x_range": None}

        def capture_x_range(plot, element):
            session["x_range"] = plot.handles["x_range"]
            fig = plot.state
            for renderer in fig.select({"type": GlyphRenderer}):
                cds = renderer.data_source
                attached = cds.js_property_callbacks.get("change:data", [])
                if any("fe_retrigger" in cb.tags for cb in attached):
                    continue
                cds.js_on_change("data", CustomJS(
                    args={"fig": fig}, code=_RETRIGGER_JS, tags=["fe_retrigger"]))

        def attach_range_tool(plot, element):
            fig = plot.state
            if session["x_range"] is None or any(
                    isinstance(t, RangeTool) for t in fig.tools):
                return
            tool = RangeTool(x_range=session["x_range"])
            tool.overlay.fill_alpha = 0.15
            fig.add_tools(tool)

        ticker_w = (pn.widgets.Select(
            name="ticker", options=self._tickers,
            value=self._default_ticker or self._tickers[0],
            width=self._selector_width) if self._tickers else None)
        norm_w = (pn.widgets.RadioButtonGroup(
            options=list(NORMALIZATIONS), value="raw",
            width=self._selector_width) if self._normalization else None)

        curve_opts = opts.Curve(tools=["hover"], muted_alpha=0.15, line_width=1.3)
        panel_opts = opts.NdOverlay(
            width=self._width, height=self._height, autorange="y",
            show_grid=True, legend_position="right",
            legend_opts={"click_policy": "mute"},
            active_tools=["pan", "wheel_zoom"], hooks=[capture_x_range])

        selectors, panels = [], []
        for i, (name, spec) in enumerate(self._structure.items()):
            selector = pn.widgets.CrossSelector(
                name=str(name), options=spec["options"],
                value=spec.get("value", []), width=self._selector_width)
            selectors.append(selector)
            vdim = hv.Dimension(f"value_{i}", label=str(name))

            def callback(features, norm, ticker, name=name, vdim=vdim):
                return self._overlay(name, vdim, features, norm, ticker)

            dmap = hv.DynamicMap(pn.bind(callback, features=selector,
                                         norm=norm_w if norm_w is not None else "raw",
                                         ticker=ticker_w))
            panels.append(dmap.opts(curve_opts, panel_opts))

        minimap = hv.DynamicMap(pn.bind(self._minimap_curve, ticker=ticker_w)).opts(
            opts.Curve(width=self._width, height=self._minimap_height,
                       yaxis=None, xlabel="", color="#888888",
                       default_tools=[], shared_axes=False, toolbar=None,
                       framewise=True,  # rescale to the new ticker's extent
                       hooks=[attach_range_tool]))

        layout = hv.Layout(panels + [minimap]).cols(1)
        controls = [w for w in (ticker_w, norm_w) if w is not None]
        return pn.Row(pn.Column(*controls, *selectors), layout)

    # -- entry points -----------------------------------------------------------
    @property
    def app(self) -> pn.Row:
        """A fresh app instance (for notebook display or manual serving)."""
        return self._make_app()

    def serve(self, port: int = 0, show: bool = True, **kwargs):
        """Serve the explorer; each browser session gets fresh widgets."""
        return pn.serve(self._make_app, port=port, show=show, **kwargs)


