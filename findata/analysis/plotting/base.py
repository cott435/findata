"""Shared machinery for the interactive explorers (HoloViews/Bokeh via Panel).

Every explorer follows the same session contract:

- Widgets and DynamicMaps are constructed PER SESSION: ``ExplorerBase.serve``
  passes the ``_make_app`` factory to ``pn.serve``. Serving shared instances
  breaks live sessions — when any other session is destroyed (page reload,
  stray HTTP request), holoviews' cleanup strips the stream subscriptions that
  push plot updates.
- A :class:`SessionSync` instance links the main panels to a minimap: the
  ``capture_x_range`` hook goes on every main panel (records the shared
  x-range and installs the autorange retrigger), ``attach_range_tool`` goes on
  the minimap curve (binds a Bokeh RangeTool to the captured range).
- Panels share their x-axis through the common 'date' dimension; give each
  panel a UNIQUE y-dimension name AND label, because bokeh links axes whose
  dimension labels match.
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


def _to_dt_index(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame
    return frame.set_axis(pd.to_datetime(frame.index))


class SessionSync:
    """Per-session x-range link between main panels and the minimap RangeTool."""

    def __init__(self):
        self.x_range = None

    def capture_x_range(self, plot, element):
        self.x_range = plot.handles["x_range"]
        fig = plot.state
        for renderer in fig.select({"type": GlyphRenderer}):
            cds = renderer.data_source
            attached = cds.js_property_callbacks.get("change:data", [])
            if any("fe_retrigger" in cb.tags for cb in attached):
                continue
            cds.js_on_change("data", CustomJS(
                args={"fig": fig}, code=_RETRIGGER_JS, tags=["fe_retrigger"]))

    def attach_range_tool(self, plot, element):
        fig = plot.state
        if self.x_range is None or any(isinstance(t, RangeTool) for t in fig.tools):
            return
        tool = RangeTool(x_range=self.x_range)
        tool.overlay.fill_alpha = 0.15
        fig.add_tools(tool)


def minimap_opts(*, width: int, height: int, sync: SessionSync,
                 framewise: bool = False, color: str = "#888888") -> opts.Curve:
    """Standard minimap styling with the RangeTool hook attached."""
    return opts.Curve(width=width, height=height, yaxis=None, xlabel="",
                      color=color, default_tools=[], shared_axes=False,
                      toolbar=None, framewise=framewise,
                      hooks=[sync.attach_range_tool])


class ExplorerBase:
    """Serving contract: subclasses build a fresh layout per session in
    ``_make_app`` (never share widget/DynamicMap instances across sessions)."""

    def _make_app(self) -> pn.Row:
        raise NotImplementedError

    @property
    def app(self) -> pn.Row:
        """A fresh app instance (for notebook display or manual serving)."""
        return self._make_app()

    def serve(self, port: int = 0, show: bool = True, **kwargs):
        """Serve the explorer; each browser session gets fresh widgets."""
        return pn.serve(self._make_app, port=port, show=show, **kwargs)
