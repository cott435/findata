"""PCA / RMT tab: cross-sectional structure of one indicator panel.

Eigenvalue density against the Marchenko-Pastur law (signal count, top-mode
share), explained variance of the leading modes, the global + sector mode
time series from the decomposition, and per-ticker loadings by sector.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from holoviews import opts

from findata.analysis.plotting.style import LAYER_PALETTE
from findata.analysis.plotting.workbench import panels as pel
from findata.analysis.plotting.workbench.data import WorkbenchData, key_label
from findata.analysis.plotting.workbench.selectors import SingleSelector


class PcaTab:
    def __init__(self, data: WorkbenchData, *, width: int = 1050,
                 rail_width: int = 300):
        self.data = data
        self.width = width
        self.rail_width = rail_width

    def build(self) -> pn.Row:
        data = self.data
        focus = SingleSelector(width=self.rail_width, value='rsi_14')
        scopes = ['Universe'] + sorted(data.sectors.dropna().unique())
        scope_w = pn.widgets.Select(name='scope', options=scopes, width=self.rail_width)
        dates = data.close_panel().index
        range_w = pn.widgets.DateRangeSlider(name='window', start=dates[0].date(),
                                             end=dates[-1].date(),
                                             value=(dates[0].date(), dates[-1].date()),
                                             width=self.rail_width)

        def _scope_tickers(scope):
            if scope == 'Universe':
                return None
            return sorted(data.sectors.index[data.sectors == scope])

        def spectrum_view(key, scope, date_range):
            spec = data.spectrum(key, _scope_tickers(scope), date_range)
            return pel.mp_spectrum_elements(spec, 'p_density').opts(
                opts.Histogram(width=self.width // 2, height=300,
                               shared_axes=False))

        def variance_view(key, scope, date_range):
            spec = data.spectrum(key, _scope_tickers(scope), date_range)
            eigvals = np.sort(np.asarray(spec['eigvals']))[::-1][:15]
            share = eigvals / spec['n']
            bars = hv.Bars((list(range(1, len(share) + 1)), share),
                           hv.Dimension('p_mode', label='mode'),
                           hv.Dimension('p_share', label='variance share'))
            return bars.opts(width=self.width // 2 - 20, height=300,
                             color=LAYER_PALETTE[0], shared_axes=False,
                             title=f'top modes (λ+ cut at {spec["n_signal"]} signal)')

        def modes_view(key):
            modes = data.decompose(key)['modes']
            vdim = hv.Dimension('p_modes', label='cumulative mode value')
            series = {c: modes[c].cumsum() for c in modes.columns}
            return pel.series_overlay(series, vdim).opts(
                opts.Curve(tools=['hover'], line_width=1.3),
                opts.NdOverlay(width=self.width, height=280, autorange='y',
                               show_grid=True, legend_position='right',
                               legend_opts={'click_policy': 'mute'},
                               title=f'global + sector modes — {key_label(key)}'))

        def loadings_view(key):
            loadings = data.decompose(key)['loadings']
            order = loadings.sort_values(['sector', 'beta_global']).index
            frame = loadings.loc[order, ['beta_global', 'beta_sector']]
            frame.index.name = 'ticker'
            return pel.heatmap(frame.T, kdims=['ticker', 'beta'], vdim='loading',
                               title='mode loadings by sector (sector-sorted)',
                               width=self.width, height=220).opts(
                opts.HeatMap(xrotation=90, shared_axes=False))

        def stats_view(key, scope, date_range):
            spec = data.spectrum(key, _scope_tickers(scope), date_range)
            rows = {k: spec[k] for k in ('n', 't', 'q', 'mean_corr', 'std_corr',
                                         'sigma2', 'lam_plus', 'n_signal',
                                         'top_eig', 'top_eig_share')}
            df = pd.DataFrame({'value': pd.Series(rows).round(4)})
            return pel.table(df, width=320, height=300)

        row1 = pn.Row(
            hv.DynamicMap(pn.bind(spectrum_view, key=focus.param.value,
                                  scope=scope_w, date_range=range_w)),
            pn.bind(variance_view, key=focus.param.value, scope=scope_w,
                    date_range=range_w),
            pn.bind(stats_view, key=focus.param.value, scope=scope_w,
                    date_range=range_w))
        row2 = hv.DynamicMap(pn.bind(modes_view, key=focus.param.value))
        row3 = pn.bind(loadings_view, key=focus.param.value)

        rail = pn.Column(pn.pane.Markdown('**indicator**'), focus.panel, scope_w,
                         range_w, width=self.rail_width + 30)
        return pn.Row(rail, pn.Column(row1, row2, row3))
