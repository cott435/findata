"""Compare tab: cross-ticker comparison sheet for one indicator.

Zone A -- levels: the focus indicator across the selected tickers
(normalized), or per-ticker small multiples. Zone B -- relative: spread
curves (vs the selection mean or vs the first ticker) and each ticker's
latest cross-sectional percentile within the universe. Zone C -- predictive:
cumulative daily rank-IC of the indicator vs forward returns, an IC-decay
curve across horizons, and the per-horizon summary table.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from holoviews import opts

from findata.analysis.plotting.base import NORMALIZATIONS, _normalize, minimap_opts
from findata.analysis.plotting.style import LAYER_PALETTE
from findata.analysis.plotting.workbench import panels as pel
from findata.analysis.plotting.workbench.data import WorkbenchData, key_label
from findata.analysis.plotting.workbench.stock_tab import RestoringSync

HORIZONS = (1, 5, 10, 21, 42, 63)


class CompareTab:
    def __init__(self, data: WorkbenchData, *, width: int = 1050,
                 height: int = 260, rail_width: int = 300):
        self.data = data
        self.width = width
        self.height = height
        self.rail_width = rail_width

    def build(self) -> pn.Row:
        from findata.analysis.plotting.workbench.selectors import SingleSelector
        data = self.data
        sync = RestoringSync()

        default = data.universe[:4]
        tickers_w = pn.widgets.CrossSelector(name='tickers', options=data.universe,
                                             value=default, width=self.rail_width,
                                             height=220)
        focus = SingleSelector(width=self.rail_width, value='rsi_14')
        norm_w = pn.widgets.RadioButtonGroup(options=list(NORMALIZATIONS),
                                             value='raw', width=self.rail_width)
        horizon_w = pn.widgets.Select(name='IC horizon (days)', options=list(HORIZONS),
                                      value=21, width=self.rail_width)
        spread_w = pn.widgets.RadioButtonGroup(options=['vs mean', 'vs first'],
                                               value='vs mean', width=self.rail_width)
        multiples_w = pn.widgets.Checkbox(name='small multiples', value=False)

        curve_opts = opts.Curve(tools=['hover'], muted_alpha=0.1, line_width=1.3)

        def panel_opts(height):
            return opts.NdOverlay(
                width=self.width, height=height, autorange='y', show_grid=True,
                legend_position='right', legend_opts={'click_policy': 'mute'},
                active_tools=['pan', 'wheel_zoom'], hooks=[sync.capture_x_range])

        def _selected_panel(key, selected):
            panel = data.item_panel(key)
            return panel[[t for t in selected if t in panel.columns]]

        # -------- Zone A: levels ------------------------------------------ #

        def level_view(key, selected, norm):
            vdim = hv.Dimension('c_level', label=key_label(key))
            panel = _selected_panel(key, selected)
            series = {t: _normalize(panel[t].dropna(), norm) for t in panel.columns}
            return pel.series_overlay(series, vdim)

        def multiples_layout(key, selected, norm):
            panel = _selected_panel(key, selected)
            cells = []
            for i, t in enumerate(panel.columns):
                s = _normalize(panel[t].dropna(), norm)
                vdim = hv.Dimension(f'c_sm_{i}', label=t)
                cells.append(hv.Curve((s.index, s.to_numpy(float)), 'date', vdim)
                             .opts(color=LAYER_PALETTE[i % len(LAYER_PALETTE)],
                                   width=self.width // 2, height=180,
                                   autorange='y', show_grid=True, title=t,
                                   hooks=[sync.capture_x_range]))
            return hv.Layout(cells).cols(2) if cells else hv.Layout([hv.Curve([])])

        # -------- Zone B: relative ---------------------------------------- #

        def spread_view(key, selected, mode):
            vdim = hv.Dimension('c_spread', label=f'{key_label(key)} spread')
            panel = _selected_panel(key, selected)
            series = {}
            if len(panel.columns) > 1:
                base = panel.mean(axis=1) if mode == 'vs mean' else panel.iloc[:, 0]
                label = 'mean' if mode == 'vs mean' else panel.columns[0]
                for t in panel.columns:
                    if mode == 'vs first' and t == panel.columns[0]:
                        continue
                    series[f'{t} − {label}'] = (panel[t] - base)
            return pel.series_overlay(series, vdim)

        def percentile_strip(key, selected):
            panel = data.item_panel(key)
            latest = panel.dropna(how='all').iloc[-1]
            pct = latest.rank(pct=True)
            rows = [(t, 'pctile', float(pct.get(t, np.nan))) for t in selected]
            hm = hv.HeatMap(rows, kdims=['ticker', 'metric'], vdims='pct')
            labels = hv.Labels([(t, m, f'{v:.0%}') for t, m, v in rows
                                if not np.isnan(v)],
                               kdims=['ticker', 'metric'], vdims='label')
            return (hm.opts(width=self.width, height=90, cmap='RdBu_r',
                            clim=(0, 1), colorbar=False, toolbar=None,
                            title='latest cross-sectional percentile (universe)',
                            shared_axes=False)
                    * labels.opts(text_color='black', text_font_size='9pt'))

        # -------- Zone C: predictive -------------------------------------- #

        def ic_view(key, horizon):
            vdim = hv.Dimension('c_ic', label=f'cum IC h={horizon}')
            daily, summary = data.fwd_ic(key, horizon)
            return pel.cumulative_ic(daily, vdim).opts(
                opts.Curve(width=self.width, height=200, show_grid=True,
                           autorange='y', hooks=[sync.capture_x_range],
                           title=f'cumulative daily rank IC — {key_label(key)} '
                                 f'(ICIR {summary["icir"]:.2f})'))

        def decay_view(key):
            decay = data.ic_decay(key, HORIZONS)
            curve = hv.Curve((decay.index, decay['ic_mean']), 'horizon',
                             hv.Dimension('c_decay', label='mean IC')).opts(
                color=LAYER_PALETTE[0], line_width=2)
            dots = hv.Scatter((decay.index, decay['ic_mean']), 'horizon',
                              'c_decay').opts(color=LAYER_PALETTE[0], size=6)
            zero = hv.HLine(0).opts(color='#888888', line_dash='dotted')
            return (curve * dots * zero).opts(
                opts.Curve(width=420, height=200, show_grid=True,
                           shared_axes=False, title='IC decay vs horizon'))

        def summary_table(key):
            rows = []
            for h in HORIZONS:
                _, s = data.fwd_ic(key, h)
                rows.append({'horizon': h, 'ic_mean': round(float(s['ic_mean']), 4),
                             'icir': round(float(s['icir']), 3),
                             't_stat': round(float(s['t_stat']), 2),
                             'p_value': round(float(s['p_value']), 4)})
            return pel.table(pd.DataFrame(rows).set_index('horizon'),
                             width=self.width - 460, height=200)

        # -------- assembly ------------------------------------------------ #

        plot_area = pn.Column()

        def rebuild(*_):
            if sync.x_range is not None:
                sync.pending = (sync.x_range.start, sync.x_range.end)
            sync.x_range = None
            if multiples_w.value:  # static grid, rebuilt on any change
                zone_a = multiples_layout(focus.value, tickers_w.value, norm_w.value)
            else:
                zone_a = hv.DynamicMap(pn.bind(level_view, key=focus.param.value,
                                               selected=tickers_w, norm=norm_w)) \
                    .opts(curve_opts, panel_opts(self.height))
            zone_b = hv.DynamicMap(pn.bind(spread_view, key=focus.param.value,
                                           selected=tickers_w, mode=spread_w)) \
                .opts(curve_opts, panel_opts(200))
            strip = hv.DynamicMap(pn.bind(percentile_strip, key=focus.param.value,
                                          selected=tickers_w))
            ic = hv.DynamicMap(pn.bind(ic_view, key=focus.param.value,
                                       horizon=horizon_w))
            row_c = pn.Row(
                pn.bind(decay_view, key=focus.param.value),
                pn.bind(summary_table, key=focus.param.value))
            plot_area[:] = [zone_a, zone_b, strip, ic, row_c]

        def rebuild_if_multiples(_event):
            if multiples_w.value:
                rebuild()

        multiples_w.param.watch(rebuild, 'value')
        tickers_w.param.watch(rebuild_if_multiples, 'value')
        focus.param.watch(rebuild_if_multiples, 'value')
        norm_w.param.watch(rebuild_if_multiples, 'value')
        rebuild()

        rail = pn.Column(tickers_w, pn.pane.Markdown('**focus indicator**'),
                         focus.panel, norm_w, spread_w, multiples_w, horizon_w,
                         width=self.rail_width + 30)
        return pn.Row(rail, plot_area)
