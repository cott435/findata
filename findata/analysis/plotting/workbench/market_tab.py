"""Market tab: production overview at a selected as-of date.

Sector x horizon return heatmap, market breadth + cumulative global/sector
modes, and the "interesting residuals" screen -- names whose idiosyncratic
move over the ranking horizon is large while the market+sector contribution
is flat or opposite (moving on their own).
"""

from __future__ import annotations

import holoviews as hv
import pandas as pd
import panel as pn
from holoviews import opts

from findata.analysis.plotting.style import LAYER_PALETTE
from findata.analysis.plotting.workbench import panels as pel
from findata.analysis.plotting.workbench.data import WorkbenchData

HORIZON_OPTIONS = (1, 5, 21, 63)


class MarketTab:
    def __init__(self, data: WorkbenchData, *, width: int = 1050,
                 rail_width: int = 280):
        self.data = data
        self.width = width
        self.rail_width = rail_width

    def build(self) -> pn.Row:
        data = self.data
        last_date = data.close_panel().index[-1].date()
        asof_w = pn.widgets.DatePicker(name='as of', value=last_date,
                                       width=self.rail_width)
        horizons_w = pn.widgets.CheckButtonGroup(options=list(HORIZON_OPTIONS),
                                                 value=[1, 5, 21, 63],
                                                 width=self.rail_width)
        rank_w = pn.widgets.Select(name='residual ranking horizon',
                                   options=list(HORIZON_OPTIONS), value=21,
                                   width=self.rail_width)
        top_n_w = pn.widgets.IntSlider(name='table rows', start=10, end=60,
                                       value=25, width=self.rail_width)

        def sector_view(asof, horizons):
            if not horizons:
                horizons = [21]
            frame = data.sector_returns(pd.Timestamp(asof), tuple(sorted(horizons)))
            return pel.heatmap(frame, kdims=['horizon', 'sector'], vdim='mean log ret',
                               title=f'sector mean log return, trailing horizons '
                                     f'@ {asof}', width=self.width // 2,
                               height=360).opts(opts.HeatMap(shared_axes=False))

        def breadth_view():
            breadth = data.breadth()
            vdim = hv.Dimension('m_breadth', label='share above EMA')
            series = {c: breadth[c] for c in breadth.columns}
            over = pel.series_overlay(series, vdim)
            half = hv.HLine(0.5).opts(color='#888888', line_dash='dotted')
            return (over * half).opts(
                opts.Curve(tools=['hover'], line_width=1.3),
                opts.Overlay(width=self.width // 2 - 20, height=360,
                             show_grid=True, legend_position='bottom',
                             shared_axes=False, title='breadth'))

        def modes_view():
            modes = data.decompose('log_ret')['modes']
            vdim = hv.Dimension('m_modes', label='cumulative mode return')
            series = {c: modes[c].cumsum() for c in modes.columns}
            return pel.series_overlay(series, vdim).opts(
                opts.Curve(tools=['hover'], line_width=1.3),
                opts.NdOverlay(width=self.width, height=280, autorange='y',
                               show_grid=True, legend_position='right',
                               legend_opts={'click_policy': 'mute'},
                               shared_axes=False,
                               title='cumulative global + sector return modes'))

        def movers_table(asof, horizon, top_n):
            movers = data.residual_movers(pd.Timestamp(asof), horizon)
            frame = movers.head(top_n).reset_index(names='ticker')
            frame['flag'] = frame['own_move'].map({True: '⚡ own move', False: ''})
            frame = frame[['ticker', 'sector', 'ret', 'market_sector',
                           'residual', 'flag']].round(4)
            return pel.table(frame.set_index('ticker'), width=self.width // 2,
                             height=420)

        def movers_scatter(asof, horizon):
            movers = data.residual_movers(pd.Timestamp(asof), horizon).reset_index(
                names='ticker')
            movers['color'] = movers['own_move'].map(
                {True: LAYER_PALETTE[1], False: LAYER_PALETTE[7]})
            points = hv.Points(
                movers,
                kdims=[hv.Dimension(('market_sector', 'market+sector contribution')),
                       hv.Dimension(('residual', 'residual return'))],
                vdims=['ticker', 'sector', 'color']).opts(
                color='color', size=6, alpha=0.75, tools=['hover'],
                width=self.width // 2 - 20, height=420, show_grid=True,
                shared_axes=False,
                title=f'residual vs market move ({horizon}d)')
            zero_h = hv.HLine(0).opts(color='#aaaaaa', line_width=1)
            zero_v = hv.VLine(0).opts(color='#aaaaaa', line_width=1)
            return points * zero_h * zero_v

        row1 = pn.Row(pn.bind(sector_view, asof=asof_w, horizons=horizons_w),
                      breadth_view())
        row2 = modes_view()
        row3 = pn.Row(pn.bind(movers_table, asof=asof_w, horizon=rank_w,
                              top_n=top_n_w),
                      pn.bind(movers_scatter, asof=asof_w, horizon=rank_w))

        rail = pn.Column(asof_w, pn.pane.Markdown('**heatmap horizons**'),
                         horizons_w, rank_w, top_n_w, width=self.rail_width + 30)
        return pn.Row(rail, pn.Column(row1, row2, row3))
