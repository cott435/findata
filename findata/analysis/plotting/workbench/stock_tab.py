"""Stock tab: single-name explorer.

Main candlestick chart with price-scale overlays, plus dynamic indicator
panels. Each panel owns a hierarchical selector and a view toggle:
'raw' shows the selected series; 'modes' overlays the reconstructed
global+sector market signal for the first selected key (from the universe
mode decomposition); 'modes+resid' additionally appends a linked residual
row beneath the panel. Selection changes patch DynamicMaps in place (zoom
survives); structural changes (add/remove panel, residual row) rebuild the
layout with the zoom window restored.
"""

from __future__ import annotations

import holoviews as hv
import pandas as pd
import panel as pn
from holoviews import opts

from findata.analysis.plotting.base import SessionSync, minimap_opts
from findata.analysis.plotting.workbench import panels as pel
from findata.analysis.plotting.workbench.data import WorkbenchData, key_label
from findata.analysis.plotting.workbench.presets import PRESETS
from findata.analysis.plotting.workbench.selectors import IndicatorSelector
from findata.preprocess.calculators.technical import EMA_WINDOWS

MAX_PANELS = 6


class RestoringSync(SessionSync):
    """SessionSync that re-applies a saved zoom window after a rebuild."""

    def __init__(self):
        super().__init__()
        self.pending = None

    def capture_x_range(self, plot, element):
        super().capture_x_range(plot, element)
        if self.pending is not None:
            start, end = self.pending
            plot.handles['x_range'].start = start
            plot.handles['x_range'].end = end

    def attach_range_tool(self, plot, element):
        super().attach_range_tool(plot, element)
        self.pending = None  # minimap renders last -> restore is done


def _overlay_options() -> list:
    keys = [f'ema_close_{p}' for p in EMA_WINDOWS['ema_close']]
    return keys + ['bb_upper_20', 'bb_middle_20', 'bb_lower_20']


class StockTab:
    def __init__(self, data: WorkbenchData, *, width: int = 1050,
                 height: int = 240, price_height: int = 320,
                 minimap_height: int = 80, rail_width: int = 280):
        self.data = data
        self.width = width
        self.height = height
        self.price_height = price_height
        self.minimap_height = minimap_height
        self.rail_width = rail_width

    # every widget/DynamicMap below is built fresh per session
    def build(self) -> pn.Row:
        data = self.data
        sync = RestoringSync()

        ticker_w = pn.widgets.Select(name='ticker', options=data.universe,
                                     width=self.rail_width)
        persist_w = pn.widgets.Checkbox(name='persist computed windows', value=False)
        overlay_w = pn.widgets.MultiChoice(name='price overlays',
                                           options=_overlay_options(),
                                           value=['ema_close_26'],
                                           width=self.rail_width)
        add_btn = pn.widgets.Button(name='➕ add panel', button_type='primary',
                                    width=self.rail_width)
        specs: list[dict] = []
        plot_area = pn.Column(sizing_mode='fixed')
        rail_specs = pn.Column()

        curve_opts = opts.Curve(tools=['hover'], muted_alpha=0.1, line_width=1.4)

        def panel_opts(height):
            return opts.NdOverlay(
                width=self.width, height=height, autorange='y', show_grid=True,
                legend_position='right', legend_opts={'click_policy': 'mute'},
                active_tools=['pan', 'wheel_zoom'], hooks=[sync.capture_x_range])

        # ---------------- plot callbacks (bound per panel) ---------------- #

        def price_view(ticker, overlays, persist):
            frame = data.ticker_frame(ticker)
            vdim = hv.Dimension('s_price', label=f'{ticker} price')
            over = pel.candlestick(frame, vdim)
            series = {}
            for key in overlays:
                s = data.ticker_series(ticker, key, persist=persist or None)
                if not s.dropna().empty:
                    series[key_label(key)] = s
            if series:
                over = over * pel.series_overlay(series, vdim)
            return over.opts(
                opts.Overlay(width=self.width, height=self.price_height,
                             autorange='y', show_grid=True, ylabel='price',
                             active_tools=['pan', 'wheel_zoom'],
                             legend_position='right',
                             legend_opts={'click_policy': 'mute'},
                             hooks=[sync.capture_x_range], title=str(ticker)))

        def panel_view(ticker, keys, view, persist, vdim):
            series = {key_label(k): data.ticker_series(ticker, k, persist=persist or None)
                      for k in keys}
            if view != 'raw' and keys:
                decomp = data.decompose(keys[0])
                combined = decomp['panels']['combined_modes']
                if ticker in combined.columns:
                    series['market signal'] = combined[ticker]
            return pel.series_overlay(series, vdim,
                                      dashes={'market signal': 'dashed'})

        def resid_view(ticker, keys, vdim):
            series = {}
            if keys:
                decomp = data.decompose(keys[0])
                residual = decomp['panels']['residual']
                if ticker in residual.columns:
                    series[f'{key_label(keys[0])} residual'] = residual[ticker]
            return pel.series_overlay(series, vdim)

        def minimap_view(ticker):
            s = data.ticker_frame(ticker)['close'].dropna()
            return hv.Curve((s.index, s.to_numpy(float)), 'date', 'mm')

        # ---------------- layout assembly -------------------------------- #

        def rebuild():
            if sync.x_range is not None:
                sync.pending = (sync.x_range.start, sync.x_range.end)
            sync.x_range = None
            items = [hv.DynamicMap(pn.bind(price_view, ticker=ticker_w,
                                           overlays=overlay_w, persist=persist_w))]
            for spec in specs:
                vdim = hv.Dimension(f's_val_{spec["id"]}', label=f'panel {spec["id"]}')
                dmap = hv.DynamicMap(pn.bind(panel_view, ticker=ticker_w,
                                             keys=spec['selector'].param.value,
                                             view=spec['view'], persist=persist_w,
                                             vdim=vdim))
                items.append(dmap.opts(curve_opts, panel_opts(self.height)))
                if spec['view'].value == 'modes+resid':
                    rdim = hv.Dimension(f's_res_{spec["id"]}',
                                        label=f'panel {spec["id"]} residual')
                    rmap = hv.DynamicMap(pn.bind(resid_view, ticker=ticker_w,
                                                 keys=spec['selector'].param.value,
                                                 vdim=rdim))
                    items.append(rmap.opts(curve_opts,
                                           panel_opts(max(120, self.height - 80))))
            minimap = hv.DynamicMap(pn.bind(minimap_view, ticker=ticker_w)).opts(
                minimap_opts(width=self.width, height=self.minimap_height,
                             sync=sync, framewise=True))
            items.append(minimap)
            plot_area[:] = [hv.Layout(items).cols(1)]
            rail_specs[:] = [spec['card'] for spec in specs]

        def make_spec(keys=None):
            spec_id = (specs[-1]['id'] + 1) if specs else 1
            selector = IndicatorSelector(width=self.rail_width - 20, value=keys or [])
            view = pn.widgets.RadioButtonGroup(
                options=['raw', 'modes', 'modes+resid'], value='raw',
                width=self.rail_width - 20)
            remove = pn.widgets.Button(name='remove panel', button_type='warning',
                                       width=self.rail_width - 20)
            spec = {'id': spec_id, 'selector': selector, 'view': view,
                    'resid_shown': False}

            def on_remove(_):
                specs.remove(spec)
                rebuild()

            def on_view(event):
                shown = event.new == 'modes+resid'
                if shown != spec['resid_shown']:
                    spec['resid_shown'] = shown
                    rebuild()

            remove.on_click(on_remove)
            view.param.watch(on_view, 'value')
            spec['card'] = pn.Card(view, selector.panel, remove,
                                   title=f'panel {spec_id}', collapsed=False,
                                   width=self.rail_width)
            return spec

        def on_add(_):
            if len(specs) >= MAX_PANELS:
                return
            specs.append(make_spec())
            rebuild()

        def apply_preset(name):
            def handler(_):
                preset = PRESETS[name]
                overlay_w.value = [k for k in preset['overlays']
                                   if k in overlay_w.options]
                specs.clear()
                for keys in preset['panels'][:MAX_PANELS]:
                    specs.append(make_spec(keys=keys))
                rebuild()
            return handler

        add_btn.on_click(on_add)
        preset_buttons = []
        for name in PRESETS:
            button = pn.widgets.Button(name=name, button_type='default', width=105)
            button.on_click(apply_preset(name))
            preset_buttons.append(button)
        preset_row = pn.FlexBox(*preset_buttons)

        specs.append(make_spec(keys=['rsi_14']))
        rebuild()

        rail = pn.Column(ticker_w, preset_row, overlay_w, persist_w, add_btn,
                         rail_specs, width=self.rail_width + 30)
        def remove_last():
            if specs:
                specs.pop()
                rebuild()

        # handles for headless tests (harmless in production; overwritten per
        # build, so never shared live state read by anything)
        self._session = {
            'specs': specs, 'overlays': overlay_w, 'plot_area': plot_area,
            'ticker': ticker_w, 'add': lambda: on_add(None),
            'apply_preset': lambda name: apply_preset(name)(None),
            'set_view': lambda i, v: specs[i]['view'].param.update(value=v),
            'remove_last': remove_last,
        }
        return pn.Row(rail, plot_area)
