"""Hierarchical indicator picker driven by the calculator registry.

The tree is family -> calculator -> output items -> periods, so the widget
surface stays small and deeper options only appear as you descend. 'Add'
appends the item x period combinations as selection keys ('rsi_14',
'fund:earnings_yield') to ``value`` -- a param.List others ``pn.bind`` on --
and each active key gets a removable chip.
"""

from __future__ import annotations

import param
import panel as pn

from findata.analysis.plotting.workbench.data import FUND_PREFIX, key_label
from findata.preprocess.calculators.base import column_name, registry_tree


def _key(family: str, item: str, period: int) -> str:
    if family == 'fundamental':
        return FUND_PREFIX + item
    return column_name(item, period)


class IndicatorSelector(param.Parameterized):
    """Cascading family/calculator/item/period picker with an active-key list."""

    value = param.List(default=[], doc='Active selection keys.')

    def __init__(self, *, families=('technical', 'fundamental'), value=None,
                 width: int = 300, name_prefix: str = '', **params):
        super().__init__(**params)
        self._tree = registry_tree(tuple(families))
        self._width = width
        self._chips = pn.Column(sizing_mode='stretch_width')

        family_opts = [f for f in families if f in self._tree]
        self._family = pn.widgets.Select(name=f'{name_prefix}family',
                                         options=family_opts, width=width)
        self._calc = pn.widgets.Select(name='indicator', width=width)
        self._items = pn.widgets.MultiChoice(name='outputs', width=width)
        self._periods = pn.widgets.MultiChoice(name='periods', width=width)
        self._custom = pn.widgets.IntInput(name='custom period', value=0, start=0,
                                           width=width // 2)
        self._add = pn.widgets.Button(name='Add', button_type='primary',
                                      width=width // 2 - 10)

        self._family.param.watch(self._on_family, 'value')
        self._calc.param.watch(self._on_calc, 'value')
        self._add.on_click(self._on_add)
        self._on_family(None)
        if value:
            self.value = list(value)
        self._render_chips()

    # -- cascading updates ------------------------------------------------ #

    def _on_family(self, _):
        calcs = sorted(self._tree.get(self._family.value, {}))
        self._calc.options = calcs
        if calcs:
            self._calc.value = calcs[0]
        self._on_calc(None)

    def _on_calc(self, _):
        node = self._tree.get(self._family.value, {}).get(self._calc.value, {})
        outputs = list(node.get('outputs', []))
        periods = list(node.get('default_periods', []))
        self._items.options = outputs
        self._items.value = outputs[:1]
        self._periods.options = periods
        self._periods.value = periods[:1]
        fundamental = self._family.value == 'fundamental'
        self._periods.visible = not fundamental
        self._custom.visible = not fundamental

    def _on_add(self, _):
        periods = list(self._periods.value)
        if self._custom.value:
            periods.append(int(self._custom.value))
        if self._family.value == 'fundamental':
            periods = [0]
        new = [_key(self._family.value, item, period)
               for item in self._items.value for period in (periods or [0])]
        merged = list(dict.fromkeys([*self.value, *new]))
        if merged != self.value:
            self.value = merged
            self._render_chips()

    def _remove(self, key):
        def handler(_):
            self.value = [k for k in self.value if k != key]
            self._render_chips()
        return handler

    def _render_chips(self):
        rows = []
        for key in self.value:
            close = pn.widgets.Button(name='✕', width=28, height=24,
                                      button_type='light')
            close.on_click(self._remove(key))
            rows.append(pn.Row(pn.pane.Markdown(f'`{key_label(key)}`',
                                                margin=(0, 4)), close, height=30))
        self._chips[:] = rows or [pn.pane.Markdown('*nothing selected*', margin=(0, 4))]

    def set_value(self, keys: list):
        """Programmatic selection (presets)."""
        self.value = list(dict.fromkeys(keys))
        self._render_chips()

    @property
    def panel(self) -> pn.Column:
        return pn.Column(self._family, self._calc, self._items, self._periods,
                         pn.Row(self._custom, self._add), self._chips,
                         width=self._width + 20)


class SingleSelector(param.Parameterized):
    """One-key variant for tabs that focus a single indicator."""

    value = param.String(default='')

    def __init__(self, *, families=('technical', 'fundamental'),
                 value: str = 'rsi_14', width: int = 300, **params):
        super().__init__(**params)
        self._tree = registry_tree(tuple(families))
        self._family = pn.widgets.Select(name='family', options=[
            f for f in families if f in self._tree], width=width)
        self._calc = pn.widgets.Select(name='indicator', width=width)
        self._item = pn.widgets.Select(name='output', width=width)
        self._period = pn.widgets.Select(name='period', width=width)
        self._family.param.watch(self._on_family, 'value')
        self._calc.param.watch(self._on_calc, 'value')
        for widget in (self._item, self._period):
            widget.param.watch(self._emit, 'value')
        self._on_family(None)
        self.value = value

    def _on_family(self, _):
        calcs = sorted(self._tree.get(self._family.value, {}))
        self._calc.options = calcs
        if calcs and self._calc.value not in calcs:
            self._calc.value = calcs[0]
        self._on_calc(None)

    def _on_calc(self, _):
        node = self._tree.get(self._family.value, {}).get(self._calc.value, {})
        self._item.options = list(node.get('outputs', []))
        if self._item.options and self._item.value not in self._item.options:
            self._item.value = self._item.options[0]
        periods = list(node.get('default_periods', []))
        self._period.options = periods
        if periods and self._period.value not in periods:
            self._period.value = periods[0]
        self._period.visible = self._family.value != 'fundamental'
        self._emit(None)

    def _emit(self, _):
        if not self._item.value:
            return
        period = 0 if self._family.value == 'fundamental' else (self._period.value or 0)
        self.value = _key(self._family.value, self._item.value, period)

    @property
    def panel(self) -> pn.Column:
        return pn.Column(self._family, self._calc, self._item, self._period)
