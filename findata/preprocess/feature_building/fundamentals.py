"""Fundamental feature group: slow-moving cross-sectional signals from the
derived statement items.

Consumes the ``fund_<item>`` columns that ``get_all_data(...,
include_fundamentals=True)`` merges into the wide panel (the group itself
never touches the database, matching the other groups). Each item is clipped
to its OutputSpec fe_bounds; scale rules group items by their fe_scaling
key. Cross-sectional z-scoring belongs in the processing chain
(``cs_zscore``), not here.
"""

from __future__ import annotations

import pandas as pd

from findata.preprocess.calculators.fundamental import FUNDAMENTAL_CALCULATORS
from findata.preprocess.feature_building.base import FeatureGroup, ScaleRule

ITEM_SPECS = {spec.item: spec for calc in FUNDAMENTAL_CALCULATORS for spec in calc.outputs}


class Fundamentals(FeatureGroup):
    """Point-in-time fundamental ratios as pipeline features."""

    name = 'fundamental'
    default_scaler = 'robust'

    def __init__(self, feature_set: str = 'med', verbose: bool = False, items=None):
        super().__init__(feature_set, verbose)
        self.items = {item: spec for item, spec in ITEM_SPECS.items()
                      if items is None or item in items}

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = {item: f'fund_{item}' for item in self.items if f'fund_{item}' in data.columns}
        if not cols:
            raise KeyError('No fund_* columns in the panel -- load it with '
                           'get_all_data(include_fundamentals=True)')
        out = pd.DataFrame(index=data.index)
        for item, col in cols.items():
            series = data[col]
            bounds = self.items[item].fe_bounds
            if bounds is not None:
                series = series.clip(*bounds)
            out[item] = series
        return out

    def scale_rules(self, columns) -> list[ScaleRule]:
        by_scaler: dict[str, list] = {}
        for col in columns:
            spec = self.items.get(col)
            key = spec.fe_scaling if spec is not None else self.default_scaler
            by_scaler.setdefault(key, []).append(col)
        return [ScaleRule(columns=cols, scaler=scaler)
                for scaler, cols in by_scaler.items()]


if __name__ == "__main__":
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:20]
    data, _ = get_all_data(tickers, include_fundamentals=True)

    mom = Fundamentals()
    features = mom.engineer(data)

    #mom.plot_fe(features, data)

    rsi = features['rsi_short_raw'].unstack("ticker").sort_index() .dropna(axis=1, how='any')
