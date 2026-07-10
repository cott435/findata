"""Volatility feature group — feature engineering ported unchanged from v1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from findata.preprocess.base import FeatureGroup, get_column, per_ticker


def fe_volatility(df, windows=None, feature_set='med'):
    assert feature_set in ['low', 'med', 'high']
    if windows is None:
        windows = [20, 100] if feature_set == 'low' else [7, 20, 100]
    out = pd.DataFrame(index=df.index)
    returns = df["close"].pct_change()
    for window in windows:
        out[f"vol{window}"] = returns.rolling(window).std()
        park = (np.log(df["high"] / df["low"]) ** 2) / (4 * np.log(2))
        out[f"vol_park{window}"] = np.sqrt(park.rolling(window).mean())
        rs_part = (
            np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
            np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
        )
        out[f"vol_rs{window}"] = np.sqrt(rs_part.rolling(window).mean())
        if feature_set == 'high':
            gk = (
                0.5 * (np.log(df["high"] / df["low"])) ** 2
                - (2 * np.log(2) - 1) * (np.log(df["close"] / df["open"])) ** 2
            )
            out[f"vol_gk{window}"] = np.sqrt(gk.rolling(window).mean())
            rv_proxy = ((df["high"] - df["low"]) / df["close"]) ** 2 \
                       + ((df["close"] - df["open"]) / df["close"]) ** 2
            out[f"vol_rvp{window}"] = np.sqrt(rv_proxy.rolling(window).mean())
    return out


class Volatility(FeatureGroup):
    name = "volatility"
    default_scaler = "power"

    def __init__(self, windows=None, feature_set: str = "med", verbose: bool = False):
        super().__init__(feature_set=feature_set, verbose=verbose)
        self.windows = windows

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        out = per_ticker(data, lambda df: fe_volatility(
            df, windows=self.windows, feature_set=self.feature_set))
        out['bb_bandwidth'] = (
            (get_column(data, 'bb_upper') - get_column(data, 'bb_lower'))
            / get_column(data, 'bb_middle')
        )
        out['atr_norm'] = get_column(data, 'atr') / data['close']
        return out
