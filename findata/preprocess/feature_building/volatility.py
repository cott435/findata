"""Volatility feature group — feature engineering ported unchanged from v1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from findata.preprocess.feature_building.base import FeatureGroup, get_column, per_ticker


def fe_volatility(df, windows, feature_set: str = 'med') -> pd.DataFrame:
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
        self.windows = [20, 100] if feature_set == 'low' else [7, 20, 100]

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        out = per_ticker(data, lambda df: fe_volatility(
            df, windows=self.windows, feature_set=self.feature_set))
        out['bb_bandwidth'] = (
            (get_column(data, 'bb_upper') - get_column(data, 'bb_lower'))
            / get_column(data, 'bb_middle')
        )
        out['atr_norm'] = get_column(data, 'atr') / data['close']
        return out

    def plot_fe(self, engineered: pd.DataFrame, raw: pd.DataFrame,ticker: str | None = None):
        from collections import OrderedDict
        from findata.analysis import FeatureExplorer

        df = pd.concat([raw, engineered], axis=1)
        price_options = ['close'] + [c for c in raw.columns if 'ema_close' in c]

        data_dict = OrderedDict()
        data_dict['Price'] = {'value': [c for c in ('close', 'ema_close_26', 'ema_close_52')
                                        if c in price_options],
                              'options': price_options}
        data_dict[f'Volatility'] = {'value': ['vol20', 'vol100'],
                              'options': list(engineered.columns)}
        data_dict[f'Volatility2'] = {'value': ['bb_bandwidth', 'atr_norm'],
                              'options': list(engineered.columns)}
        explorer = FeatureExplorer(df, data_dict, minimap_feature='close', ticker=ticker)
        explorer.serve()
        return explorer


if __name__ == "__main__":
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:20]
    data, _ = get_all_data(tickers)

    vol = Volatility()
    features = vol.engineer(data)

    vol.plot_fe(features, data)
