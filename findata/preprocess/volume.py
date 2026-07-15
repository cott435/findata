"""Volume feature group — feature engineering ported unchanged from v1."""
from __future__ import annotations

import pandas as pd

from findata.preprocess.base import (FeatureGroup, ScaleRule, fe_oscillator_momentum,
                                     get_column_names, per_ticker)


def calc_z_score_vel(df, ema_windows):
    out = pd.DataFrame(index=df.index)
    for i in range(len(ema_windows) - 1):
        f, s = ema_windows[i], ema_windows[i + 1]
        price_vel = df[f'ema_close_{f}'] - df[f'ema_close_{s}']
        out[f'price_zvel{f}_{s}'] = (price_vel - price_vel.rolling(window=s * 2).mean()) / price_vel.rolling(window=s * 2).std()
        obv_vel = df[f'ema_obv_{f}'] - df[f'ema_obv_{s}']
        out[f'obv_zvel{f}_{s}'] = (obv_vel - obv_vel.rolling(window=s * 2).mean()) / obv_vel.rolling(window=s * 2).std()
        ad_vel = df[f'ema_ad_{f}'] - df[f'ema_ad_{s}']
        out[f'ad_zvel{f}_{s}'] = (ad_vel - ad_vel.rolling(window=s * 2).mean()) / ad_vel.rolling(window=s * 2).std()
    return out


class Volume(FeatureGroup):
    name = "volume"
    default_scaler = "robust"

    def __init__(self, ema_windows=(12, 26, 52), feature_set: str = "med", verbose: bool = False):
        super().__init__(feature_set=feature_set, verbose=verbose)
        self.ema_windows = list(ema_windows)

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        return per_ticker(data, lambda df: calc_z_score_vel(df, self.ema_windows))

    def scale_rules(self, columns) -> list[ScaleRule]:
        """zvel columns are already rolling z-scores — pass through unscaled (v1 behavior)."""
        columns = list(columns)
        scale_cols = [c for c in columns if 'zvel' not in c]
        pass_cols = [c for c in columns if 'zvel' in c]
        rules = [ScaleRule(columns=scale_cols, scaler=self.default_scaler)]
        if pass_cols:
            rules.append(ScaleRule(columns=pass_cols, kind="passthrough"))
        return rules

    def plot_fe(self, engineered: pd.DataFrame, raw: pd.DataFrame,ticker: str | None = None):
        from collections import OrderedDict
        from findata.analysis import FeatureExplorer

        df = pd.concat([raw, engineered], axis=1)
        price_options = ['close'] + [c for c in raw.columns if 'ema_close' in c]

        data_dict = OrderedDict()
        data_dict['Price'] = {'value': [c for c in ('close', 'ema_close_26', 'ema_close_52')
                                        if c in price_options],
                              'options': price_options}
        data_dict[f'Short'] = {'value': [c for c in engineered.columns if f"{self.ema_windows[0]}_{self.ema_windows[1]}" in c],
                              'options': list(engineered.columns)}
        data_dict[f'Long'] = {'value': [c for c in engineered.columns if f"{self.ema_windows[1]}_{self.ema_windows[2]}" in c],
                              'options': list(engineered.columns)}
        explorer = FeatureExplorer(df, data_dict, minimap_feature='close', ticker=ticker)
        explorer.serve()
        return explorer

if __name__ == "__main__":
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:20]
    data, _ = get_all_data(tickers)

    volume = Volume()
    features = volume.engineer(data)

    volume.plot_fe(features, data)

