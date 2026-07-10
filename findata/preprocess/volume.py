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
        osc_cols = get_column_names(['mfi', 'cmf'])

        def apply_fe(df):
            return pd.concat([
                calc_z_score_vel(df, self.ema_windows),
                fe_oscillator_momentum(df, osc_cols, feature_set=self.feature_set),
            ], axis=1)

        return per_ticker(data, apply_fe)

    def scale_rules(self, columns) -> list[ScaleRule]:
        """zvel columns are already rolling z-scores — pass through unscaled (v1 behavior)."""
        columns = list(columns)
        scale_cols = [c for c in columns if 'zvel' not in c]
        pass_cols = [c for c in columns if 'zvel' in c]
        rules = [ScaleRule(columns=scale_cols, scaler=self.default_scaler)]
        if pass_cols:
            rules.append(ScaleRule(columns=pass_cols, kind="passthrough"))
        return rules
