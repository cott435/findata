"""Trend feature group — feature engineering ported unchanged from v1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from findata.preprocess.feature_building.base import (FeatureGroup, get_column, per_ticker, sig_span,
                                                      sort_columns, _resolve_ema, calc_velocity_acceleration)


def fe_unbounded_momentum(df, ema_windows=None, feature_set='med', base_name='ema_close'):
    """Log-ratio velocity — appropriate for price EMAs (always positive)."""
    assert feature_set in ['low', 'med', 'high']
    if ema_windows is None:
        ema_windows = [12, 26, 52] if feature_set == 'low' else [6, 12, 26, 52, 104]
        if feature_set == 'high':
            ema_windows.append(208)
    out = pd.DataFrame(index=df.index)
    emas = []
    for i in range(len(ema_windows) - 1):
        f, s = ema_windows[i], ema_windows[i + 1]
        emas.extend([f, s])
        fast = df[f'{base_name}_{f}']
        slow = df[f'{base_name}_{s}']
        assert all([(fast > 0).all(), (slow > 0).all()])
        out[f'vel{f}_{s}'] = ratio = np.log(fast / slow)
        out[f'acc{f}_{s}'] = ratio - ratio.ewm(span=sig_span(s), adjust=False).mean()
    for e in set(emas):
        out[f'rel{e}'] = np.log(df['close'] / df[f'{base_name}_{e}'])
    return sort_columns(out, ['rel', 'vel', 'acc'])

QUANTITIES = ("velocity", "acceleration")

class Trend(FeatureGroup):
    name = "trend"
    default_scaler = "robust"
    default_arcsinh = True

    def __init__(self, ema_windows=None, feature_set: str = "med", verbose: bool = False):
        super().__init__(feature_set=feature_set, verbose=verbose)
        self.ema_windows = [12, 26, 52] if feature_set == 'low' else [12, 26, 52, 104]

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        out = per_ticker(data, self._engineer_ticker)
        out['adx'] = get_column(data, 'adx')
        out['plus_di'] = get_column(data, 'plus_di')
        out['minus_di'] = get_column(data, 'minus_di')
        return out

    def _engineer_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for i in range(len(self.ema_windows) - 1):
            fast, slow = self.ema_windows[i], self.ema_windows[i + 1]
            ema_fast = _resolve_ema(df, f'ema_close', fast)
            ema_slow = _resolve_ema(df, f'ema_close', slow)
            vel_acc = calc_velocity_acceleration(df['close'], ema_fast, ema_slow, log=True)
            for q in QUANTITIES:
                out[f"{q}_{fast}/{slow}"] = vel_acc[q]
        if self.feature_set != 'low':
            for e in self.ema_windows:
                out[f'rel_{e}'] = np.log(df['close'] / _resolve_ema(df, f'ema_close', e))
        return pd.DataFrame(out, index=df.index)

    def plot_fe(self, engineered: pd.DataFrame, raw: pd.DataFrame,ticker: str | None = None):
        from collections import OrderedDict
        from findata.analysis import FeatureExplorer

        df = pd.concat([raw, engineered], axis=1)
        price_options = ['close'] + [c for c in raw.columns if 'ema_close' in c]

        data_dict = OrderedDict()
        data_dict['Price'] = {'value': [c for c in ('close', 'ema_close_26', 'ema_close_52')
                                        if c in price_options],
                              'options': price_options}
        data_dict[f'Derivatives'] = {'value': [c for c in engineered.columns if any([n in c for n in ['vel', 'pos', 'acc']])],
                              'options': [c for c in engineered.columns if any([n in c for n in ['vel', 'pos', 'acc', 'rel']])]}
        data_dict[f'ADX'] = {'value': ['adx', 'plus_di', 'minus_di'],
                              'options': ['adx', 'plus_di', 'minus_di']}
        explorer = FeatureExplorer(df, data_dict, minimap_feature='close', ticker=ticker)
        explorer.serve()
        return explorer

    def plot_fe_legacy(self, engineered: pd.DataFrame, raw: pd.DataFrame,
                ticker: str | None = None, tail: int = 500):
        import matplotlib.pyplot as plt
        from findata.preprocess.feature_building.base import get_columns

        ticker = ticker or engineered.index.get_level_values("ticker")[0]
        fe = engineered.loc[ticker].tail(tail)
        px = raw.loc[ticker].tail(tail)
        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        axes[0].plot(px.index, px['close'], label='close')
        for w in (12, 26, 52):
            axes[0].plot(px.index, px[f'ema_close_{w}'], lw=0.9, label=f'ema{w}')
        for ax, keys in zip(axes[1:], ('vel', 'acc', ['adx', 'plus_di', 'minus_di'])):
            cols = get_columns(fe, keys) if isinstance(keys, str) else keys
            for c in cols:
                ax.plot(fe.index, fe[c], lw=1.0, label=c)
            ax.axhline(0, color='gray', lw=0.7, ls='--')
            ax.legend(fontsize=8, ncol=3)
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].set_title(f"Trend construction ({ticker})")
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:20]
    data, _ = get_all_data(tickers)

    trend = Trend()
    features = trend.engineer(data)

    trend.plot_fe(features, data)
