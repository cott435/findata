"""Trend feature group — feature engineering ported unchanged from v1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from findata.preprocess.base import (FeatureGroup, get_column, per_ticker, sig_span,
                                     sort_columns)


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


class Trend(FeatureGroup):
    name = "trend"
    default_scaler = "robust"
    default_arcsinh = True

    def __init__(self, ema_windows=None, feature_set: str = "med", verbose: bool = False):
        super().__init__(feature_set=feature_set, verbose=verbose)
        self.ema_windows = ema_windows

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        out = per_ticker(data, lambda df: fe_unbounded_momentum(
            df, ema_windows=self.ema_windows, feature_set=self.feature_set))
        out['adx'] = get_column(data, 'adx')
        out['plus_di'] = get_column(data, 'plus_di')
        out['minus_di'] = get_column(data, 'minus_di')
        return out

    def plot_fe(self, engineered: pd.DataFrame, raw: pd.DataFrame,
                ticker: str | None = None, tail: int = 500):
        import matplotlib.pyplot as plt
        from findata.preprocess.base import get_columns

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
