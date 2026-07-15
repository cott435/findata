"""Oscillators feature group (v2).

Design from ``scripts/testing/momentum_analysis.py``: each oscillator is
computed at its own NATIVE lookback windows (short/long) directly from price —
the analog of trend's fast/slow price EMAs — rather than adding a second layer
of EMA windows on top of one fixed-window oscillator. Each native window then
gets ONE (fast, slow) EMA smoothing pair for its own temporal dynamics:

    raw           oscillator at native window w
    position      raw - ema_slow(raw)
    velocity      ema_fast(raw) - ema_slow(raw)
    acceleration  velocity - ema_signal(velocity)      (MACD-style)

plus cross-window contrasts (short minus long) per quantity — the actual
short-vs-long momentum signal.
"""
from __future__ import annotations

import pandas as pd

from findata.database.technical_calculators import ema
from findata.preprocess.base import FeatureGroup, per_ticker, _resolve_feature, calc_velocity_acceleration

# oscillators that take (ohlcv_df, period) and return a single Series
SERIES_OSCILLATORS = ('rsi', 'cci', 'willr', 'mfi', 'cmf', 'stochastic')
QUANTITIES = ("raw", "velocity", "acceleration")#, 'ema_fast', 'ema_slow')


class Oscillators(FeatureGroup):
    name = "momentum"
    default_scaler = "standard"

    def __init__(self, oscillators=SERIES_OSCILLATORS, windows=(("short", 14), ("long", 28)),
                 smoothing_pair=(6, 20), signal_window=9,
                 feature_set: str = "med", verbose: bool = False):
        super().__init__(feature_set=feature_set, verbose=verbose)
        oscillators = [oscillators] if isinstance(oscillators, str) else list(oscillators)
        unknown = [o for o in oscillators if o not in SERIES_OSCILLATORS]
        if unknown:
            raise ValueError(
                f"Unsupported oscillator(s) {unknown}: must return a single Series and "
                f"accept a period argument. Supported: {list(SERIES_OSCILLATORS)}")
        self.oscillators = oscillators
        self.windows = [tuple(w) for w in windows]
        self.smoothing_pair = tuple(smoothing_pair)
        self.signal_window = signal_window

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        #self.raw_data['bb_percent'] = (self.raw_data['close'] - get_column(self.raw_data, 'bb_lower')) / bb_range
        return per_ticker(data, self._engineer_ticker)

    def _engineer_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        fast, slow = self.smoothing_pair
        out: dict[str, pd.Series] = {}
        for osc in self.oscillators:
            per_window = {}
            for wname, w in self.windows:
                raw = _resolve_feature(df, osc, period=w)
                per_window[wname] = calc_velocity_acceleration(raw, fast, slow, signal_window=self.signal_window)
                for q in QUANTITIES:
                    out[f"{osc}_{wname}_{q}"] = per_window[wname][q]

        return pd.DataFrame(out, index=df.index)

    def plot_fe(self, engineered: pd.DataFrame, raw: pd.DataFrame, indicator='rsi',
                ticker: str | None = None):
        from collections import OrderedDict

        from findata.analysis import FeatureExplorer

        df = pd.concat([raw, engineered], axis=1)
        price_options = ['close'] + [c for c in raw.columns if 'ema_close' in c]

        data_dict = OrderedDict()
        data_dict['Price'] = {'value': [c for c in ('close', 'ema_close_26', 'ema_close_52')
                                        if c in price_options],
                              'options': price_options}
        data_dict[f'{indicator}'] = {'value': [c for c in engineered.columns if any([n in c for n in ['raw', 'ema']]) and indicator in c and 'contrast' not in c],
                                     'options': [c for c in engineered.columns if
                                                 any([n in c for n in ['raw', 'ema']]) and 'contrast' not in c]}
        data_dict[f'Derivatives'] = {'value': [c for c in engineered.columns if any([n in c for n in ['vel', 'pos', 'acc']]) and indicator in c and 'contrast' not in c],
                              'options': [c for c in engineered.columns if any([n in c for n in ['vel', 'pos', 'acc']]) and 'contrast' not in c]}

        explorer = FeatureExplorer(df, data_dict, minimap_feature='close', ticker=ticker)
        explorer.serve()
        return explorer


    def plot_fe_legacy(self, engineered: pd.DataFrame, raw: pd.DataFrame,
                ticker: str | None = None, tail: int = 200, save_path=None):
        """Construction plot in the style of momentum_analysis.py (one ticker)."""
        import matplotlib.pyplot as plt
        from collections import OrderedDict

        ticker = ticker or engineered.index.get_level_values("ticker")[0]
        fe = engineered.loc[ticker].tail(tail)
        px = raw.loc[ticker, "close"].tail(tail)
        osc = self.oscillators[0]
        wnames = [w[0] for w in self.windows]

        fig, axes = plt.subplots(len(wnames) + 2, 1, figsize=(16, 12), sharex=True)
        ax = axes[0]
        for wname, w in self.windows:
            ax.plot(fe.index, fe[f"{osc}_{wname}_raw"], lw=1.0, label=f"{osc}-{w} raw")
        ax.axhline(30, color="gray", lw=0.7, ls="--")
        ax.axhline(70, color="gray", lw=0.7, ls="--")
        ax.set_title(f"MOMENTUM: {osc} at native windows {[w for _, w in self.windows]} ({ticker})")
        ax.legend(fontsize=8)
        for i, wname in enumerate(wnames):
            ax = axes[1 + i]
            ax.plot(fe.index, fe[f"{osc}_{wname}_position"], lw=1.0, label="position", color="darkblue")
            ax.plot(fe.index, fe[f"{osc}_{wname}_velocity"], lw=1.4, label="velocity", color="darkred")
            ax.plot(fe.index, fe[f"{osc}_{wname}_acceleration"], lw=1.4, label="acceleration", color="darkorange")
            ax.axhline(0, color="gray", lw=0.7, ls="--")
            ax.set_title(f"{wname} position / velocity / acceleration")
            ax.legend(fontsize=8)
        axes[-1].plot(px.index, px.values, color="darkred", lw=1.4, label="close")
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130)
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    from findata import get_all_data, get_all_tickers
    from findata.configs import EXPERIMENT_DIR

    tickers = get_all_tickers()[:20]
    data, _ = get_all_data(tickers)

    mom = Oscillators()
    features = mom.engineer(data)

    mom.plot_fe(features, data)

    rsi = features['rsi_short_raw'].unstack("ticker").sort_index() .dropna(axis=1, how='any')

    from findata.analysis import CorrelationStructureAnalysis
    cc = CorrelationStructureAnalysis()
    res = cc.run(features)



    #both = Oscillators(oscillators=["rsi", "cci"])
    #print(f"\nWith oscillators=['rsi','cci']: {len(both.engineer(data).columns)} columns")

    import numpy as np
    ff = [c for c in features.columns if 'contrast' not in c]
    vals = features.dropna()[ff].to_numpy(dtype=float)
    corr = np.corrcoef(vals, rowvar=False)

    eigvals = np.linalg.eigvalsh(corr)[::-1]





