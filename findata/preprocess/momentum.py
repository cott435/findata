"""Momentum feature group (v2).

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

from findata.database.technical_calculators import INDICATOR_FUNCS, ema
from findata.preprocess.base import FeatureGroup, per_ticker

# oscillators that take (ohlcv_df, period) and return a single Series
SERIES_OSCILLATORS = ('rsi', 'cci', 'willr', 'mfi', 'cmf')
QUANTITIES = ("raw", "position", "velocity", "acceleration")


class Momentum(FeatureGroup):
    name = "momentum"
    default_scaler = "standard"

    def __init__(self, oscillators=("rsi",), windows=(("short", 14), ("long", 28)),
                 smoothing_pair=(5, 20), signal_window=9,
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
        return per_ticker(data, self._engineer_ticker)

    def _engineer_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        fast, slow = self.smoothing_pair
        out: dict[str, pd.Series] = {}
        for osc in self.oscillators:
            per_window = {}
            for wname, w in self.windows:
                raw = INDICATOR_FUNCS[osc](df, period=w)
                ema_slow = ema(raw, slow)
                velocity = ema(raw, fast) - ema_slow
                per_window[wname] = {
                    "raw": raw,
                    "position": raw - ema_slow,
                    "velocity": velocity,
                    "acceleration": velocity - ema(velocity, self.signal_window),
                }
                for q in QUANTITIES:
                    out[f"{osc}_{wname}_{q}"] = per_window[wname][q]
            # cross-window contrasts: every native-window pair, per quantity
            wnames = [w[0] for w in self.windows]
            for i in range(len(wnames)):
                for j in range(i + 1, len(wnames)):
                    a, b = wnames[i], wnames[j]
                    for q in QUANTITIES:
                        out[f"{osc}_contrast_{q}_{a}_minus_{b}"] = (
                            per_window[a][q] - per_window[b][q])
        return pd.DataFrame(out, index=df.index)

    def plot_fe(self, engineered: pd.DataFrame, raw: pd.DataFrame,
                ticker: str | None = None, tail: int = 200, save_path=None):
        """Construction plot in the style of momentum_analysis.py (one ticker)."""
        import matplotlib.pyplot as plt

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
    from findata import get_price_data, get_all_tickers
    from findata.configs import EXPERIMENT_DIR

    tickers = get_all_tickers()[:1]
    data = get_price_data(tickers)

    mom = Momentum()
    features = mom.engineer(data)
    print(f"Momentum v2 columns ({len(features.columns)}):")
    for c in features.columns:
        print(f"  {c}")

    out_dir = EXPERIMENT_DIR / "momentum_analysis" / "rsi"
    out_dir.mkdir(parents=True, exist_ok=True)
    mom.plot_fe(features, data, save_path=out_dir / "v2_momentum_feature_construction.png")

    both = Momentum(oscillators=["rsi", "cci"])
    print(f"\nWith oscillators=['rsi','cci']: {len(both.engineer(data).columns)} columns")
