import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from findata.configs import EXPERIMENT_DIR
from findata.preprocess.calculators.technical import INDICATOR_FUNCS, ema

INDICATOR = 'rsi'
OUTPUT_DIR = EXPERIMENT_DIR / "momentum_analysis" / INDICATOR
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MOMENTUM_WINDOWS = [("short", 14), ("long", 28)]
MOMENTUM_SMOOTHING_PAIR = (5, 20)   # fixed (fast, slow) EMA pair used for velocity/acceleration, per RSI window
SIGNAL_WINDOW = 9          # EMA window used to smooth velocity -> derive acceleration (MACD-style)

def tail(data, n=500):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.tail(n)
    if isinstance(data, (list, tuple)):
        return [tail(d, n) for d in data]
    if isinstance(data, dict):
        return {k: tail(v, n=n) for k, v in data.items()}
    return data

def build_momentum_features(close, windows=MOMENTUM_WINDOWS, smoothing_pair=MOMENTUM_SMOOTHING_PAIR,
                            signal_window=SIGNAL_WINDOW, oscillator_name="rsi"):
    """
    Oscillators "scale" = RSI's OWN native lookback window (e.g. 14 vs. 28), computed
    directly from price -- NOT a secondary EMA-smoothing window applied on top of a
    single fixed-window RSI. This is the direct analog of trend's ema_fast/ema_slow
    on price: vary the lookback on the ORIGINAL series, don't add a second layer of
    windowing on an already-windowed derived quantity.

    Each native RSI window gets ONE (fast, slow) smoothing pair to derive that
    window's own velocity/acceleration (temporal dynamics), kept as a separate
    concept from the window-length axis itself.
    """
    fast_smooth, slow_smooth = smoothing_pair
    intermediates = {}
    per_window = {}

    for name, window in windows:
        raw = INDICATOR_FUNCS[oscillator_name](close, period=window)
        ema_fast = ema(raw, fast_smooth)
        ema_slow = ema(raw, slow_smooth)
        position = raw - ema_slow
        velocity = ema_fast - ema_slow
        acceleration = velocity - ema(velocity, signal_window)

        intermediates[name] = {"raw": raw, "ema_fast": ema_fast, "ema_slow": ema_slow,
                                "window": window}
        per_window[name] = {"raw": raw, "position": position, "velocity": velocity,
                             "acceleration": acceleration}

    features = {'price': close}
    for name, d in per_window.items():
        features[f"momentum_{name}_raw_{oscillator_name}"] = d["raw"]  # raw value per window -- let MP/ablation decide
        features[f"momentum_{name}_position"] = d["position"]
        features[f"momentum_{name}_velocity"] = d["velocity"]
        features[f"momentum_{name}_acceleration"] = d["acceleration"]

    # cross-WINDOW contrasts (the real short-vs-long-momentum signal): every pair of
    # native windows, for raw/position/velocity/acceleration independently
    window_names = [w[0] for w in windows]
    for i in range(len(window_names)):
        for j in range(i + 1, len(window_names)):
            a, b = window_names[i], window_names[j]
            for quantity in ["raw", "position", "velocity", "acceleration"]:
                features[f"momentum_contrast_{quantity}_{a}_minus_{b}"] = (
                    per_window[a][quantity] - per_window[b][quantity]
                )

    return pd.DataFrame(features), intermediates

def plot_momentum_construction(features, intermediates, oscillator_name="rsi", n=200):
    fig, axes = plt.subplots(4, 1, figsize=(25, 18), sharex=True)
    colors = {"short": "steelblue", "long": "crimson"}

    # panel 1: the two NATIVE-WINDOW raw RSI series + their own smoothing EMAs
    if n:
        features, intermediates = tail([features, intermediates], n=n)
    ax = axes[0]
    for name, interm in intermediates.items():
        dates = interm["raw"].index
        c = colors.get(name, None)
        ax.plot(dates, interm["raw"], color=c, lw=1.0,
                 label=f"{name}: raw {oscillator_name}-{interm['window']}")
        ax.plot(dates, interm["ema_slow"], color=c, lw=1.6, ls="--",
                 label=f"{name}: smoothing ema_slow")
    ax.set_title(f"MOMENTUM: {oscillator_name} computed at TWO NATIVE WINDOWS directly (14 vs. 28)")
    ax.axhline(30, color="gray", lw=0.7, ls="--")
    ax.axhline(70, color="gray", lw=0.7, ls="--")
    ax.legend(fontsize=8, ncol=2)

    # panel 2: raw RSI contrast (the headline short-vs-long momentum signal)
    ax = axes[1]
    ax.plot(dates, (features[f"momentum_short_raw_{oscillator_name}"] - 50)/3, color="darkblue", lw=0.6,
            label=f"Raw {oscillator_name} (scaled)", ls='--')
    ax.plot(dates, features["momentum_short_velocity"], color="darkred", lw=1.6,
            label="Velocity")
    ax.plot(dates, features["momentum_short_acceleration"], color="darkorange", lw=1.6,
             label="Acceleration")
    ax.axhline(0, color="gray", lw=0.7, ls='--')
    ax.set_title(f"Short Velocity and Acceleration")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(dates, (features[f"momentum_long_raw_{oscillator_name}"] - 50)/3, color="darkblue", lw=0.6,
            label=f"Raw {oscillator_name} (scaled)", ls='--')
    ax.plot(dates, features["momentum_long_velocity"], color="darkred", lw=1.6,
            label="Velocity")
    ax.plot(dates, features["momentum_long_acceleration"], color="darkorange", lw=1.6,
             label="Acceleration")
    ax.axhline(0, color="gray", lw=0.7, ls='--')
    ax.set_title(f"Long Velocity and Acceleration")
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.plot(dates, features["price"], color="darkred", lw=1.6,
            label="Close Price")

    """ax.plot(dates, features["momentum_contrast_velocity_short_minus_long"], color="darkred", lw=1.6,
            label="Velocity contrast (short minus long)")
    ax.plot(dates, features["momentum_contrast_raw_short_minus_long"], color="darkblue", lw=1.6,
             label=f"Raw {oscillator_name} contrast (short minus long)")
    ax.plot(dates, features["momentum_contrast_acceleration_short_minus_long"], color="darkorange", lw=1.6,
            label=f"Raw {oscillator_name} contrast (short minus long)")
    ax.axhline(0, color="gray", lw=0.7, ls='--')
    ax.set_title(f"{oscillator_name} contrast")
    ax.legend(fontsize=8)"""

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_momentum_feature_construction.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '2_momentum_feature_construction.png'}")

def plot_quick_correlation_check(features):
    """
    Not the full MP/clustering pipeline -- just a fast raw-correlation sanity check
    so you can see, before running the full diagnostic script, whether the contrast
    columns visually look distinct from the level columns.
    """
    corr = features.corr()

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, label="correlation")
    ax.set_title("Quick raw-correlation check: levels vs. contrast columns\n"
                 "(this is NOT the MP/clustering diagnostic -- run full_feature_taxonomy_analysis.py for that)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_quick_raw_correlation_check.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '3_quick_raw_correlation_check.png'}")

def main():
    from findata import get_all_tickers, get_price_data

    ticker = get_all_tickers()[0]
    data = get_price_data(ticker).loc[ticker]
    close = data["close"]

    momentum_features, momentum_intermediates = build_momentum_features(close, oscillator_name=INDICATOR)
    plot_momentum_construction(momentum_features, momentum_intermediates, oscillator_name="rsi")
    plot_quick_correlation_check(momentum_features)





if __name__ == "__main__":
    main()
