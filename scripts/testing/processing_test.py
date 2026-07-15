"""
build_position_velocity_acceleration_features.py

Constructs position / velocity / acceleration features for TREND (EMA of price)
and MOMENTUM (EMA of RSI, or any bounded oscillator) at multiple time SCALES
(short vs. long EMA pairs), plus explicit CONTRAST features that isolate what's
different between scales -- rather than relying on PCA to find that difference
buried in near-noise eigenvalues.

WHY CONTRASTS EXIST (recap of the reasoning, so the code makes sense standalone):
  Position, velocity, and acceleration all derive from the SAME underlying price
  (or oscillator) process, so they're structurally correlated with each other.
  Short-scale and long-scale versions of the same quantity are ALSO structurally
  correlated, since they share most of their input history. PCA/MP will correctly
  collapse this whole family down to ~1-2 signal components -- that's a
  statement about what's statistically distinguishable at your sample size, not
  a statement that short-vs-long carries no useful information. So: build the
  short-vs-long DIFFERENCE explicitly as its own feature, rather than hoping an
  unsupervised method surfaces it.

CONVENTIONS USED (matching your existing calculation style):
  TREND (price-based):
    - position  = log(close / ema_slow)            [bounded-ratio quantity -> log]
    - velocity  = log(ema_fast / ema_slow)          [ratio of two EMAs -> log]
    - acceleration = velocity - EMA(velocity, signal_window)   [MACD-histogram style]
    - raw close price itself is NOT included as a feature (non-stationary)

  MOMENTUM (oscillator-based, e.g. RSI -- bounded, already centered-ish):
    - position  = raw_oscillator - ema_slow(oscillator)   [difference, not ratio/log --
                                                            oscillator can cross zero-ish
                                                            regions where ratios misbehave]
    - velocity  = ema_fast(oscillator) - ema_slow(oscillator)   [difference, centered at 0]
    - acceleration = velocity - EMA(velocity, signal_window)    [MACD-histogram style]
    - raw oscillator value is ALSO included as its own feature per your note that
      you're unsure if it's useful -- let the downstream MP/ablation script decide.

  BOTH:
    - "scale" = one (fast, slow) EMA window pair, e.g. short=(10,20), long=(50,100)
    - contrasts are computed between EVERY pair of scales, for position, velocity,
      AND acceleration independently: contrast = value_scale_A - value_scale_B
      (always a plain difference -- since velocity/acceleration/position are already
      constructed to be centered near 0, differencing them keeps that property;
      never take a ratio of an already-centered quantity)

This script does NOT run the MP/clustering diagnostics -- feed its output into
full_feature_taxonomy_analysis.py (prefix column names with "trend_" / "momentum_")
to test which of these actually clear the noise threshold and merit inclusion.

HOW TO USE WITH YOUR REAL DATA:
  Replace `build_synthetic_price_series()` with your real close-price series
  (a single ticker to start, since this script is for visual sanity-checking the
  construction, not the full-universe statistical test). Adjust SCALE PAIRS below
  to match the windows you actually use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("./feature_construction_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Scale pairs: (scale_name, fast_window, slow_window), shortest to longest.
# Same pairs are used for both trend and momentum by default -- change independently if needed.
TREND_SCALE_PAIRS = [("short", 10, 20), ("long", 50, 100)]
# NOTE ON MOMENTUM SCALES: RSI is already a windowed calculation (its own lookback,
# e.g. 14 periods), unlike raw close price. So "short vs. long" for momentum should
# vary RSI's OWN native window (RSI-14 vs. RSI-28), not add a second EMA-smoothing
# window on top of a single fixed-window RSI -- that would conflate "how much
# momentum history" with "how much smoothing," which are different things.
# Each native RSI window still gets ONE small smoothing pair to derive velocity/
# acceleration (temporal dynamics of that window), kept separate from the window axis.
MOMENTUM_RSI_WINDOWS = [("short", 14), ("long", 28)]
MOMENTUM_SMOOTHING_PAIR = (5, 20)   # fixed (fast, slow) EMA pair used for velocity/acceleration, per RSI window
SIGNAL_WINDOW = 9          # EMA window used to smooth velocity -> derive acceleration (MACD-style)


# =============================================================================
# INDICATOR PRIMITIVES
# =============================================================================

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def rsi(close, window=14):
    """Standard Wilder's-smoothing RSI, 0-100 bounded."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# =============================================================================
# TREND FEATURE FAMILY (EMA of price)
# =============================================================================

def build_trend_features(close, scale_pairs=TREND_SCALE_PAIRS, signal_window=SIGNAL_WINDOW):
    """
    Returns (features_df, intermediates) where intermediates holds the raw EMAs
    for plotting, and features_df holds the actual position/velocity/acceleration
    + cross-scale contrast columns, prefixed 'trend_'.
    """
    intermediates = {}
    per_scale = {}

    for name, fast_w, slow_w in scale_pairs:
        ema_fast = ema(close, fast_w)
        ema_slow = ema(close, slow_w)
        position = np.log(close / ema_slow)
        velocity = np.log(ema_fast / ema_slow)
        acceleration = velocity - ema(velocity, signal_window)

        intermediates[name] = {"ema_fast": ema_fast, "ema_slow": ema_slow,
                                "fast_w": fast_w, "slow_w": slow_w}
        per_scale[name] = {"position": position, "velocity": velocity, "acceleration": acceleration}

    features = {}
    for name, d in per_scale.items():
        features[f"trend_{name}_position"] = d["position"]
        features[f"trend_{name}_velocity"] = d["velocity"]
        features[f"trend_{name}_acceleration"] = d["acceleration"]

    # cross-scale contrasts: every pair of scales, for each of position/velocity/acceleration
    scale_names = [s[0] for s in scale_pairs]
    for i in range(len(scale_names)):
        for j in range(i + 1, len(scale_names)):
            a, b = scale_names[i], scale_names[j]
            for quantity in ["position", "velocity", "acceleration"]:
                features[f"trend_contrast_{quantity}_{a}_minus_{b}"] = (
                    per_scale[a][quantity] - per_scale[b][quantity]
                )

    return pd.DataFrame(features), intermediates


# =============================================================================
# MOMENTUM FEATURE FAMILY (EMA of an oscillator, e.g. RSI)
# =============================================================================

def build_momentum_features(close, rsi_windows=MOMENTUM_RSI_WINDOWS, smoothing_pair=MOMENTUM_SMOOTHING_PAIR,
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

    for name, rsi_w in rsi_windows:
        raw_rsi = rsi(close, window=rsi_w)
        ema_fast = ema(raw_rsi, fast_smooth)
        ema_slow = ema(raw_rsi, slow_smooth)
        position = raw_rsi - ema_slow
        velocity = ema_fast - ema_slow
        acceleration = velocity - ema(velocity, signal_window)

        intermediates[name] = {"raw_rsi": raw_rsi, "ema_fast": ema_fast, "ema_slow": ema_slow,
                                "rsi_window": rsi_w}
        per_window[name] = {"raw": raw_rsi, "position": position, "velocity": velocity,
                             "acceleration": acceleration}

    features = {}
    for name, d in per_window.items():
        features[f"momentum_{name}_raw_{oscillator_name}"] = d["raw"]  # raw value per window -- let MP/ablation decide
        features[f"momentum_{name}_position"] = d["position"]
        features[f"momentum_{name}_velocity"] = d["velocity"]
        features[f"momentum_{name}_acceleration"] = d["acceleration"]

    # cross-WINDOW contrasts (the real short-vs-long-momentum signal): every pair of
    # native RSI windows, for raw/position/velocity/acceleration independently
    window_names = [w[0] for w in rsi_windows]
    for i in range(len(window_names)):
        for j in range(i + 1, len(window_names)):
            a, b = window_names[i], window_names[j]
            for quantity in ["raw", "position", "velocity", "acceleration"]:
                features[f"momentum_contrast_{quantity}_{a}_minus_{b}"] = (
                    per_window[a][quantity] - per_window[b][quantity]
                )

    return pd.DataFrame(features), intermediates


# =============================================================================
# VISUALS
# =============================================================================

def plot_trend_construction(dates, close, features, intermediates):
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)

    # panel 1: price + both scales' EMAs
    ax = axes[0]
    ax.plot(dates, close, color="black", lw=0.8, label="close")
    colors = {"short": "steelblue", "long": "crimson"}
    for name, interm in intermediates.items():
        c = colors.get(name, None)
        ax.plot(dates, interm["ema_fast"], color=c, lw=1.2, ls="--",
                 label=f"{name} ema_fast ({interm['fast_w']})")
        ax.plot(dates, interm["ema_slow"], color=c, lw=1.6,
                 label=f"{name} ema_slow ({interm['slow_w']})")
    ax.set_title("TREND: price with short-scale and long-scale EMA pairs")
    ax.legend(fontsize=8, ncol=3)

    # panel 2: position, both scales + contrast
    ax = axes[1]
    for name in intermediates:
        ax.plot(dates, features[f"trend_{name}_position"], color=colors.get(name), lw=1.3, label=f"{name} position")
    ax.plot(dates, features["trend_contrast_position_short_minus_long"], color="darkorange", lw=1.5,
             label="contrast (short - long)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title("Position = log(close / ema_slow) -- per scale, and the contrast between scales")
    ax.legend(fontsize=8)

    # panel 3: velocity, both scales + contrast
    ax = axes[2]
    for name in intermediates:
        ax.plot(dates, features[f"trend_{name}_velocity"], color=colors.get(name), lw=1.3, label=f"{name} velocity")
    ax.plot(dates, features["trend_contrast_velocity_short_minus_long"], color="darkorange", lw=1.5,
             label="contrast (short - long)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title("Velocity = log(ema_fast / ema_slow) -- per scale, and the contrast between scales\n"
                 "(this is the number that answers 'is short-term trend outrunning long-term trend')")
    ax.legend(fontsize=8)

    # panel 4: acceleration, both scales + contrast
    ax = axes[3]
    for name in intermediates:
        ax.plot(dates, features[f"trend_{name}_acceleration"], color=colors.get(name), lw=1.3,
                 label=f"{name} acceleration")
    ax.plot(dates, features["trend_contrast_acceleration_short_minus_long"], color="darkorange", lw=1.5,
             label="contrast (short - long)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title("Acceleration = velocity - EMA(velocity, signal_window)  [MACD-histogram style]")
    ax.legend(fontsize=8)
    ax.set_xlabel("date")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_trend_feature_construction.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '1_trend_feature_construction.png'}")


def plot_momentum_construction(dates, features, intermediates, oscillator_name="rsi"):
    fig, axes = plt.subplots(5, 1, figsize=(25, 18), sharex=True)
    colors = {"short": "steelblue", "long": "crimson"}

    # panel 1: the two NATIVE-WINDOW raw RSI series + their own smoothing EMAs
    ax = axes[0]
    for name, interm in intermediates.items():
        c = colors.get(name, None)
        ax.plot(dates, interm["raw_rsi"], color=c, lw=1.0,
                 label=f"{name}: raw {oscillator_name}-{interm['rsi_window']}")
        ax.plot(dates, interm["ema_slow"], color=c, lw=1.6, ls="--",
                 label=f"{name}: smoothing ema_slow")
    ax.set_title(f"MOMENTUM: {oscillator_name} computed at TWO NATIVE WINDOWS directly (14 vs. 28)\n"
                 f"(this is the fix -- varying the oscillator's own lookback, not smoothing a fixed one)")
    ax.legend(fontsize=8, ncol=2)

    # panel 2: raw RSI contrast (the headline short-vs-long momentum signal)
    ax = axes[1]
    for name, interm in intermediates.items():
        ax.plot(dates, interm["raw_rsi"], color=colors.get(name), lw=1.0, alpha=0.6,
                 label=f"{name} raw {oscillator_name}-{interm['rsi_window']}")
    ax.plot(dates, features["momentum_contrast_raw_short_minus_long"], color="darkorange", lw=1.6,
             label="contrast (RSI-short minus RSI-long)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title("Raw window contrast = RSI-short - RSI-long -- the direct short-vs-long momentum signal")
    ax.legend(fontsize=8)

    ax = axes[2]
    for name in intermediates:
        ax.plot(dates, features[f"momentum_{name}_position"], color=colors.get(name), lw=1.3, label=f"{name} position")
    ax.plot(dates, features["momentum_contrast_position_short_minus_long"], color="darkorange", lw=1.5,
             label="contrast (short - long)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title("Position = raw RSI(window) - its own smoothing ema_slow -- per window, and the contrast")
    ax.legend(fontsize=8)

    ax = axes[3]
    for name in intermediates:
        ax.plot(dates, features[f"momentum_{name}_velocity"], color=colors.get(name), lw=1.3, label=f"{name} velocity")
    ax.plot(dates, features["momentum_contrast_velocity_short_minus_long"], color="darkorange", lw=1.5,
             label="contrast (short - long)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title("Velocity = ema_fast(RSI_window) - ema_slow(RSI_window) -- per window, and the contrast\n"
                 "(temporal dynamics of momentum AT that window -- separate axis from the window contrast above)")
    ax.legend(fontsize=8)

    ax = axes[4]
    for name in intermediates:
        ax.plot(dates, features[f"momentum_{name}_acceleration"], color=colors.get(name), lw=1.3,
                 label=f"{name} acceleration")
    ax.plot(dates, features["momentum_contrast_acceleration_short_minus_long"], color="darkorange", lw=1.5,
             label="contrast (short - long)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title("Acceleration = velocity - EMA(velocity, signal_window)  [MACD-histogram style]")
    ax.legend(fontsize=8)
    ax.set_xlabel("date")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_momentum_feature_construction.png", dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / '2_momentum_feature_construction.png'}")


def plot_quick_correlation_check(trend_features, momentum_features):
    """
    Not the full MP/clustering pipeline -- just a fast raw-correlation sanity check
    so you can see, before running the full diagnostic script, whether the contrast
    columns visually look distinct from the level columns.
    """
    combined = pd.concat([trend_features, momentum_features], axis=1).dropna()
    corr = combined.corr()

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


# =============================================================================
# SYNTHETIC DEMO DATA (replace with your real close-price series)
# =============================================================================

def build_synthetic_price_series(n=800, seed=3):
    """
    A single price series with distinct regime changes (uptrend -> chop -> downtrend
    -> uptrend) so short-vs-long scale divergence is visually obvious in the plots.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    regime_len = n // 4
    drifts = [0.0015, 0.0000, -0.0012, 0.0018]
    log_returns = []
    for drift in drifts:
        log_returns.append(rng.normal(drift, 0.012, regime_len))
    log_returns = np.concatenate(log_returns)
    log_returns = np.concatenate([log_returns, rng.normal(0, 0.012, n - len(log_returns))])

    log_price = np.cumsum(log_returns) + np.log(100)
    close = pd.Series(np.exp(log_price), index=dates, name="close")
    return dates, close


# =============================================================================
# MAIN
# =============================================================================

def main():
    from findata import get_all_tickers, get_price_data

    ticker = get_all_tickers()[0]
    data = get_price_data(ticker).loc[ticker]
    close = data["close"]
    dates = pd.Series(close.index)

    trend_features, trend_intermediates = build_trend_features(close)
    momentum_features, momentum_intermediates = build_momentum_features(close, oscillator_name="rsi")

    print("Trend feature columns:")
    for c in trend_features.columns:
        print(f"  {c}")
    print("\nOscillators feature columns:")
    for c in momentum_features.columns:
        print(f"  {c}")

    def tail(data, n=500):
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.tail(n)
        if isinstance(data, (list, tuple)):
            return [tail(d, n) for d in data]
        if isinstance(data, dict):
            return {k: tail(v, n=n) for k, v in data.items()}
        return data

    plot_trend_construction(*tail([dates, close, trend_features, trend_intermediates]))
    plot_momentum_construction(*tail([dates, momentum_features, momentum_intermediates]), oscillator_name="rsi")
    plot_quick_correlation_check(trend_features, momentum_features)

    # write out the combined engineered feature set so you can plug it straight
    # into full_feature_taxonomy_analysis.py (add 'date'/'ticker' columns for your real panel)
    combined = pd.concat([trend_features, momentum_features], axis=1)
    combined.insert(0, "date", dates)
    out_path = OUTPUT_DIR / "engineered_trend_momentum_features.parquet"
    combined.to_parquet(out_path)
    print(f"\nSaved engineered feature set: {out_path}")
    print(f"All plots saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


