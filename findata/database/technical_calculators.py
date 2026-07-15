"""Derived metric calculators for price data.

Stateless module-level functions -- one per indicator -- so each can be called
directly with whatever period(s) are wanted. Every function takes a
single-(interval, ticker) OHLCV frame indexed by date (columns open, high,
low, close, volume) and returns a Series or DataFrame aligned to that index.

Seeding
-------
EWM-based items (the EMAs, atr, plus_dm, minus_dm, adx) and the cumulative
items (obv, ad) accept an optional ``seed``: a one-row Series holding the last
stored value, indexed by its date. With a seed the calculation continues from
that value over the bars *after* the seed date instead of recomputing the full
history. Cold start uses ewm(adjust=True); a seeded continuation uses
ewm(adjust=False), so it converges to the cold-start values exactly once the
history behind the seed is long enough.

The input frame must contain the seed date's bar (the seeded output starts on
the bar after it). Rolling-window indicators (rsi, bollinger, ...) need no
seed -- just include enough trailing bars to warm their window and the values
for the new dates are exact.
"""

import numpy as np
import pandas as pd

EMA_WINDOWS = {
    'ema_close': [6, 12, 26, 52, 104],
    'ema_obv': [12, 26, 52],
    'ema_ad': [12, 26, 52],
}

INDICATOR_WINDOWS = {
    'rsi': [14, 21],
    'bollinger': 20,
    'stochastic': 14,
    'adx': 14,
    'cci': 20,
    'willr': 14,
    'aroon': 14,
    'mfi': 20,
    'cmf': 20,
}

# indicator columns whose incremental update needs an explicit seed value
# (everything else is either rolling-window or an ema_* item)
SEEDED_ITEMS = ('obv', 'ad', 'atr', 'plus_dm', 'minus_dm', 'adx')


def _as_list(periods):
    return list(periods) if isinstance(periods, (list, tuple, set)) else [periods]


def _seeded_ewm(series: pd.Series, span: int, seed: pd.Series = None) -> pd.Series:
    """EWM mean that can continue from a stored value.

    Cold start (seed=None) runs ewm(adjust=True) over the whole series.
    Seeded, the seed value is prepended and ewm(adjust=False) runs over the
    bars after the seed date; the seed row is dropped from the output.
    """
    if seed is None:
        return series.ewm(span=span, adjust=True).mean()
    seed = seed.dropna().iloc[[-1]]
    tail = series[series.index > seed.index[0]]
    return pd.concat([seed, tail]).ewm(span=span, adjust=False).mean().iloc[1:]


def _seeded_cumsum(flow: pd.Series, seed: pd.Series = None) -> pd.Series:
    """Cumulative sum that can continue from a stored running total."""
    if seed is None:
        return flow.cumsum()
    seed = seed.dropna().iloc[[-1]]
    tail = flow[flow.index > seed.index[0]]
    return pd.concat([seed, tail]).cumsum().iloc[1:]


def _rolling_mad(series: pd.Series, period: int) -> pd.Series:
    """Rolling mean-absolute-deviation about each window's own mean.

    Vectorized stand-in for rolling().apply(): one strided view over the array
    and array-wide reductions instead of a Python callback per window. Matches
    the native rolling contract (NaN for the leading period-1 bars and for any
    window containing a gap, since the mean then propagates NaN).
    """
    arr = series.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    if arr.size >= period:
        windows = np.lib.stride_tricks.sliding_window_view(arr, period)
        out[period - 1:] = np.abs(windows - windows.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(out, index=series.index)


def _bars_since_extreme(series: pd.Series, period: int, argfunc) -> np.ndarray:
    """Bars since each window's most recent max/min (0 = current bar).

    Vectorized stand-in for rolling().apply(argmax/argmin on the reversed
    window). NaN-masked to match the native rolling contract (argmax/argmin
    would otherwise treat a NaN as the extreme rather than yield NaN).
    """
    arr = series.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    if arr.size >= period:
        windows = np.lib.stride_tricks.sliding_window_view(arr, period)[:, ::-1]
        since = argfunc(windows, axis=1).astype(float)
        since[np.isnan(windows).any(axis=1)] = np.nan
        out[period - 1:] = since
    return out


def ema(series: pd.Series, period: int, seed: pd.Series = None) -> pd.Series:
    """EMA of an arbitrary series (close, obv, ad, ...)."""
    return _seeded_ewm(series, span=period, seed=seed)


def rsi(data: pd.DataFrame|pd.Series, period: int = 14) -> pd.Series:
    data = data['close'] if isinstance(data, pd.DataFrame) else data
    delta = data.diff()
    avg_gain = delta.clip(lower=0).rolling(period).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).rename('rsi')


def bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    rolling = df['close'].rolling(period)
    middle = rolling.mean()
    std = rolling.std()
    return pd.DataFrame({
        'bb_middle': middle,
        'bb_upper': middle + num_std * std,
        'bb_lower': middle - num_std * std,
    })


def stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low_n = df['low'].rolling(period).min()
    high_n = df['high'].rolling(period).max()
    return 100 * (df['close'] - low_n) / (high_n - low_n)


def atr(df: pd.DataFrame, period: int = 14, seed: pd.Series = None) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return _seeded_ewm(tr, span=period, seed=seed).rename('atr')


def adx(df: pd.DataFrame, period: int = 14, seeds: dict = None) -> pd.DataFrame:
    """Directional movement family: plus_dm, minus_dm, atr, plus_di, minus_di, adx.

    ``seeds`` maps the seeded member names ('plus_dm', 'minus_dm', 'atr',
    'adx') to their one-row seed Series. Provide all of them or none.
    """
    seeds = seeds or {}
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    plus = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0), index=df.index)
    minus = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0), index=df.index)

    out = pd.DataFrame({
        'plus_dm': _seeded_ewm(plus, span=period, seed=seeds.get('plus_dm')),
        'minus_dm': _seeded_ewm(minus, span=period, seed=seeds.get('minus_dm')),
        'atr': atr(df, period=period, seed=seeds.get('atr')),
    })
    out['plus_di'] = 100 * out['plus_dm'] / out['atr']
    out['minus_di'] = 100 * out['minus_dm'] / out['atr']
    dx = 100 * (out['plus_di'] - out['minus_di']).abs() / (out['plus_di'] + out['minus_di'])
    out['adx'] = _seeded_ewm(dx, span=period, seed=seeds.get('adx'))
    return out


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period).mean()
    mad = _rolling_mad(tp, period)
    return ((tp - sma) / (0.015 * mad)).rename('cci')


def obv(df: pd.DataFrame, seed: pd.Series = None) -> pd.Series:
    flow = (np.sign(df['close'].diff()) * df['volume']).fillna(0)
    return _seeded_cumsum(flow, seed=seed).rename('obv')


def ad(df: pd.DataFrame, seed: pd.Series = None) -> pd.Series:
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    return _seeded_cumsum(mfm * df['volume'], seed=seed).rename('ad')


def willr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_n = df['high'].rolling(period).max()
    low_n = df['low'].rolling(period).min()
    return (-100 * (high_n - df['close']) / (high_n - low_n)).rename('willr')


def aroon(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    since_high = _bars_since_extreme(df['high'], period, np.argmax)
    since_low = _bars_since_extreme(df['low'], period, np.argmin)
    return pd.DataFrame({'aroon_up': 100 * (period - since_high) / period,
                         'aroon_down': 100 * (period - since_low) / period},
                        index=df.index)


def mfi(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    direction = tp.diff()
    pos = rmf.where(direction > 0, 0).rolling(period).sum()
    neg = rmf.where(direction < 0, 0).rolling(period).sum()
    return (100 - 100 / (1 + pos / neg)).rename('mfi')


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * df['volume']
    return (mfv.rolling(period).sum() / df['volume'].rolling(period).sum()).rename('cmf')


# dispatch tables used by calculate_all and the db manager
INDICATOR_FUNCS = {
    'rsi': rsi,
    'bollinger': bollinger,
    'stochastic': stochastic,
    'adx': adx,
    'cci': cci,
    'willr': willr,
    'aroon': aroon,
    'mfi': mfi,
    'cmf': cmf,
}
INDICATOR_OUTPUTS = {
    'rsi': ['rsi'],
    'bollinger': ['bb_middle', 'bb_upper', 'bb_lower'],
    'stochastic': ['stochastic'],
    'adx': ['plus_dm', 'minus_dm', 'atr', 'plus_di', 'minus_di', 'adx'],
    'cci': ['cci'],
    'willr': ['willr'],
    'aroon': ['aroon_up', 'aroon_down'],
    'mfi': ['mfi'],
    'cmf': ['cmf'],
}


def calculate_all(ohlcv: pd.DataFrame, seeds: dict = None, **overrides) -> pd.DataFrame:
    """Compute every EMA and indicator in one frame.

    ``ohlcv`` is either a single-(interval, ticker) frame indexed by date, or
    the multi-ticker frame from yf.py indexed by (interval, ticker, date) --
    the latter is grouped and calculated per (interval, ticker).

    Periods come from EMA_WINDOWS / INDICATOR_WINDOWS and any can be
    overridden by keyword with an int or a list of ints, e.g.
    ``calculate_all(df, rsi=[14, 25], cci=30, ema_close=21)``.

    ``seeds`` maps (item, period) column keys to one-row seed Series (see
    build_seeds). When seeding, supply seeds for *all* seedable items so the
    EWM/cumulative columns continue rather than restart. For the grouped
    input, key ``seeds`` by the (interval, ticker) tuple instead.

    Returns a frame with MultiIndex columns (item, period); obv and ad use
    period 0.
    """
    if isinstance(ohlcv.index, pd.MultiIndex):
        group_levels = list(range(ohlcv.index.nlevels - 1))
        seeds = seeds or {}

        def _per_group(group):
            return _calculate_all(group.droplevel(group_levels), seeds.get(group.name), overrides)

        return ohlcv.groupby(level=group_levels, group_keys=True).apply(_per_group)
    return _calculate_all(ohlcv, seeds, overrides)


def _calculate_all(df: pd.DataFrame, seeds: dict, overrides: dict) -> pd.DataFrame:
    seeds = seeds or {}
    windows = {**EMA_WINDOWS, **INDICATOR_WINDOWS}
    unknown = set(overrides) - set(windows)
    if unknown:
        raise KeyError(f'Unknown window override(s): {sorted(unknown)}')
    windows.update(overrides)

    out = {
        ('obv', 0): obv(df, seed=seeds.get(('obv', 0))),
        ('ad', 0): ad(df, seed=seeds.get(('ad', 0))),
    }

    for name, func in INDICATOR_FUNCS.items():
        for period in _as_list(windows[name]):
            if name == 'adx':
                adx_seeds = {item: seeds[(item, period)] for item in ('plus_dm', 'minus_dm', 'atr', 'adx')
                             if (item, period) in seeds}
                res = func(df, period=period, seeds=adx_seeds)
            else:
                res = func(df, period=period)
            if isinstance(res, pd.Series):
                out[(res.name, period)] = res
            else:
                for col in res.columns:
                    out[(col, period)] = res[col]

    bases = {'close': df['close'], 'obv': out[('obv', 0)], 'ad': out[('ad', 0)]}
    for base, series in bases.items():
        for period in _as_list(windows[f'ema_{base}']):
            out[(f'ema_{base}', period)] = ema(series, period, seed=seeds.get((f'ema_{base}', period)))

    result = pd.concat(out, axis=1)
    result.columns = result.columns.set_names(['item', 'period'])
    return result.sort_index(axis=1)


def build_seeds(calc_df: pd.DataFrame) -> dict:
    """Extract seeds from a calculate_all result (single ticker, date index).

    Takes the last valid value of every seedable column and returns the
    {(item, period): one-row Series} dict that calculate_all expects.
    """
    seeds = {}
    for col in calc_df.columns:
        item, _ = col
        if item in SEEDED_ITEMS or item.startswith('ema_'):
            series = calc_df[col].dropna()
            if not series.empty:
                seeds[col] = series.iloc[[-1]]
    return seeds


if __name__ == '__main__':
    # Seeding efficacy check: cold-start the full AAPL daily history, then
    # seed from the last bar of 2019 and recompute 2020-onward only. The two
    # results should agree to numerical noise.
    from yf import YahooFinance

    CUTOFF = pd.Timestamp('2020-01-01').date()
    WARMUP_BARS = 150  # trailing bars so rolling-window indicators are exact

    print('Pulling full daily history for AAPL...')
    data = YahooFinance('AAPL').request_ticker_financials()
    prices = data['prices'].xs(('daily', 'AAPL'))

    full = calculate_all(prices)
    print(f'Cold-start: {full.shape[0]} rows x {full.shape[1]} columns '
          f'({prices.index[0]} -> {prices.index[-1]})')

    history = full[full.index < CUTOFF]
    seeds = build_seeds(history)
    seed_date = history.index[-1]
    seed_pos = prices.index.get_loc(seed_date)
    window = prices.iloc[max(seed_pos - WARMUP_BARS, 0):]

    seeded = calculate_all(window, seeds=seeds)
    seeded = seeded[seeded.index >= CUTOFF]
    print(f'Seeded from {seed_date}: recomputed {seeded.shape[0]} rows '
          f'({len(seeds)} seed values, {WARMUP_BARS} warmup bars)')

    expected = full[full.index >= CUTOFF]
    diff = (expected - seeded).abs()
    summary = pd.DataFrame({
        'max_abs_diff': diff.max(),
        'mean_abs_diff': diff.mean(),
        'last_full': expected.iloc[-1],
        'last_seeded': seeded.iloc[-1],
    }).sort_values('max_abs_diff', ascending=False)

    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.float_format', '{:,.6g}'.format):
        print('\nFull-history vs seeded recomputation (2020 onward):')
        print(summary)
