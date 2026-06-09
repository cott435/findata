import pandas as pd
import numpy as np

BASE_CLOSE_EMA = [6, 12, 26, 52, 104]
BASE_OBV_AD_EMA = [12, 26, 52]
INDICATOR_WINDOWS = {
    'rsi': [14, 21],
    'bollinger': 20,
    'stochastic': 14,
    'adx': 14,
    'cci': 20,
    'willr': 14,
    'acroon': 14,
    'mfi' :20,
    'cmf': 20
}

class StockCalculator:

    def __init__(self, ohlc, current_data=None):
        self.ohlc = ohlc
        self.current_data = current_data
        self.update = current_data is not None

    def calculate_items(self):
        grouped = self.ohlc.groupby(level=[0,1], group_keys=False)
        ind = grouped.apply(self.calculate_indicators)
        if self.ema_vol:
            self.ohlc['obv'] = ind['obv']
            self.ohlc['ad'] = ind['ad']
        ema = grouped.apply(self.calculate_emas)
        return ema, ind

    def _seeded_ewm(self, series, span=14, seed=None):
        if seed is not None:
            # prepend seed so calculation continues from last known value
            series = seed.combine_first(series[seed.index[0]:])
            ema = series.ewm(span=span, adjust=False).mean()
            return ema.iloc[1:]
        else:
            return series.ewm(span=span, adjust=False).mean()

    def calculate_emas(self, df):
        current = self.current_data.loc[(*df.index[0][:-1], slice(None)), :] if self.update else None
        ema_dict, obv_dict, ad_dict = {}, {}, {}
        for period in BASE_CLOSE_EMA:
            ema_dict[f'ema{period}'] = self._seeded_ewm(df['close'].copy(), span=period, seed=current[f'ema{period}'] if self.update else None)
        for period in BASE_OBV_AD_EMA:
            obv_dict[f'ema{period}'] = self._seeded_ewm(df['obv'].copy(), span=period,
                                                        seed=current[f'ema{period}'] if self.update else None)
            ad_dict[f'ema{period}'] = self._seeded_ewm(df['ad'].copy(), span=period,
                                                        seed=current[f'ema{period}'] if self.update else None)
        if self.ema_vol:
            return pd.concat([pd.DataFrame(dic, index=df.index).dropna(how='all') for dic in [ema_dict, obv_dict, ad_dict]],
                             keys=['price', 'obv', 'ad'], names=['base'])
        return pd.DataFrame(ema_dict, index=df.index).dropna(how='all')

    def calculate_indicators(self, df):
        current = self.current_data.loc[(*df.index[0][:-1], slice(None)), :] if self.update else None
        ind_dict = {}

        # RSI (14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        ind_dict['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20,2)
        rolling = df['close'].rolling(20)
        ind_dict['bb_middle'] = rolling.mean()
        std = rolling.std()
        ind_dict['bb_upper'] = ind_dict['bb_middle'] + 2 * std
        ind_dict['bb_lower'] = ind_dict['bb_middle'] - 2 * std

        # Stochastic (14,3)
        low14 = df['low'].rolling(14).min()
        high14 = df['high'].rolling(14).max()
        ind_dict['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14)
        ind_dict['stoch_d'] = ind_dict['stoch_k'].rolling(3).mean()

        # ADX (14)
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=df.index)
        minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=df.index)
        ind_dict['plus_dm'] = self._seeded_ewm(plus_dm, span=14, seed=current['plus_dm'] if self.update else None)
        ind_dict['minus_dm'] = self._seeded_ewm(minus_dm, span=14, seed=current['minus_dm'] if self.update else None)
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift()),
            'lc': abs(df['low'] - df['close'].shift())
        }).max(axis=1)
        ind_dict['atr'] = self._seeded_ewm(tr, span=14, seed=current['atr'] if self.update else None)
        ind_dict['plus_di'] = 100 * ind_dict['plus_dm'] / ind_dict['atr']
        ind_dict['minus_di'] = 100 * ind_dict['minus_dm'] / ind_dict['atr']
        dx = 100 * abs(ind_dict["plus_di"] - ind_dict["minus_di"]) / (ind_dict["plus_di"] + ind_dict["minus_di"])
        ind_dict['adx'] = self._seeded_ewm(dx, span=14, seed=current['adx'] if self.update else None)

        # CCI (20)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        ind_dict['cci'] = (tp - sma_tp) / (0.015 * mad)

        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0)
        if self.update:
            ind_dict['obv'] = current['obv'].combine_first(obv[current['obv'].index[0]:]).cumsum()
        else:
            ind_dict['obv'] = obv.cumsum()
            # A/D
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
        mfv = mfm * df['volume']
        if self.update:
            ind_dict["ad"] = current['ad'].combine_first(mfv[current['ad'].index[0]:]).cumsum()
        else:
            ind_dict["ad"] = mfv.cumsum()

            # Williams %R (14)
        ind_dict['willr'] = -100 * (high14 - df['close']) / (high14 - low14)

        # Acroon
        highs = df["high"].rolling(14)
        lows = df["low"].rolling(14)
        ind_dict['aroon_up'] = highs.apply(lambda x: 100 * (14 - np.argmax(x.values[::-1])) / 14, raw=False)
        ind_dict['aroon_down'] = lows.apply(lambda x: 100 * (14 - np.argmin(x.values[::-1])) / 14, raw=False)

        # MFI
        tp = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        direction = tp.diff()
        pmf = rmf.where(direction > 0, 0)
        nmf = rmf.where(direction < 0, 0)
        mfr = pmf.rolling(20).sum() / nmf.rolling(20).sum()
        mfi = 100 - (100 / (1 + mfr))
        ind_dict['mfi'] = mfi.fillna(50)

        # CMF
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([pd.NA, pd.NaT, np.nan, float('inf'), -float('inf')], 0)  # avoid division issues
        mfv = mfm * df['volume']
        ind_dict['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()

        return pd.DataFrame(ind_dict, index=df.index).dropna(how='any')

