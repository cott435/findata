from findata.preprocess.feature_building.v1.base import *

def compute_over_under_sold(df, levels):
    over = pd.DataFrame(index=df.index)
    under = pd.DataFrame(index=df.index)
    for ind, (l, h) in levels.items():
        under[f'{ind}'] = df[ind] < l
        over[f'{ind}'] = df[ind] > h
    out = pd.DataFrame(index=df.index)
    out['undersold'] = under.astype(int).sum(axis=1)
    out['oversold'] = over.astype(int).sum(axis=1)
    return out

class Momentum(PCAProcessor):
    """
    Features:
        1. RSI, CCI, WillR, BB %, StochK
        2. EMAs (4, 8, 16)
        3. Velocity (4 - 8) and (8 - 16)
        4. Acceleration
        5. Sum of overbought and oversold
    Scaling:
        1. Items bounded in reasonable range, standard scaling
    PCA:
        1. One large PCA for all
        2. Grouped PCA (Raw and EMA), Vel, Acc
    """

    def __init__(self, data, dates=None, n_components=0.95, scaler='standard', arcsinh=False, verbose=False,
                 feature_set='med', whiten_final=True, final_pca=True, final_n_components=0.95, state=None):
        self.levels = {'rsi': (30, 70), 'cci': (-100, 100), 'willr': (20, 80), 'stoch_k': (20, 80),
                       'bb_percent': (0.05, 0.95)}
        pca_groups = {'main': ['raw', 'ema'], 'vel': ['vel'], 'acc': ['acc']}
        super(Momentum, self).__init__(data, dates, n_components=n_components, scaler=scaler, arcsinh=arcsinh,
                                    verbose=verbose, feature_set=feature_set, pca_groups=pca_groups,
                                    whiten_final=whiten_final, final_pca=final_pca,
                                    final_n_components=final_n_components, state=state)

    def _feature_engineer(self):
        bb_range = get_column(self.raw_data, 'bb_upper') - get_column(self.raw_data, 'bb_lower')
        bb_range[bb_range == 0] = 1e-9
        self.raw_data['bb_percent'] = (self.raw_data['close'] - get_column(self.raw_data, 'bb_lower')) / bb_range

        def compute_rolling_indicators(df):
            inds =get_column_names(['rsi', 'willr', 'cci', 'stoch_k'])
            inds.append('bb_percent')
            out = fe_oscillator_momentum(df, inds, feature_set=self.feature_set)
            return out

        processed_data = (
            self.raw_data.groupby('ticker', group_keys=False)
            .apply(compute_rolling_indicators)
        )
        levels = {get_column_names(k): v for k, v in self.levels.items()}
        over_under_sold = compute_over_under_sold(self.raw_data, levels)
        self.feat_eng_data = pd.concat([self.feat_eng_data, processed_data, over_under_sold], axis=1, join='inner')

    def plot_fe(self, ticker=None, tail=None):
        tail = tail or self.tail
        raw = self.raw_data.loc[ticker or self.ticker].tail(tail)
        fe = self.feat_eng_data.loc[ticker or self.ticker].tail(tail)
        for ind, levels in self.levels.items():
            plots = [raw[['close', 'ema_close_12', 'ema_close_26']], fe[get_columns(fe, {ind: ['raw', 'ema']})], fe[get_columns(fe, {ind: 'vel'})]]
            if self.feature_set == 'high':
                plots.append(fe[get_columns(fe, {ind: 'acc'})])
            plot_dfs(*plots, line={1: levels})


if __name__ == '__main__':
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers
    tickers = get_all_tickers()[:10]
    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    processor = Momentum(data, DataSplits(), verbose=True, scaler='standard', n_components=0.96, feature_set='high')
    processor.plot_fe()
    processor.plot_corr(col_keys=['rsi', 'cci'])
    processor.plot_pca_corr()
    processor.plot_scale_compare(['rsi', 'cci'])

