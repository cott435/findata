from findata.preprocess.v1.base import *


def fe_unbounded_momentum(df, ema_windows=None, feature_set='med', base_name='ema_close'):
    """
    Uses log ratio velocity — appropriate for price EMAs (always positive).
    """
    assert feature_set in ['low', 'med', 'high']
    if ema_windows is None:
        ema_windows = [12, 26, 52] if feature_set == 'low' else [6, 12, 26, 52, 104]
        if feature_set == 'high':
            ema_windows.append(208)
    out = pd.DataFrame(index=df.index)
    emas=[]
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


class Trend(PCAProcessor):

    def __init__(self, data, dates=None, n_components=0.95, scaler='robust', arcsinh=True, verbose=False,
                 feature_set='med', whiten_final=True, final_pca=True, final_n_components=0.95, state=None):
        pca_groups = {g: [g] for g in ['rel', 'vel', 'acc', 'di']}
        super(Trend, self).__init__(data, dates, n_components=n_components, scaler=scaler, arcsinh=arcsinh,
                                    verbose=verbose, feature_set=feature_set, pca_groups=pca_groups,
                                    whiten_final=whiten_final, final_pca=final_pca,
                                    final_n_components=final_n_components, state=state)

    def _feature_engineer(self):
        processed_data = (
            self.raw_data.groupby('ticker', group_keys=False)
            .apply(fe_unbounded_momentum)
        )
        self.feat_eng_data = pd.concat([self.feat_eng_data, processed_data], axis=1, join='inner')
        self.feat_eng_data['adx'] = get_column(self.raw_data, 'adx')
        self.feat_eng_data['plus_di'] = get_column(self.raw_data, 'plus_di')
        self.feat_eng_data['minus_di'] = get_column(self.raw_data, 'minus_di')

    def plot_fe(self, ticker=None, tail=None):
        tail = tail or self.tail
        raw = self.raw_data.loc[ticker or self.ticker].tail(tail)
        fe = self.feat_eng_data.loc[ticker or self.ticker].tail(tail)
        plot_dfs(
            raw[['close', 'ema_close_12', 'ema_close_26', 'ema_close_52']],
            fe[get_columns(fe, 'vel')],
            fe[get_columns(fe, 'acc')],
            fe[['adx', 'plus_di', 'minus_di']],
            line={2: 0, 1: 0},
            dark=False
        )


if __name__ == '__main__':
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:10]
    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    processor = Trend(data, DataSplits(), verbose=True, scaler='robust', arcsinh=True)
    processor.plot_corr()
    processor.plot_pca_corr()
    processor.plot_fe()
    processor.plot_scale_compare()
