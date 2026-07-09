from findata.preprocess.base import *

def fe_volatility(df, windows=None, feature_set='med'):
    assert feature_set in ['low', 'med', 'high']
    if windows is None:
        windows = [20, 100] if feature_set == 'low' else [7, 20, 100]
    out = pd.DataFrame(index=df.index)
    returns = df["close"].pct_change()
    for window in windows:
        out[f"vol{window}"] = returns.rolling(window).std()
        park = (np.log(df["high"] / df["low"]) ** 2) / (4 * np.log(2))
        out[f"vol_park{window}"] = np.sqrt(park.rolling(window).mean())
        rs_part = (
                np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
                np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
        )
        out[f"vol_rs{window}"] = np.sqrt(rs_part.rolling(window).mean())
        gk = (
                0.5 * (np.log(df["high"] / df["low"])) ** 2
                - (2 * np.log(2) - 1) * (np.log(df["close"] / df["open"])) ** 2
        )
        if feature_set == 'high':
            out[f"vol_gk{window}"] = np.sqrt(gk.rolling(window).mean())
            rv_proxy = ((df["high"] - df["low"]) / df["close"]) ** 2 \
                       + ((df["close"] - df["open"]) / df["close"]) ** 2
            out[f"vol_rvp{window}"] = np.sqrt(rv_proxy.rolling(window).mean())
    return out

class Volatility(PCAProcessor):
    """
    Features:
        1. BB Bandwidth
        2. ATR
        3. Volatility Calculations
    Scaling:
        1. power
    PCA:
        1. One PCA group, all features are highly correlated
    """

    def __init__(self, data, dates=None, n_components=None, scaler='power', arcsinh=False, verbose=False,
                 feature_set='med', whiten_final=True, final_pca=True, final_n_components=0.95, state=None):
        pca_groups = {}
        super(Volatility, self).__init__(data, dates, n_components=n_components, scaler=scaler, arcsinh=arcsinh,
                                    verbose=verbose, feature_set=feature_set, pca_groups=pca_groups,
                                    whiten_final=whiten_final, final_pca=final_pca,
                                    final_n_components=final_n_components, state=state)

    def _feature_engineer(self):

        def vol(df):
            return fe_volatility(df, feature_set=self.feature_set)
        processed_data = (
            self.raw_data.groupby('ticker', group_keys=False)
            .apply(vol)
        )
        self.feat_eng_data['bb_bandwidth'] = (get_column(self.raw_data, 'bb_upper') - get_column(self.raw_data, 'bb_lower')) / get_column(self.raw_data, 'bb_middle')
        self.feat_eng_data['atr_norm'] = get_column(self.raw_data, 'atr') / self.raw_data['close']
        self.feat_eng_data = pd.concat([self.feat_eng_data, processed_data], axis=1, join='inner')


    def plot_fe(self, ticker=None, tail=None):
        tail = tail or self.tail
        raw = self.raw_data.loc[ticker or self.ticker].tail(tail)
        scaled = self.scaled_data.loc[ticker or self.ticker].tail(tail)
        plot_dfs(
            raw[['close', 'ema_close_12', 'ema_close_26']],
            scaled[[c for c in scaled.columns if '7' in c]],
            scaled[[c for c in scaled.columns if '20' in c]]
        )


if __name__ == '__main__':
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:10]

    dates = DataSplits()
    tickers = ['ADBE', 'ADT', 'BRO', 'CEG', 'CFG', 'CL']
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    processor = Volatility(data, dates, verbose=True, scaler='power', n_components=0.96)
    processor.plot_corr()
    processor.plot_final_pca_corr()
    processor.plot_fe()
    processor.plot_scale_compare()
