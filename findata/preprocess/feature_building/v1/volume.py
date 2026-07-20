from findata.preprocess.feature_building.v1.base import *

def calc_z_score_vel(df, ema_windows):
    out = pd.DataFrame(index=df.index)
    for i in range(len(ema_windows) - 1):
        f, s = ema_windows[i], ema_windows[i + 1]

        price_vel = df[f'ema_close_{f}'] - df[f'ema_close_{s}']
        out[f'price_zvel{f}_{s}'] = (price_vel - price_vel.rolling(window=s*2).mean()) / price_vel.rolling(window=s*2).std()
        obv_vel = df[f'ema_obv_{f}'] - df[f'ema_obv_{s}']
        out[f'obv_zvel{f}_{s}'] = (obv_vel - obv_vel.rolling(window=s*2).mean()) / obv_vel.rolling(window=s*2).std()
        ad_vel = df[f'ema_ad_{f}'] - df[f'ema_ad_{s}']
        out[f'ad_zvel{f}_{s}'] = (ad_vel - ad_vel.rolling(window=s*2).mean()) / ad_vel.rolling(window=s*2).std()
    return out


class Volume(PCAProcessor):
    """
    Features:
        1. CMF, MFI, their EMAs, velocity, and acceleration
        2. Close, OBV and AD Z-Score Velocity
            PCA on above 3 z-score velocities account for their divergence and their sum
    Scaling:
        1. Items bounded in reasonable range, standard scaling
    PCA:
        1. One large PCA for all
        2. Grouped PCA (Raw and EMA), Vel, Acc
    """

    def __init__(self, data, dates=None, n_components=0.95, scaler='robust', arcsinh=False, verbose=False,
                 feature_set='med', whiten_final=True, final_pca=True, final_n_components=0.95, state=None):
        self.ema_windows = [12, 26, 52]
        vel_pca = {f'z_vel{i+1}': [f'zvel{self.ema_windows[i]}_{self.ema_windows[i+1]}'] for i in range(len(self.ema_windows) - 1)}
        pca_groups = {'mom_main': ['_raw', '_ema'], 'mom_vel': ['_vel'], **vel_pca}
        super(Volume, self).__init__(data, dates, n_components=n_components, scaler=scaler, arcsinh=arcsinh,
                                     verbose=verbose, feature_set=feature_set, pca_groups=pca_groups,
                                     whiten_final=whiten_final, final_pca=final_pca,
                                     final_n_components=final_n_components, state=state)


    def _feature_engineer(self):
        def apply_fe(df):
            cols = get_column_names(['mfi', 'cmf'])
            return pd.concat([
                calc_z_score_vel(df, self.ema_windows),
                fe_oscillator_momentum(df, cols, feature_set=self.feature_set),
            ], axis=1)

        processed_data = (
            self.raw_data.groupby('ticker', group_keys=False)
            .apply(apply_fe)
        )
        self.feat_eng_data = pd.concat([self.feat_eng_data, processed_data], axis=1, join='inner')

    def _scale(self):
        scaling_columns = [c for c in self.feat_eng_data.columns if 'zvel' not in c]
        self._fit_apply_scaler(scaling_columns)
        pass_columns = [c for c in self.feat_eng_data.columns if c not in scaling_columns]
        self.scaled_data[pass_columns] = self.feat_eng_data[pass_columns]


    def plot_fe(self, ticker=None, tail=None, raw_vol=True):
        tail = tail or self.tail
        raw = self.raw_data.loc[ticker or self.ticker].tail(tail)
        fe = self.feat_eng_data.loc[ticker or self.ticker].tail(tail)
        plots = [raw[['close', 'ema12', 'ema26']]]
        if raw_vol:
            for vi in ['obv', 'ad']:
                plots.append(raw[[vi, f'{vi}_ema4', f'{vi}_ema12', f'{vi}_ema26']])
        plots.append(fe[get_columns(fe, f'{self.ema_windows[0]}_{self.ema_windows[1]}')])
        plots.append(fe[get_columns(fe, f'{self.ema_windows[1]}_{self.ema_windows[2]}')])
        plots.append(fe[get_columns(fe, f'vel{self.ema_windows[0]}_{self.ema_windows[1]}')])
        plot_dfs(*plots)

    def plot_raw(self, ticker=None, tail=None):
        tail = tail or self.tail
        raw = self.raw_data.loc[ticker or self.ticker].tail(tail)
        plots = [raw[['close', 'ema12', 'ema26']]]
        for vi in ['obv', 'ad']:
            plots.append(raw[[vi, f'{vi}_ema4', f'{vi}_ema12', f'{vi}_ema26']])
        plot_dfs(*plots)


if __name__ == '__main__':
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers
    tickers = get_all_tickers()[:10]

    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    processor = Volume(data, dates, verbose=True, scaler='robust', arcsinh=False)
    processor.plot_fe(ticker='BTC-USD', tail=800)
    processor.plot_corr()
    processor.plot_pca_corr()
    processor.plot_scale_compare()
    processor.plot_corr(transformed=True)
