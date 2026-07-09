from findata.preprocess.base import *
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering

def find_reversals(data, val_col='close', distance=3, prominence=0.05):
    value = data[val_col]
    roll = value.rolling(50)
    log_value = np.array((value-roll.mean())/roll.std()) if val_col in ['obv', 'ad'] else np.log(np.array(value))
    top_peaks = find_peaks(log_value, distance=distance, prominence=prominence)[0]
    bottom_peaks = find_peaks(-log_value, distance=distance, prominence=prominence)[0]
    rows = []
    for i in top_peaks:
        rows.append({
            'ticker': data.index.get_level_values(0)[0],
            'date': data.index.get_level_values(1)[i],
            'value': value.iloc[i],
            'log_value': log_value[i],
            'type': 'top',
            'base': val_col
        })
    for i in bottom_peaks:
        rows.append({
            'ticker': data.index.get_level_values(0)[0],
            'date': data.index.get_level_values(1)[i],
            'value': value.iloc[i],
            'log_value': log_value[i],
            'type': 'bottom',
            'base': val_col
        })
    return pd.DataFrame(rows).set_index(['ticker', 'base', 'date'])

def cluster_peaks(peak_values, linkage='average', distance_threshold=0.05):
    ticker = peak_values.index.get_level_values(0)[0]
    peak_values = np.array(peak_values)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        linkage=linkage,
        distance_threshold=distance_threshold
    ).fit(peak_values.reshape(-1, 1))
    levels = []
    for cluster_id in set(clustering.labels_):
        cluster_prices = peak_values[clustering.labels_ == cluster_id]
        level_price = cluster_prices.mean()
        strength = len(cluster_prices)
        levels.append((level_price, strength))
    rows = []
    for level, n in levels:
        if n>2:
            rows.append({
                'ticker': ticker,
                'level': np.exp(level),
                'log_level': level,
                'n': n
            })
    return pd.DataFrame(rows).set_index(['ticker'])

class SupportResistance(PCAProcessor):

    def __init__(self, data, dates, n_components=None, scaler='standard', arcsinh=False, verbose=False,
                 feature_set='med'):
        pca_groups = None
        super(SupportResistance, self).__init__(data, dates, n_components=n_components, scaler=scaler, arcsinh=arcsinh,
                                     verbose=verbose, feature_set=feature_set, pca_groups=pca_groups)

    def _feature_engineer(self):

        def get_all_sr(df):
            price = find_reversals(df)
            obv = find_reversals(df, val_col='obv', prominence=1)
            ad = find_reversals(df, val_col='ad', prominence=1)
            return pd.concat([price, obv, ad], axis=0).sort_index(level=["ticker", "date", "base"])

        self.pivot_points = (
            self.raw_data.groupby('ticker', group_keys=False)
            .apply(get_all_sr)
        )
        # TODO add a density sum which will be the log(price / support) and set to 0 if more than 10% away
        #  could use these calculated lines or just rolling min and max

        self.levels = self.pivot_points.xs('close', level=1, axis=0)['log_value'].groupby(['ticker'], group_keys=False).apply(cluster_peaks)


    def plot_fe(self):
        plt.figure(figsize=(12, 8))
        plt.plot()

    def plot_support_resistance(self, ticker=None):
        plt.figure(figsize=(12, 8))
        plt.plot(self.raw_data.loc[ticker]['close'], label='price', color='black')
        pivot_points = self.pivot_points.loc[ticker, 'close']
        top=pivot_points[pivot_points['type']=='top']
        bottom=pivot_points[pivot_points['type']=='bottom']
        plt.scatter(top.index, top['value'], label='Top', color='lime', marker='^', s=80, edgecolor='black')
        plt.scatter(bottom.index, bottom['value'], label='Bottom', color='orange', marker='v', s=80, edgecolor='black')
        for _, row in self.levels.loc[ticker].iterrows():
            if row['n']>3:
                plt.axhline(row['level'], color='red', linestyle='--', linewidth=2)


if __name__ == '__main__':
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers
    tickers = get_all_tickers()[:10]
    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    processor = SupportResistance(data, dates, verbose=True, scaler='power', arcsinh=False)
    processor.plot_scale_compare()
    processor.plot_corr()
    processor.plot_fe()
    """
    check nearest top reversal and bottom reversal of price that the current is between (above bottom below top)
    check obv
    """
