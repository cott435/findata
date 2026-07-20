from findata.preprocess.feature_building.v1.base import *

class Candle(PCAProcessor):
    """
    Features:
        1. body: log(C/O)
        2. range: log(H/L)
        3. skew: log(H/max(O, C)) - log(min(O, C)/L)
        4. gap: log(O/C.shift(1))
        5. body ratio: (C-O)/(H-L)
    Scaling:
        1. log features have heavy outliers; robust scaling preferred with arcsinh transform
        2. range is bounded positive and skewed left, prefer power scaling
        3. body ratio scaled to ∈[-1, 1]
    PCA:
        1. No PCA needed, features not correlated
    """

    def __init__(self, data, dates=None, n_components=None, scaler='robust', arcsinh=True, verbose=False,
                 feature_set='med', whiten_final=True, state=None):
        pca_groups = {}
        super(Candle, self).__init__(data, dates, n_components=n_components, scaler=scaler, arcsinh=arcsinh,
                                     verbose=verbose, feature_set=feature_set, pca_groups=pca_groups,
                                     whiten_final=whiten_final, final_pca=False, state=state)

    def _feature_engineer(self):
        df = self.raw_data.copy()
        O, H, L, C = (
            df["open"],
            df["high"],
            df["low"],
            df["close"],
        )
        R = (H - L) + self.eps
        body = C - O
        upper_wick = H - np.maximum(O, C)
        lower_wick = np.minimum(O, C) - L

        self.candle_ratios = pd.DataFrame({
            "body_ratio": body / R,
            "upper_ratio": upper_wick / R,
            "lower_ratio": lower_wick / R,
        })

        processed_data = pd.DataFrame(
            {
                'body_size': (body / R).abs(),

                # Size
                "body_log": np.log(C / O),
                "range_log": np.log(H/L),
                'skew_log': np.log(H / np.maximum(O, C)) - np.log(np.minimum(O, C) / L),
                'gap_log': np.log(O / C.shift(1))
            },
            index=df.index,
        )
        self.feat_eng_data = pd.concat([self.feat_eng_data, processed_data], axis=1, join='inner')

    def _scale(self):
        """will not scaled bounded components (candle_size based ones)"""
        scaling_columns = [c for c in self.feat_eng_data.columns if 'log' in c]
        self._fit_apply_scaler(scaling_columns)
        non_scaling_columns = [c for c in self.feat_eng_data.columns if 'log' not in c]
        self.scaled_data[non_scaling_columns] = self.feat_eng_data[non_scaling_columns] * 2 -1


    def plot_fe(self, n=3):
        idx = np.random.choice(self.feat_eng_data.index, size=min(n, len(self.feat_eng_data)), replace=False)
        for i in idx:
            fig, axes = plt.subplots(
                2, 1, figsize=(7, 8)
            )
            O, H, L, C = self.raw_data.loc[i, ["open", "high", "low", "close"]]
            upper, mid, lower = self.candle_ratios.loc[i, ["upper_ratio", "body_ratio", "lower_ratio"]]
            color = "green" if C >= O else "red"
            axes[0].plot([0, 0], [L, H], color="black")
            axes[0].bar(
                0,
                C - O,
                bottom=O,
                color=color,
                width=0.3,
                edgecolor="black",
            )
            axes[0].set_xlim(-0.3,0.3)
            d = (H-L)*0.1
            axes[0].set_ylim(L -d, H +d)
            for bot, top, val in ((O, C, mid), (H, max(O, C), upper), (L, min(O, C), lower)):
                axes[0].annotate(
                    "",  # no text for the bracket itself
                    xy=(0.2, top),  # top of bracket
                    xytext=(0.2, bot),  # bottom of bracket
                    arrowprops=dict(
                        arrowstyle="|-|",  # bracket
                        lw=1.5,
                        color='black'
                    )
                )
                axes[0].text(
                    0.28,
                    (top + bot) / 2,
                    f"{val:.2f}",
                    ha="right",
                    va="center",
                    color='black',
                    fontsize=9
                )

            axes[0].set_title(f"Candle @ index {i}")
            axes[0].set_xticks([])
            axes[0].set_ylabel("Price")
            fvals = self.feat_eng_data.loc[i, [c for c in self.feat_eng_data.columns if 'log' in c]]
            axes[1].barh(fvals.index, fvals.values)
            axes[1].axvline(0, color="black", lw=0.8)
            axes[1].set_title("Log Features")
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    from findata.configs import DataSplits
    from findata import get_all_data, get_all_tickers

    tickers = get_all_tickers()[:10]
    dates = DataSplits()
    data, ticker_info = get_all_data(tickers, dates.data_start, dates.data_end)
    processor = Candle(data, DataSplits(), verbose=True, n_components=0.96)
    processor.plot_corr()
    processor.plot_fe()
    processor.plot_scale_compare()




