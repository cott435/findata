from typing import Iterable, List

import numpy as np
import pandas as pd
from findata.utils.plotting import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from findata.configs import TECHNICAL_WINDOWS, DATA_MAP

PERIOD_MAP = {name: TECHNICAL_WINDOWS[base] for base, names in DATA_MAP.items() for name in names}


class LogStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = np.asarray(X)
        X_log = np.log1p(X + self.epsilon)
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_log = np.log1p(X + self.epsilon)
        return self.scaler.transform(X_log)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def get_scaler(scaler):
    if scaler == 'minmax':
        return MinMaxScaler()
    elif scaler == 'robust':
        return RobustScaler()
    elif scaler == 'standard':
        return StandardScaler()
    elif scaler == 'log_standard':
        return LogStandardScaler()
    elif scaler == 'quantile':
        return QuantileTransformer(output_distribution='normal')
    elif scaler == 'power':
        return PowerTransformer(method="yeo-johnson", standardize=True)
    else:
        raise ValueError('Unknown scaler')

class PCAProcessor:
    """Feature-engineering + scaling + PCA pipeline for one feature family.

    Two modes:
    - fit (``state=None``): scaler and PCAs are fit on the training split of
      `data_splits`, then applied to all rows.
    - apply (``state`` from a previous fit's :meth:`get_state`): the fitted
      transformers are reused to transform new raw data — nothing is refit, and
      `data_splits` is not required. Config args (scaler, feature_set, ...) are
      ignored in favor of the values captured in the state.
    """

    def __init__(self, data, data_splits=None, n_components=None, scaler='standard', pca_groups=None,
                 arcsinh=False, verbose=False, feature_set='med', whiten_final=True,
                 final_pca=False, final_n_components=None, state=None):
        self.raw_data = data
        self.data_splits = data_splits
        self.state = state
        self._fit = state is None
        if self._fit:
            if data_splits is None:
                raise ValueError("data_splits is required when fitting (pass state= to apply a saved fit)")
            self.scaler = get_scaler(scaler)
            self.pcas = {}
            self.arcsinh = arcsinh
            self.feature_set = feature_set
            self.whiten_final = whiten_final
            self.final_pca = final_pca
            self.final_n_components = final_n_components if final_n_components is not None else n_components
            self.n_components = n_components
            self.feat_eng_data = pd.DataFrame(index=data.index[data.index.get_level_values('date') >= data_splits.train_start])
        else:
            self.scaler = state['scaler']
            self.pcas = state['pcas']
            self.arcsinh = state['arcsinh']
            self.feature_set = state['feature_set']
            self.whiten_final = state['whiten_final']
            self.final_pca = state['final_pca']
            self.final_n_components = state['final_n_components']
            self.n_components = state['n_components']
            self.feat_eng_data = pd.DataFrame(index=data.index)
        self.pca_groups = pca_groups if pca_groups else {}
        self.eps = 1e-7
        self.verbose = verbose
        self.ticker = self.raw_data.index.get_level_values(0).unique().to_series().sample(1).iloc[0]
        self._process()

    def _process(self):
        self._feature_engineer()
        self._resolve_feat_eng_data()
        if not self._fit:
            self._align_to_state()
        self._scale()
        self._transform()

    def _align_to_state(self):
        """Apply mode: enforce the exact feature-engineering column set/order the
        scaler and PCAs were fit on (sklearn transformers are positional)."""
        expected = self.state['feat_eng_cols']
        missing = [c for c in expected if c not in self.feat_eng_data.columns]
        extra = [c for c in self.feat_eng_data.columns if c not in expected]
        if missing or extra:
            raise ValueError(
                f"{type(self).__name__}: engineered columns don't match the fitted state "
                f"(missing={missing}, extra={extra}). Was the state saved with the same feature_set/raw columns?"
            )
        self.feat_eng_data = self.feat_eng_data[expected]

    def get_state(self) -> dict:
        """Everything needed to re-apply this processor's fitted transforms to new data."""
        if not self._fit:
            return self.state
        return {
            'class': type(self).__name__,
            'feature_set': self.feature_set,
            'scaler': self.scaler,
            'pcas': self.pcas,
            'arcsinh': self.arcsinh,
            'n_components': self.n_components,
            'final_pca': self.final_pca,
            'whiten_final': self.whiten_final,
            'final_n_components': self.final_n_components,
            'feat_eng_cols': list(self.feat_eng_data.columns),
            'pca_group_cols': dict(self.resolved_pca_groups_),
            'transformed_cols': list(self.transformed_data.columns),
            'final_cols': list(self.final_data.columns),
        }

    def _get_training_data(self, data):
        d = data.index.get_level_values('date')
        date_mask = (d >= self.data_splits.train_start) & (d <= self.data_splits.train_end)
        if self.data_splits.val_holdout_tickers:
            ticker_mask = ~data.index.get_level_values('ticker').isin(self.data_splits.val_holdout_tickers)
            return data[date_mask & ticker_mask].copy()
        return data[date_mask].copy()

    def _feature_engineer(self):
        raise NotImplementedError("Please implement this method")

    def _resolve_feat_eng_data(self):
        """Drop rolling-window burn-in NaN rows so this processor can scale/PCA cleanly.

        Ticker eligibility (sufficient training-window coverage) is a global decision
        resolved once in `build_features`; here we only clean this processor's own NaNs.
        """
        if self.data_splits is not None:
            # Tickers whose raw data doesn't reach back to the full pre-training window
            raw_min_dates = self.raw_data.groupby(level='ticker').apply(
                lambda x: x.index.get_level_values('date').min()
            )
            suspect_tickers = set(
                raw_min_dates[raw_min_dates > self.data_splits.data_start].index
            )

            # Drop NaN rows; warn only for tickers not expected to have NaN heads
            nan_mask = self.feat_eng_data.isna().any(axis=1)
            nan_tickers = set(self.feat_eng_data[nan_mask].index.get_level_values('ticker').unique())
            unexpected_nan = nan_tickers - suspect_tickers
            if unexpected_nan:
                print(f"  Dropping NaN rows from unexpected tickers: {sorted(unexpected_nan)}")
        self.feat_eng_data = self.feat_eng_data.dropna()
        self.scaled_data, self.transformed_data, self.final_data = (pd.DataFrame(index=self.feat_eng_data.index).copy() for _ in range(3))


    @property
    def tail(self):
        return len(self.feat_eng_data.loc[self.ticker])

    @property
    def dataset(self):
        """Final transformed dataset for model training."""
        return self.final_data

    def _scale(self):
        self._fit_apply_scaler(list(self.feat_eng_data.columns))

    def _fit_apply_scaler(self, scaling_columns):
        """Fit the scaler on the train split (fit mode) or reuse the loaded scaler
        (apply mode), then transform all rows. Shared by the base `_scale` and the
        subclass overrides that scale only a subset of columns."""
        if self._fit:
            train_data = self._get_training_data(self.feat_eng_data)
            self.scaler.fit(train_data[scaling_columns])
        scaled = self.scaler.transform(self.feat_eng_data[scaling_columns])
        self.scaled_data[scaling_columns] = np.arcsinh(scaled).clip(-3.5, 3.5) if self.arcsinh else scaled


    def _transform(self):
        if self._fit:
            train_data = self._get_training_data(self.scaled_data)
            # Resolved {group: columns} is captured here (not re-derived at apply time)
            # so apply mode feeds each PCA exactly the columns it was fit on.
            self.resolved_pca_groups_ = {}
            for name, col_keys in self.pca_groups.items():
                cols = key_search(self.scaled_data.columns, col_keys)
                if not cols:
                    continue
                self.resolved_pca_groups_[name] = cols
                pca = PCA(n_components=self.n_components)
                self.pcas[name] = pca
                pca.fit(train_data[cols])
        else:
            self.resolved_pca_groups_ = self.state['pca_group_cols']
        used_cols = []
        for name, cols in self.resolved_pca_groups_.items():
            used_cols.extend(cols)
            pca_transformed = self.pcas[name].transform(self.scaled_data[cols])
            new_columns = [f'{name}_pc{i + 1}' for i in range(pca_transformed.shape[1])]
            self.transformed_data[new_columns] = pca_transformed
            if self.verbose and self._fit:
                pca = self.pcas[name]
                cum_var = np.cumsum(pca.explained_variance_ratio_).round(2)
                cum_var = cum_var[:np.sum(cum_var < 0.995) + 1]
                print(f'{name} PCA variance for {len(cum_var)} (of {len(cols)}) components: {cum_var}')
        for col in self.scaled_data.columns:
            if col not in used_cols:
                self.transformed_data[col] = self.scaled_data[col]
        if self.final_pca:
            self._apply_final_pca()
        else:
            self.final_data = self.transformed_data
        if not self._fit:
            expected = self.state['final_cols']
            missing = [c for c in expected if c not in self.final_data.columns]
            if missing:
                raise ValueError(
                    f"{type(self).__name__}: applied output is missing fitted columns {missing}"
                )
            self.final_data = self.final_data[expected]

    def _apply_final_pca(self):
        if self._fit:
            train_pcs = self._get_training_data(self.transformed_data)
            pca_final = PCA(n_components=self.final_n_components, whiten=self.whiten_final)
            self.pcas['_final'] = pca_final
            pca_final.fit(train_pcs)
            input_cols = list(self.transformed_data.columns)
        else:
            pca_final = self.pcas['_final']
            input_cols = self.state['transformed_cols']
            missing = [c for c in input_cols if c not in self.transformed_data.columns]
            if missing:
                raise ValueError(
                    f"{type(self).__name__}: final-PCA input is missing fitted columns {missing}"
                )
        final_transformed = pca_final.transform(self.transformed_data[input_cols])
        final_cols = [f'final_pc{i + 1}' for i in range(final_transformed.shape[1])]
        self.final_data[final_cols] = final_transformed
        if self.verbose and self._fit:
            cum_var = np.cumsum(pca_final.explained_variance_ratio_).round(2)
            cum_var = cum_var[:np.sum(cum_var < 0.995) + 1]
            print(f'{self.__class__.__name__} Final PCA: {len(final_cols)} components from {len(self.transformed_data.columns)}: {cum_var}')

    def plot_scale_compare(self, cols=None):
        d = self.feat_eng_data.index.get_level_values('date')
        pre_train = self._get_training_data(self.scaled_data)
        pre_val_test = self.scaled_data[d > self.data_splits.validation_start]
        cols = get_columns(pre_train, cols) if cols else pre_train.columns
        for col in cols:
            plot_df_hists(self.feat_eng_data[[col]], pre_train[[col]], pre_val_test[[col]], sharex_ax=[1, 2],
                          titles=[f'Unscaled Train Data: {col}', 'Scaled Train Data', 'Scaled Validation Data'])

    def plot_corr(self, col_keys=None, title=None, transformed=False):
        data = self.transformed_data if transformed else self.feat_eng_data
        cols = get_columns(data, col_keys) if col_keys else data.columns
        corr = data[cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title(title if title else f"{self.__class__.__name__} Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def plot_pca_corr(self):
        total_df = pd.DataFrame(index=self.feat_eng_data.index)
        all_cols = []
        for name, col_keys in self.pca_groups.items():
            cols = key_search(self.feat_eng_data.columns, col_keys)
            all_cols.extend(cols)
            if not cols:
                continue
            df = self.feat_eng_data[cols].copy()
            pc_cols = key_search(self.transformed_data.columns, name)
            df[pc_cols] = self.transformed_data[pc_cols]
            total_df[pc_cols] = self.transformed_data[pc_cols]
            corr = df.corr()
            plt.figure(figsize=(12, 10))
            plt.title(f'{name} Correlation')
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        unused = [c for c in self.feat_eng_data if c not in all_cols]
        if unused:
            total_df[unused] = self.feat_eng_data[unused]
        corr = total_df.corr()
        plt.figure(figsize=(12, 10))
        plt.title('PCA Correlation')
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')

    def plot_final_pca_corr(self):
        data = pd.concat([self.transformed_data, self.final_data], axis=1)
        corr = data.corr()
        plt.figure(figsize=(12, 10))
        plt.title('PCA Correlation')
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')


    def plot_fe(self, ticker=None, tail=None):
        raise NotImplementedError("Please implement this method")

    def compare_scalers(self, scalers=['standard', 'robust', 'power'], col_keys=None, x_lim=True):
        data = self.feat_eng_data
        cols = get_columns(data, col_keys) if col_keys else data.columns
        data = data[cols].copy()
        scaled_data = {s: get_scaler(s).fit_transform(data) for s in scalers}
        for i in range(len(cols)):
            fig, axs = plt.subplots(nrows=len(scalers))
            for ax, scaler in zip(axs, scalers):
                ax.hist(scaled_data[scaler][:, i], bins=200)
                ax.set_xlim(-3, 3)
            axs[0].set_title(cols[i])


# ---------------------------------------------------------------------------
# Column utilities
# ---------------------------------------------------------------------------

def get_columns(df, keys, ignore=None):
    ignore = ignore if isinstance(ignore, list) else [ignore] if isinstance(ignore, str) else []
    if isinstance(keys, dict):
        cols = []
        for base, k in keys.items():
            k = [k] if isinstance(k, str) else k
            cols.extend([c for c in df.columns if base in c and any(cc in c for cc in k) and not any(i in c for i in ignore)])
    else:
        keys = [keys] if isinstance(keys, str) else keys
        cols = [c for c in df.columns if any(cc in c for cc in keys) and not any(i in c for i in ignore)]
    return cols

def key_search(search_items: Iterable[str], keys: List[str]|str, ignore: List[str] | str=None) -> List[str]:
    ignore = ignore if isinstance(ignore, list) else [ignore] if isinstance(ignore, str) else []
    keys = [keys] if isinstance(keys, str) else keys
    cols = [c for c in search_items if any(cc in c for cc in keys) and not any(i in c for i in ignore)]
    return cols

def sort_columns(df, keys):
    def col_rank(col):
        for i, kw in enumerate(keys):
            if kw in col:
                return i
        return len(keys)
    return df[sorted(df.columns, key=col_rank)]



def fe_velocity(source, ema_windows, log=False, acceleration=False, z_norm=False, z_window=None):
    """
    Compute EMA-based velocity (and optionally acceleration) for consecutive window pairs.

    Parameters
    ----------
    source : pd.Series or dict[int, pd.Series]
        Series  → EMAs are computed internally via ewm(span=w).
        dict    → pre-computed EMA series keyed by window size (e.g. from DB).
    ema_windows : list[int]
        Ascending window sizes. Each consecutive pair (fast, slow) produces one output column.
    log : bool
        False → velocity = ema_fast - ema_slow          (oscillators, OBV, AD)
        True  → velocity = log(ema_fast / ema_slow)     (price EMAs; values must be > 0)
    acceleration : bool
        If True, also emit acc{f}_{s} = velocity - EMA(velocity, sig_span(slow)).
    z_norm : bool
        If True, z-normalise each velocity using a rolling window before outputting.
    z_window : int or None
        Rolling window for z-normalisation. Defaults to the slow span of each pair.

    Returns
    -------
    pd.DataFrame  columns: vel{f}_{s} [, acc{f}_{s}]
    """
    if isinstance(source, dict):
        ema_cache = source
        index = next(iter(source.values())).index
    else:
        ema_cache = {w: source.ewm(span=w, adjust=False).mean() for w in ema_windows}
        index = source.index

    out = pd.DataFrame(index=index)
    for i in range(len(ema_windows) - 1):
        f, s = ema_windows[i], ema_windows[i + 1]
        fast, slow = ema_cache[f], ema_cache[s]
        vel = np.log(fast / slow) if log else (fast - slow)
        if z_norm:
            win = z_window if z_window is not None else s
            vel = (vel - vel.rolling(win).mean()) / vel.rolling(win).std()
        out[f'vel{f}_{s}'] = vel
        if acceleration:
            out[f'acc{f}_{s}'] = vel - vel.ewm(span=sig_span(s), adjust=False).mean()
    return out


def fe_oscillator_momentum(df, cols, ema_windows=None, feature_set='med'):
    """
    Velocity uses subtraction (ema_fast - ema_slow); appropriate for bounded/symmetric indicators
    """
    assert feature_set in ['low', 'med', 'high']
    if ema_windows is None:
        ema_windows = [6, 16] if feature_set == 'low' else [4, 8, 16]
    out = pd.DataFrame(index=df.index)
    for ind in cols:
        out[f'{ind}_raw'] = df[ind]
        for i in range(len(ema_windows) - 1):
            f, s = ema_windows[i], ema_windows[i + 1]
            out[f'{ind}_ema{f}'] = e1 = df[ind].ewm(span=f, adjust=False).mean()
            out[f'{ind}_ema{s}'] = e2 = df[ind].ewm(span=s, adjust=False).mean()
            out[f'{ind}_vel{f}_{s}'] = md = e1 - e2
            if feature_set=='high':
                out[f'{ind}_acc{f}_{s}'] = md - md.ewm(span=sig_span(s), adjust=False).mean()
    return sort_columns(out, ['raw', 'ema','vel','acc'])

def sig_span(window): return max(2, round(0.35 * window))

def get_column_names(names : str | Iterable[str]):
    if isinstance(names, str):
        return f'{names}_{PERIOD_MAP[names]}' if names in PERIOD_MAP else names
    return [f'{name}_{PERIOD_MAP[name]}' if name in PERIOD_MAP else name for name in names]

def get_column(df, name) -> pd.Series:
    return df[get_column_names(name)]
