# findata.preprocess

Feature engineering + composable panel processing. Everything here operates on
one panel shape: a **`(ticker, date)` MultiIndex DataFrame** whose columns are
features named **`{group}__{feature}`** (the double-underscore prefix is the
taxonomy every analysis parses — keep it stable). The `date` index level holds
python `date` objects; normalize with `base.date_values(index)` before
comparing against `DataSplits` boundaries (pandas 3 raises on `Timestamp` vs
`date` comparisons).

```
raw OHLCV+indicators ──engineer──▶ {group}__{feature} panel ──ProcessingPipeline──▶ features
       (DBManager)      (FeatureGroups)                        (GroupScaler + steps)
```

## Entry points

| What | Where | Use |
|---|---|---|
| `FeaturePipeline` | `pipeline.py` | engineer → train-fitted processing across all groups at once |
| `build_features(...)` | `findata/utils/build_features.py` | one-call wrapper returning `FeatureBundle(features, provenance, state)` |
| `PipelineState` | `pipeline.py` | serialized fitted state; `apply(raw)` featurizes new data with **nothing refitting**; raises on non-causal chains unless overridden |
| `forecast_variants()` | `variants.py` | the named dataset variants the forecasting search selects between |
| `ProcessingSearch` | `findata/analysis/search.py` | rank-IC evaluation of processing variants (see `scripts/run_feature_search.py`) |

```python
from findata import build_features, forecast_variants, get_all_data
from findata.configs import DataSplits

splits = DataSplits()
raw, ticker_info = get_all_data(tickers, splits.data_start, splits.data_end)
fb = build_features(raw, splits,
                    processing=forecast_variants()["resid_comp"],
                    ticker_meta=ticker_info)          # sector labels for 'modes'
fb.state.save("state.joblib")                          # production re-apply
```

## Leakage policy (enforced, not conventional)

`ProcessingPipeline.fit_transform` slices the training rows once
(`train_row_mask`: train date window minus `val_holdout_tickers`) and fits
**every** stage only on the train slice of its already-transformed input, then
transforms all rows for the next stage. Scalers, Kalman gains, eigenvectors,
MP fits, cluster structure — all train-window-only. At apply time nothing
refits. `SSADenoiser` is the one non-causal transform (batch SVD sees the
future); `PipelineState.apply` refuses to run non-causal states unless
`allow_non_causal=True`.

## Feature groups

| Group | Module | `name` | Engineered content |
|---|---|---|---|
| `Oscillators` | `oscillators.py` | `momentum` | native-window oscillator position / velocity / acceleration + short-long contrasts |
| `Trend` | `trend.py` | `trend` | log-ratio EMA velocities, ADX/DI |
| `Volatility` | `volatility.py` | `volatility` | Parkinson / Rogers-Satchell / GK vols, BB bandwidth, ATR |
| `Volume` | `volume.py` | `volume` | z-scored EMA velocities of price / OBV / AD |
| `Candle` | `candles.py` | `candle` | log candle geometry, body size |

Each group declares its own `ScaleRule`s (`scale_rules`), compiled into the
`GroupScaler` that always runs as stage 0. `plot_fe(ticker=...)` on any group
serves the interactive `FeatureExplorer`.

## Processing steps (`STEP_REGISTRY`)

Chains are declared as `ProcessingConfig(name, steps=[(step, params), ...])`
and compiled through the registry — adding a transform is one registry entry.

| Step | Class | Axis | What it does |
|---|---|---|---|
| `kalman` | `KalmanDenoiser` | per-ticker time | causal steady-state local-level filter, gains fit per feature |
| `ssa` | `SSADenoiser` | per-ticker time | batch SSA reconstruction — **non-causal**, diagnostics only |
| `demean` / `cs_zscore` / `vol_cs` / `vol_ts` | `CrossSectionalNormalizer` | per-date cross-section | market-level / dispersion normalization (rank-IC-preserving except `vol_ts`) |
| `signal_keep` / `signal_strip` | `SignalProjector` | feature space | keep / strip the MP-significant signal subspace (name-preserving) |
| `pca` | `PCADecorrelator` | feature space | plain PCA → `pc1..pck` |
| `zca` | `ZCAWhitener` | feature space | correlation whitening, name-preserving, optional RMT denoise |
| `hpca` | `HierarchicalPCA` | feature space | RMT-denoised corr → auto-k clusters → per-cluster whitened PCA |
| `standardize` | `Standardizer` | per column | mid-chain re-standardize (train mean/std) |
| **`modes`** | `ModeDecomposer` | **ticker cross-section** (or feature) | global + per-sector PCA mode decomposition — see below |
| **`cs_whiten`** | `CrossSectionWhitener` | ticker cross-section | per-feature ticker-space ZCA on the MP-denoised residual correlation |

## Market-mode decomposition (`market_modes.py`)

Productionizes the `scripts/market_structure` research (leakage-safe version of
`analysis.market_structure.decompose_modes`). Per feature column, on the
standardized train panel:

```
x_i(t) = beta_i^G · G(t)  +  beta_i^S · S_{s(i)}(t)  +  eps_i(t)
```

`G` = global PC1 of the ticker correlation, each sector mode = PC1 of that
sector's **global-residual** block (hierarchically orthogonalized, identity
exact at observed cells). Loadings/betas/stds fit on train rows only; transform
projects the **same-date cross-section** onto the fixed loadings — causal.

Key params: `n_global` / `n_group` (int or `"mp"` → `rmt.iterative_bulk_variance`
signal count), `output="residual"` (name-preserving) or
`"residual+components"` (adds `{col}_gmode` / `{col}_smode` per-ticker
component columns, ~3x width), `axis="feature"` for the same two-stage
decomposition across feature columns (groups = taxonomy prefix, no meta
needed), `min_names` (thin cross-sections pass through standardized).

**Sector labels**: the `modes` step needs a ticker → sector map. Pass the
`ticker_info` frame from `get_all_data` as `build_features(...,
ticker_meta=...)` (or `FeaturePipeline.fit(..., meta=...)`); the fitted map is
stored inside the transform, so `PipelineState.apply` needs nothing extra.
Tickers absent from the train window (e.g. val-holdouts) have no fitted stats
and come back NaN.

## Named dataset variants (`variants.py`)

`forecast_variants()` returns the datasets the forecasting search's
`data.dataset` axis selects between:

| Name | Chain | Question it answers |
|---|---|---|
| `base` | scale-only | control |
| `global_resid` | `modes(group_stage=False)` | is removing just the market mode enough? |
| `resid` | `modes` | global + sector modes removed, residual transformed back |
| `resid_comp` | `modes(output=residual+components)` | is it better to keep the removed structure as separate inputs? |
| `resid_white` | `modes` → `cs_whiten` | does whitening the denoised residual cross-section help? |

## RMT toolbox (`rmt.py`)

Single home for Marchenko-Pastur math (`q = T/N`, T = unique dates).
`iterative_bulk_variance` is the preferred noise/edge estimator (recursive
bulk mean — no bandwidth knob, trace-preserving); `mp_edge` defaults to it.
`denoise_correlation` (residual / shrink), `cluster_corr` (correlation-distance
hierarchical clustering, silhouette auto-k), `spectrum_probe` (per-panel MP
diagnostics used by the search).

## v1/

Frozen reference implementation of the original per-group processors — kept
for comparison, not maintained. Its README documents the v1 formula tables.
