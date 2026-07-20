# findata.analysis

Feature evaluation + interactive visualization. Depends on `findata.preprocess`
(never the reverse). Panel conventions match preprocess: `(ticker, date)`
MultiIndex feature panels, `{group}__{feature}` column taxonomy; the
market-structure module additionally works on wide `date × ticker` frames
(one indicator at a time).

## Modules

| Module | Class / surface | What it answers |
|---|---|---|
| `rank_ic.py` | `RankICAnalysis` | daily cross-sectional Spearman rank IC vs forward log returns, ICIR summaries (`t_stat` uses `n_eff = n_days / H` for overlapping horizons); fully vectorized |
| `search.py` | `ProcessingSearch`, `default_variants()` | which processing chain (Kalman/SSA × PCA/ZCA/HPCA × market-mode removal) carries the most rank IC per horizon, with strict train-window fitting, embargoed train IC, and per-variant MP diagnostics. `default_variants(include_modes=True)` adds the ticker-space `modes` / `cs_whiten` chains (pass `ticker_meta=`). Driver: `scripts/run_feature_search.py` |
| `structure.py` | `CorrelationStructureAnalysis` | does the hand taxonomy match the data? MP spectrum → denoised correlation → within/across-group stats → data-driven clusters vs taxonomy (ARI) |
| `market_structure.py` | module of functions | the wide-panel RMT research toolkit: `indicator_panel` / `feature_panels` loaders, `spectrum`, null models (`shuffle_null`, `circular_shift_null`), lead-lag, `remove_top_pcs`, `remove_group_modes`, **`decompose_modes`** (global + sector modes + residual, exact identity), `transform_layers`. Full-period, descriptive fits — the leakage-safe production twin is `preprocess.market_modes.ModeDecomposer` |
| `plotting/` | see below | interactive explorers + static savers |

## plotting/ package

```
base.py                 session machinery every explorer shares: SessionSync
                        (x-range capture + RangeTool minimap link + autorange
                        retrigger), minimap_opts, ExplorerBase (.app / .serve)
style.py                the visual encoding: color = layer, shade+dash = ticker
feature_explorer.py     FeatureExplorer   — linked per-group feature browser
transform_explorer.py   TransformExplorer — transform/mode reward-path browser
prediction_explorer.py  PredictionExplorer — forecast vs realized quantiles
static.py               matplotlib savers (layers, modes, fan charts,
                        prediction bands, calibration, median-vs-realized)
```

All explorers follow the same serving contract: widgets and DynamicMaps are
built **per session** (`serve()` passes the factory to `pn.serve`; `.app`
returns a fresh instance for notebooks). Legend clicks mute curves; the bottom
minimap drives the shared x-zoom through a Bokeh RangeTool; y-axes autorange
to the visible window.

### FeatureExplorer

Wide frame (or `(ticker, date)` MultiIndex → ticker dropdown) + a
`{panel_title: columns}` structure. Reached most easily through any feature
group's `plot_fe(ticker=...)`.

### TransformExplorer

`{layer_name: wide date × ticker}` frames (e.g. `market_structure.transform_layers()`
merged with `decompose_modes()["panels"]`) + optional mode frame. Used by
`scripts/market_structure/reward_structure.py --mode interactive`.

### PredictionExplorer

Consumes the tidy predictions frame forecasting's `scripts/test.py` writes to
`<run_dir>/test/predictions.parquet`:

```
ticker | date (decision date) | h | pred_q05..pred_q95 (cum log-return space) | realized | sigma
```

Widgets: ticker, horizon, scale (`raw` | `h-norm` = /√h | `vol-norm` = /(σ√h)),
view (`cumulative` | `per-step`), quantile-band toggles (90/80/50%), minimap
scrubber, plus a median-error strip. All conversions happen on the fly from
the raw-space columns. Per-step bands difference the cumulative quantiles —
exact for the median, approximate for outer quantiles.

```python
import pandas as pd
from findata.analysis import PredictionExplorer
preds = pd.read_parquet("run_dir/test/predictions.parquet")
PredictionExplorer(preds).serve()
```

The static savers (`save_fan_chart`, `save_prediction_bands`,
`save_calibration`, `save_pred_scatter`) take the same frame, so one parquet
feeds both the PNG report and the live app.
