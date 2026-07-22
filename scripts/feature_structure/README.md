# scripts/feature_structure — the feature-cross-section structure studies

The feature-axis companion to [`scripts/market_structure`](../market_structure/README.md).
Where that series studies the **ticker** cross-section (N×N return correlation,
global + sector modes), this one studies the **feature** cross-section (F×F
engineered-feature correlation) and how it relates to those same market modes.

Both are **descriptive** studies: everything is fit over the **full sample**
(all dates T, all tickers N), not a train window. The leakage-safe production
version of the mode decomposition is
`findata.preprocess.market_modes.ModeDecomposer` used inside the pipeline; here
it is fit on the whole panel purely to characterize structure. Three studies:

- **feature correlation (F×F)** — `feature_correlation_structure.py`: how the
  engineered features correlate, and whether the market/sector modes explain it.
- **MP information (N×N)** — `mp_information_search.py`: the Marchenko-Pastur
  signal content of the N-stock cross-section, on returns and on each feature.
- **rank IC (predictive)** — `rank_ic_decomposition.py`: how much of each
  feature's forward-return predictive power is market-timing vs genuine
  cross-sectional selection.

All three use the same three representations — `base`, `global_removed` (market
mode / PC1 removed), `sector_removed` (global + per-sector modes removed). Shared
plumbing lives in `_common.py` (`--tickers N`, `--feature-set`, `--start/--end`,
`--min-sector-size`, `--db-path`). Artifacts land under
`experiments/feature_structure/<script>/`.

## Scripts

### `feature_correlation_structure.py`
Runs `CorrelationStructureAnalysis` (MP spectrum, denoised within/across-taxonomy
correlation, data-driven clustering vs the trend/vol/momentum/volume taxonomy,
ARI) on all three feature representations, then compares them: **where do the
feature correlations differ once the market/sector modes are stripped?**

It then takes two closer looks at the residual:
- **residual vs components** — using `ModeDecomposer(output="residual+components")`
  it correlates each residual feature with the global- and sector-**component**
  columns (`{feat}_gmode`, `{feat}_smode`). A residual is orthogonal to its *own*
  component by construction, so any surviving correlation is with a *different*
  feature's mode channel — leftover shared market/sector structure.
- **surviving redundancy** — the top-N feature pairs (`--top-pairs`, default 10)
  that stay highly correlated after global+sector removal (the intrinsic
  within-name redundancy the modes never explained), plus a count of strong
  pairs (`|corr| ≥ --corr-threshold`) that survive vs get killed.

Outputs (`feature_vs_market/`): a full CSA artifact set per representation
(`base/`, `global_removed/`, `sector_removed/`), `representation_summary.csv`,
delta heatmaps (`delta_*.png`) + most-changed pairs (`top_changed_*.csv`);
`residual_vs_global_components.png` / `residual_vs_sector_components.png` +
`residual_component_leakage.csv` (per feature: strongest |corr| with any
global/sector component); and `surviving_pairs.png` / `surviving_pairs.csv`
(base vs residual |corr| for the still-correlated pairs). If feature correlation
is intrinsic (RSI≈CCI on the same name), the three panels look nearly identical
and the surviving pairs keep most of their base |corr|; if it is market-driven,
removing the modes collapses it.

```bash
.venv/bin/python scripts/feature_structure/feature_correlation_structure.py --tickers 200
.venv/bin/python scripts/feature_structure/feature_correlation_structure.py --top-pairs 15
```

### `mp_information_search.py`
MP signal content of the **N-stock cross-section** (N×N — the object
Marchenko-Pastur is built for), swept across **every panel** — the return panel
*and* each feature panel — for base / global_removed / sector_removed. How much
information each mode layer carries:

```
base            -> total signal (eigenvalues above the MP bulk edge λ₊)
base - global   -> the one global market mode
global - sector -> the sector modes
sector_removed  -> idiosyncratic structure that survives
```

Comparing returns against features (rsi, obv_vel, …) shows whether the
market/sector structure is a returns-only phenomenon or shared across the whole
feature cube. Panels come from `market_structure.feature_panels`
(`--features` to choose; default is the 8 base indicators incl. `log_return`).

Outputs (`mp_information_search/`): `mp_summary.csv` (per feature ×
representation: n, T, q, mean_corr, n_signal, λ₊, noise_var, top-eig share), the
returns eigenvalue-density panel vs the fitted MP bulk (`mp_returns_density.png`),
and per-feature comparison bars (`mp_signal_by_feature.png`,
`mp_top_share_by_feature.png`).

MP needs a real cross-section — run with a few hundred tickers. The mode-removed
spectra are rank-reduced (projection + renormalization), so on a tiny N the
noise fit degenerates; at N in the hundreds it behaves like q4/q5.

```bash
.venv/bin/python scripts/feature_structure/mp_information_search.py --tickers 300
```

### `rank_ic_decomposition.py`
Cross-sectional Spearman **rank IC** (feature vs forward return) for the three
representations, neutralizing **both sides** consistently:

```
base            all features           -> all forward returns (N stocks)
global_removed  market-mode-removed    -> market-neutral forward returns (PC1 out)
sector_removed  global+sector-removed  -> sector-neutral forward returns
```

Feature neutralization is `ModeDecomposer(axis="ticker")` (all engineered
features); return neutralization is the `market_structure` residual return
series, forward-summed over each horizon. A feature whose |ICIR| is high on
`base` but collapses on `global_removed` was predicting the **market mode**
(systematic timing); one whose |ICIR| survives on the residual returns is
genuine **cross-sectional selection**.

An exploratory **sector-level** rank IC also runs (`--no-sector` to skip):
features and returns are equal-weight-aggregated to the ~11 sectors and the
rank IC is taken across sectors — does the feature predict **sector rotation**?
With only ~11 names the daily IC is noisy (a rough look, not a verdict).

Outputs (`rank_ic_decomposition/`):
- `rank_ic_summary.xlsx` — the readable deliverable, **split by sheet**:
  `headline_top5`, `headline_full`, one `ticker_<rep>` sheet per representation
  (feature × horizon, ICIR + t-stat), a `pivot_icir_H*` sheet per plot horizon,
  and `sector_rank_ic` / `sector_headline`.
- Per plot horizon (`--plot-horizons`, up to 3): `icir_comparison_H*.png`
  (grouped bars) and `ic_attribution_H*.png` (base vs sector-neutral scatter —
  points below the diagonal are market/sector-driven); plus
  `top5_icir_vs_horizon.png` (strength vs horizon per representation).
- Per-representation subdirs (`base/`, `global_removed/`, `sector_removed/`) and
  `sector/` with cumulative-IC plots + the sector rotation bars; flat
  `rank_ic_summary.csv` for programmatic use.

```bash
.venv/bin/python scripts/feature_structure/rank_ic_decomposition.py --tickers 250
.venv/bin/python scripts/feature_structure/rank_ic_decomposition.py \
    --horizons 1 5 10 21 63 --plot-horizons 1 10 63
```

## Relationship to market_structure q7

`market_structure/q7_feature_mode_correlation.py` asks the same "do the two PCA
axes reduce to one another" question on the 8 raw `DEFAULT_FEATURES` via
`mode_feature_correlation`. This directory generalizes it to the **full
engineered feature panel** and routes it through the production
`ModeDecomposer` + the `CorrelationStructureAnalysis` / MP / rank-IC harnesses,
so the same conclusions can be checked against every feature the pipeline
actually builds.

`rank_ic_decomposition.py` also differs from `findata.analysis.ProcessingSearch`
(the `run_feature_search.py` driver): the search scores each processing variant's
features against **raw** forward returns to pick a preprocessing recipe; here
**both** the features and the target returns are neutralized together, to
attribute existing predictive power to the market / sector / idiosyncratic
layers.
