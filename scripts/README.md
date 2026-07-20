# scripts/

Research and lifecycle drivers. Subdirectories have their own READMEs:
[db_handling/](db_handling/README.md) (DB lifecycle CLIs),
[legacy_scripts/](legacy_scripts/README.md) (V1 exploratory analyses),
[market_structure/](market_structure/README.md) (the q1–q7 research series),
[feature_structure/](feature_structure/README.md) (feature comparison analysis).

## run_feature_search.py — the feature-exploration driver

Runs the full **feature-processing search + correlation-structure analysis**:
engineer once, fit/transform every `ProcessingConfig` variant (prefix-cached),
score each variant × horizon by cross-sectional rank IC, and probe each
panel's MP spectrum / cluster structure.

```bash
.venv/bin/python scripts/run_feature_search.py --tickers 15 --quick   # smoke
.venv/bin/python scripts/run_feature_search.py                        # full grid
.venv/bin/python scripts/run_feature_search.py --oscillators rsi cci --horizons 5 21 63
```

Flags: `--tickers N` (sample size), `--quick` (small grid), `--horizons ...`,
`--feature-set low|med|high`, `--oscillators ...`, `--save-states` (persist a
fitted `PipelineState` per variant for reuse).

Artifacts land under `experiments/`:

- `processing_search/<run_id>/` — `results.parquet` (per feature × variant ×
  horizon × split), `summary.csv`, `configs.json` (per-variant steps + MP
  diagnostics), ICIR heatmaps / cumulative-IC PNGs, `hpca_clusters_*.csv`,
  optional `states/*.joblib`
- `feature_structure/` — the correlation-structure report (spectrum,
  dendrogram, taxonomy-vs-clusters)

Reading the results: per-date demean/zscore/vol_cs variants have *identical*
per-feature rank IC to their input by construction (same-date shifts preserve
ranks) — their value shows up through the correlation spectrum, i.e. combined
with a decorrelation step. Kalman gains fitting to 1.0 on already-smooth
EMA-derived features is the local-level model correctly reporting "no
observation noise".

To include the ticker-space mode-decomposition variants (`modes`,
`cs_whiten`) in the search, build them via
`findata.analysis.search.default_variants(include_modes=True)` and pass
`ticker_meta` (sector labels) to `ProcessingSearch`.
