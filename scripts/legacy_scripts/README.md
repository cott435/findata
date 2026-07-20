# scripts/testing — exploratory analyses

One-off studies that motivated pieces of the pipeline. They write figures under
`experiments/` (or a local output dir) and are safe to re-run.

| Script | What it explores |
|---|---|
| `processing_test.py` | Constructs position / velocity / acceleration features for trend (EMA of price) and momentum (EMA of RSI) at multiple scales, plus explicit short-vs-long **contrast** features (instead of hoping PCA finds the difference in near-noise eigenvalues). Saves construction panels + an engineered parquet. |
| `momentum_analysis.py` | The oscillator momentum design study (RSI velocity/acceleration smoothing pairs) behind the `Oscillators` group. |
| `ful_feature_analysis.py` | Runs the entire engineered feature set through one RMT/MP-denoising + hierarchical-clustering pipeline: does the trend/vol/momentum/volume taxonomy match the real statistical structure (ARI, dendrograms, reordered heatmaps)? Superseded by `findata.analysis.structure.CorrelationStructureAnalysis`. |
| `rank_ic_analysis.py` | Cross-sectional rank-IC study for a feature bank vs forward log returns. Superseded by `findata.analysis.rank_ic.RankICAnalysis`. |
| `rmt_pipeline.py` | Staged RMT feature-validation pipeline: null construction (circular-shift vs permutation vs analytic MP), per-feature N×N spectra, market-mode removal, F×F redundancy map. |
| `market_signal_vs_sampling_noise.py` | The Potters/Bouchaud/Laloux + Nobi RMT reproduction on price panels (P(C_ij), eigenvalue density vs MP, market mode). The same analysis, parameterized, lives in `scripts/market_structure/q3`. |
| `find_viable_tickers.py` | Bulk Yahoo metadata pull over the TickerSampler universe to find seedable tickers. |
