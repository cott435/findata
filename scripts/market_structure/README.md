# scripts/market_structure — the cross-sectional structure research series

Question-driven scripts probing how much of the ticker cross-section is shared
structure (market / sector modes) vs idiosyncratic signal, and how that
structure interacts with the feature axis. All use
`findata.analysis.market_structure` on wide `date × ticker` panels, fit
descriptively on the **full period** (no train window — these are structure
studies; the leakage-safe production version of the winning decomposition is
`findata.preprocess.market_modes.ModeDecomposer`).

Shared plumbing: `_common.py` (`--start 2021-01-01 --end --tickers N
--indicator log_return --n-draws 300 --max-lag 10 --seed`; loads a dense
coverage-filtered panel + sector/cap metadata). Artifacts land under
`experiments/market_structure/<qN_...>/` as CSVs + PNGs. Figures below are from
the full universe (N≈853 tickers, T≈1320 daily obs, 2021→2026) unless noted.

## What each question asks — and what was found

| Script | Question | Answer found |
|---|---|---|
| `q1_group_correlation.py` | Do sector / cap groups correlate more than the whole market, and does Kalman smoothing change it? | **Sectors: yes (~6 of 11 beat their random-group null at p<0.05); cap buckets: mostly no.** Sector membership is real structure; market-cap membership carries little. Kalman smoothing keeps the strongest sectors (Energy, Financials, Industrials) but inflates the analytic MP signal count into nonsense, so smoothed panels are judged against a shuffle-then-smooth null instead. |
| `q2_leadlag.py` | Does time-shifting tickers raise correlations, and do lead-lag-aligned clusters match sector / cap? | **No lead-lag at daily resolution** — 96%+ of pairs already peak at lag 0, and the shift "gain" is smaller than the circular-shift selection-bias null (p≈1). Every ticker aligns to the market mode at shift 0. Sector clusters are invisible in raw correlation (ARI≈0.04) but emerge only after removing the global mode (ARI≈0.27, p<0.001); cap never emerges. |
| `q3_signal_vs_sample_noise.py` | Core RMT reproduction: P(C_ij), eigenvalue density vs Marchenko-Pastur, and the shuffle-independence demonstration | **Confirms the Nobi/Bouchaud picture** — real ⟨C_ij⟩ is strongly positive (≈0.29) while the shuffle null centers on 0 (≈1/√T); one huge "market mode" eigenvalue leaks far past λ₊ and the bulk collapses onto MP under shuffling. Correlated-ticker subgroups push ⟨C_ij⟩ and top-eigenvalue share far higher than the uncorrelated subgroup. |
| `q4_pc1_removal.py` | After PC1 is removed / transformed back / renormalized, do NEW signal eigenvalues emerge? | **No — the opposite.** Removing the market mode *raises* the noise floor (σ²: 0.37→0.54, λ₊: 1.00→1.47) because PC1 had been suppressing the bulk mean, and mean pairwise correlation collapses 0.246→−0.001. n_signal *falls* (107→87), revealing a cleaner, smaller set of sector/style modes rather than hidden new ones. The shuffle control creates zero signal, confirming the operation invents nothing. |
| `q5_sector_mode_removal.py` | With each sector's own mode removed, is there residual cross-sector (macro / style) correlation? | **No.** Raw cross-sector mean corr is +0.224 (p=0.01); after removing each sector's own PC1 it is +0.00003 with p=0.94 — statistically indistinguishable from independence. Sector modes (which each carry their slice of the global mode) fully account for cross-sector co-movement; only within-sector residual structure survives. |
| `q6_reward_decomposition.py` | How does cumulative log-return split into global + sector + idiosyncratic, and does the removal ORDER matter? | **Split ≈ 26% global / 11% sector / 63% idiosyncratic (per-ticker mean).** Order barely matters: global-first (A) vs sector-first (B) agree tightly — corr(g_A, g_B)≈1, and every sector's mode loads ~0.63–0.93 on the shared global part, so the two orderings recover the same factors. Most of a single stock's reward is idiosyncratic; the shared part is dominated by the one global mode. |
| `q7_feature_mode_correlation.py` | Two PCAs on two axes of the (time × ticker × feature) cube: do per-feature market modes explain the pooled feature-feature correlation? | **No — the two PCAs are nearly orthogonal.** Removing each feature's global market mode drops feature-feature correlation by only **~4%** (global+sector ≈9%), and the feature clusters don't reorganize (ARI=1.0). ~91% of feature correlation is intrinsic within-stock indicator redundancy (RSI≈CCI on the same name), untouched by market-neutralizing. Ticker-axis and feature-axis PCA factorize independent structure and compose cleanly in either order. |
| `reward_structure.py` | Interactive/static explorer for the whole decomposition: transform layers + mode series | Tooling, not a hypothesis — see below. |

## Takeaways for the pipeline

1. **The global + sector modes are real, hierarchically separable, and removable without inventing signal** (Q1/Q3/Q4/Q5) — exactly what `market_modes.ModeDecomposer` and the `forecast_variants()` datasets (`resid`, `resid_comp`, `resid_white`) now expose to the forecasting search.
2. **Market-neutralizing does NOT decorrelate features** (Q7): the ticker-axis mode removal and the feature-axis PCA/ZCA are complementary, not substitutes — keep both.
3. **Sector structure only exists once the global mode is stripped** (Q2/Q5): cluster on residuals, never on raw correlation.
4. **No daily lead-lag** (Q2): contemporaneous cross-sectional modes are the right object; revisit only at slower feature resolutions.

## `reward_structure.py` — the decomposition explorer

Samples N tickers and visualizes how the transforms alter cumulative reward
paths (color = transform, shade + dash = ticker, darkest = most spaced dashes).

- `--mode interactive` serves a `TransformExplorer` (select/unselect tickers,
  toggle transforms, a `focus` stepper to move through them one at a time, a
  mode panel showing global vs sector vs compounded, legend-click mute, minimap
  zoom).
- `--mode static` saves PNGs via `save_layer_plot` / `save_modes_plot`.
- Either mode always writes `layer_*.parquet`, `modes.parquet`, `loadings.csv`,
  `sample_meta.csv`.
- `--indicator log_return | rsi | cci | willr | mfi | cmf | obv_vel`.

## Running

From the repo root:

```bash
.venv/bin/python scripts/market_structure/q1_group_correlation.py
.venv/bin/python scripts/market_structure/q7_feature_mode_correlation.py --tickers 300
.venv/bin/python scripts/market_structure/reward_structure.py --mode interactive --indicator rsi
```

Add `--tickers N` to cap the universe for a faster pass; several questions also
saved a small-N sanity variant (`experiments/market_structure/<q>_200/`).
