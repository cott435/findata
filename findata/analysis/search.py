"""ProcessingSearch: evaluate feature-processing variants against rank IC.

Answers, in one reusable harness:
  - how predictive are the raw feature-engineered features (per horizon)?
  - how much does temporal filtering (Kalman / SSA) matter?
  - how much does decorrelation matter, and which kind (PCA vs ZCA vs
    auto-clustered hierarchical PCA)?
  - does market-mode removal (per-date demeaning / vol normalization /
    MP-signal projection) uncover finer structure — and does IC improve?

Efficiency contract:
  - features engineered once; GroupScaler fitted once and shared;
  - transform chains are cached by STEP PREFIX, so e.g. the Kalman panel is
    computed once and shared by all kalman+* variants;
  - the y-side of the IC computation (within-date ranks of forward returns) is
    prepared once per horizon and reused across every variant;
  - the IC math itself is fully vectorized (see analysis.rank_ic).

Reading the results:
  - per-date demean/zscore/vol_cs variants have IDENTICAL per-feature rank IC
    to their input by construction (same-date shifts/scales preserve ranks);
    they matter through the correlation spectrum, i.e. combined with a
    decorrelation step (demean+zca, demean+hpca, signal+demean, ...);
  - Kalman gains fit to 1.0 (identity) on already-smooth EMA-derived features
    are the local-level model correctly reporting "no observation noise" —
    force stronger smoothing with e.g. ("kalman", {"gain_cap": 0.3}).

Leakage policy (enforced, not conventional):
  - every fit (scalers, gains, projections, cluster structure, MP fits) sees
    only train-window rows minus val-holdout tickers;
  - train IC is embargoed: the last H trading days of the train window are
    dropped so overlapping forward returns never cross into validation;
  - the test window (>= splits.test_start) is never evaluated here;
  - non-causal variants (SSA) are flagged in every output row and separated in
    the report plots.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from findata.analysis.rank_ic import RankICAnalysis
from findata.configs import EXPERIMENT_DIR
from findata.preprocess import rmt
from findata.preprocess.base import date_values
from findata.preprocess.pipeline import FeaturePipeline, PipelineState
from findata.preprocess.transforms import (STEP_REGISTRY, ProcessingConfig,
                                           ProcessingPipeline, train_row_mask)


def default_variants() -> list[ProcessingConfig]:
    """{none, kalman, ssa} x {none, pca, zca, hpca} grid + market-mode-removal chains."""
    grid = [
        ProcessingConfig(f"{t or 'raw'}+{d or 'none'}", temporal=t, decorrelation=d)
        for t in (None, "kalman", "ssa")
        for d in (None, "pca", "zca", "hpca")
    ]
    market_neutral = [
        ProcessingConfig("demean", steps=[("demean", {}), ("standardize", {})]),
        ProcessingConfig("demean+zca", steps=[("demean", {}), ("standardize", {}), ("zca", {})]),
        ProcessingConfig("demean+hpca", steps=[("demean", {}), ("standardize", {}), ("hpca", {})]),
        ProcessingConfig("volnorm_cs", steps=[("vol_cs", {}), ("standardize", {})]),
        ProcessingConfig("signal+demean",
                         steps=[("signal_keep", {}), ("demean", {}), ("standardize", {})]),
        ProcessingConfig("signal+demean+hpca",
                         steps=[("signal_keep", {}), ("demean", {}), ("standardize", {}),
                                ("hpca", {})]),
    ]
    return grid + market_neutral


DEFAULT_VARIANTS = default_variants()


def _canonical_step(step: tuple) -> tuple:
    name, params = step
    return (name, tuple(sorted((params or {}).items())))


class ProcessingSearch:

    def __init__(self, data: pd.DataFrame, splits, variants=None,
                 horizons=(1, 5, 10, 21, 63), groups=None, feature_set: str = "med",
                 min_assets: int = 10, embargo: bool = True,
                 keep_panels=("raw+none", "demean"),
                 output_dir: str | Path | None = None, verbose: bool = True):
        self.data = data
        self.splits = splits
        self.variants = list(variants) if variants is not None else list(DEFAULT_VARIANTS)
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"Variant names must be unique: {names}")
        self.horizons = list(horizons)
        self.groups = groups
        self.feature_set = feature_set
        self.embargo = embargo
        self.keep_panels = set(keep_panels or ())
        self.analyzer = RankICAnalysis(min_assets=min_assets)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir \
            else EXPERIMENT_DIR / "processing_search" / self.run_id
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # variant panel construction (prefix-cached chains)
    # ------------------------------------------------------------------ #

    def _build_panels(self):
        """Engineer once, scale once, then fit/transform each variant chain,
        caching intermediate panels shared by several variants."""
        fp = FeaturePipeline(groups=self.groups, feature_set=self.feature_set,
                             verbose=self.verbose)
        fp.fit(self.data, self.splits)              # default config == scale-only
        self._pipeline = fp
        self._scaler = fp.processing_.transforms[0]  # fitted GroupScaler, shared
        scaled = fp.features_
        self._train_mask = train_row_mask(scaled.index, self.splits)

        # prefix -> (panel, fitted transform list); cache only shared prefixes
        chains = {v.name: [_canonical_step(s) for s in v.resolved_steps()]
                  for v in self.variants}
        prefix_counts: dict[tuple, int] = {}
        for steps in chains.values():
            for i in range(1, len(steps)):
                key = tuple(steps[:i])
                prefix_counts[key] = prefix_counts.get(key, 0) + 1
        cache: dict[tuple, tuple[pd.DataFrame, list]] = {(): (scaled, [])}

        for config in self.variants:
            steps = config.resolved_steps()
            keys = [_canonical_step(s) for s in steps]
            # longest cached prefix
            start = 0
            panel, fitted = cache[()]
            for i in range(len(keys), 0, -1):
                key = tuple(keys[:i])
                if key in cache:
                    panel, fitted = cache[key]
                    start = i
                    break
            fitted = list(fitted)
            for i in range(start, len(steps)):
                step_name, params = steps[i]
                transform = STEP_REGISTRY[step_name](**params)
                transform.fit(panel[self._train_mask])
                panel = transform.transform(panel)
                fitted.append(transform)
                key = tuple(keys[:i + 1])
                if prefix_counts.get(key, 0) > 1:
                    cache[key] = (panel, list(fitted))
            yield config, panel, [self._scaler] + fitted

    # ------------------------------------------------------------------ #
    # evaluation windows
    # ------------------------------------------------------------------ #

    def _split_masks(self, ic_index: pd.DatetimeIndex, H: int) -> dict[str, np.ndarray]:
        """Date masks for train (embargoed by H trading days) and validation."""
        train_end = pd.Timestamp(self.splits.train_end)
        val_start = pd.Timestamp(self.splits.validation_start)
        val_end = pd.Timestamp(self.splits.validation_end)
        train_dates = ic_index[ic_index <= train_end]
        if self.embargo and H > 0 and len(train_dates) > H:
            train_cut = train_dates[-H]           # drop last H trading days
            train_mask = ic_index < train_cut
        else:
            train_mask = ic_index <= train_end
        val_mask = (ic_index >= val_start) & (ic_index <= val_end)
        return {"train": np.asarray(train_mask), "val": np.asarray(val_mask)}

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #

    def run(self, save_states: bool = False) -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        closes = self.data["close"]

        rows = []
        variant_meta = {}
        panels_kept = {}
        daily_ic_cache = {}   # (variant, H) -> daily IC frame (for report plots)

        # y-side once per horizon (shared by every variant)
        targets = {}
        engineered_index = None

        for config, panel, fitted in self._build_panels():
            if engineered_index is None:
                engineered_index = panel.index
                for H in self.horizons:
                    fwd = self.analyzer.forward_log_returns(closes, H)
                    targets[H] = self.analyzer.prepare_target(fwd, engineered_index)

            causal = all(t.causal for t in fitted)
            steps_desc = [s[0] for s in config.resolved_steps()] or ["scale_only"]
            probe = rmt.spectrum_probe(panel[self._train_mask])
            variant_meta[config.name] = {
                "steps": steps_desc, "causal": causal,
                "n_features": int(panel.shape[1]),
                "n_signal": probe["n_signal"], "lam_plus": round(probe["lam_plus"], 4),
                "top_eig_share": round(probe["top_eig_share"], 4),
            }
            if self.verbose:
                print(f"[{config.name}] features={panel.shape[1]} causal={causal} "
                      f"n_signal={probe['n_signal']} top_share={probe['top_eig_share']:.1%}")

            # persist hpca cluster memberships / optional full state
            for t in fitted:
                if type(t).__name__ == "HierarchicalPCA":
                    pd.DataFrame([(cid, col) for cid, cols in t.clusters_.items()
                                  for col in cols], columns=["cluster", "feature"]) \
                        .to_csv(self.output_dir / f"hpca_clusters_{config.name}.csv", index=False)
            if save_states:
                states_dir = self.output_dir / "states"
                states_dir.mkdir(exist_ok=True)
                proc = ProcessingPipeline(fitted, config=config)
                proc.fitted_ = True
                PipelineState(groups=self._pipeline.groups, processing=proc,
                              engineered_cols=self._pipeline.engineered_cols_,
                              feature_cols=list(panel.columns),
                              feature_set=self.feature_set, causal=causal) \
                    .save(states_dir / f"{config.name}.joblib")
            if config.name in self.keep_panels:
                panels_kept[config.name] = panel

            for H in self.horizons:
                daily = self.analyzer.daily_ic(panel, targets[H])
                daily_ic_cache[(config.name, H)] = daily
                masks = self._split_masks(daily.index, H)
                for split, mask in masks.items():
                    summary = self.analyzer.summarize(daily[mask], H)
                    summary = summary.reset_index()
                    summary.insert(0, "run_id", self.run_id)
                    summary.insert(1, "variant", config.name)
                    summary.insert(2, "steps", "+".join(steps_desc))
                    summary.insert(3, "causal", causal)
                    summary.insert(4, "horizon", H)
                    summary.insert(5, "split", split)
                    summary["group"] = [f.split("__", 1)[0] if "__" in f else "component"
                                        for f in summary["feature"]]
                    rows.append(summary)

        self.results_ = pd.concat(rows, ignore_index=True)
        self.variant_meta_ = variant_meta
        self.panels_ = panels_kept
        self._daily_ic_cache = daily_ic_cache

        self.results_.to_parquet(self.output_dir / "results.parquet")
        with open(self.output_dir / "configs.json", "w") as fh:
            json.dump({name: meta for name, meta in variant_meta.items()}, fh, indent=2)
        self.summary_ = self.summarize()
        self.summary_.to_csv(self.output_dir / "summary.csv", index=False)
        self.plot_report()
        if self.verbose:
            print(f"\nSearch artifacts -> {self.output_dir.resolve()}")
        return self.results_

    # ------------------------------------------------------------------ #
    # aggregation + report
    # ------------------------------------------------------------------ #

    def summarize(self) -> pd.DataFrame:
        """One row per variant x horizon x split."""
        def agg(g: pd.DataFrame) -> pd.Series:
            icir_abs = g["icir"].abs()
            top5 = icir_abs.nlargest(5)
            best = g.loc[icir_abs.idxmax()] if icir_abs.notna().any() else None
            return pd.Series({
                "n_features": len(g),
                "mean_abs_ic": g["ic_mean"].abs().mean(),
                "mean_abs_icir": icir_abs.mean(),
                "top5_mean_abs_icir": top5.mean(),
                "n_sig_t2": int((g["t_stat"].abs() >= 2).sum()),
                "best_feature": best["feature"] if best is not None else None,
                "best_icir": best["icir"] if best is not None else np.nan,
            })

        out = (self.results_
               .groupby(["variant", "causal", "horizon", "split"], sort=False)
               .apply(agg, include_groups=False)
               .reset_index())
        meta = pd.DataFrame(self.variant_meta_).T[["n_signal", "lam_plus", "top_eig_share"]]
        out = out.merge(meta, left_on="variant", right_index=True, how="left")
        return out.sort_values(["split", "horizon", "top5_mean_abs_icir"],
                               ascending=[True, True, False]).reset_index(drop=True)

    def plot_report(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary = self.summary_
        variant_order = [v.name for v in self.variants]

        # variant x horizon heatmaps of top-5 mean |ICIR|, per split
        for split in ("train", "val"):
            sub = summary[summary["split"] == split]
            mat = sub.pivot(index="variant", columns="horizon",
                            values="top5_mean_abs_icir").reindex(variant_order)
            fig, ax = plt.subplots(figsize=(1.6 * len(self.horizons) + 4,
                                            0.42 * len(variant_order) + 2))
            im = ax.imshow(mat.to_numpy(dtype=float), cmap="viridis", aspect="auto")
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels([f"H={h}" for h in mat.columns])
            causal_map = {name: meta["causal"] for name, meta in self.variant_meta_.items()}
            ax.set_yticks(range(mat.shape[0]))
            ax.set_yticklabels([f"{v}{'' if causal_map.get(v, True) else '  [non-causal]'}"
                                for v in mat.index], fontsize=8)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.iloc[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                color="white", fontsize=7)
            ax.set_title(f"Top-5 mean |ICIR| by variant x horizon ({split})")
            fig.colorbar(im, label="top-5 mean |ICIR|")
            fig.tight_layout()
            fig.savefig(self.output_dir / f"heatmap_top5_icir_{split}.png", dpi=140)
            plt.close(fig)

        # best CAUSAL variant on validation: detailed bar + cumulative IC
        causal_val = summary[(summary["split"] == "val") & summary["causal"]]
        if causal_val["top5_mean_abs_icir"].notna().any():
            best = causal_val.loc[causal_val["top5_mean_abs_icir"].idxmax()]
            name, H = best["variant"], int(best["horizon"])
            per_feature = self.results_[
                (self.results_["variant"] == name) & (self.results_["horizon"] == H)
                & (self.results_["split"] == "val")].set_index("feature")
            self.analyzer.plot_icir_bar(
                per_feature, min(15, len(per_feature)),
                self.output_dir / f"icir_bar_best_{name}_H{H}_val.png")
            daily = self._daily_ic_cache[(name, H)]
            top_feats = per_feature["icir"].abs().nlargest(8).index.tolist()
            fig, ax = plt.subplots(figsize=(11, 6))
            for feat in top_feats:
                ax.plot(daily.index, daily[feat].fillna(0).cumsum(), lw=1.3, label=feat)
            ax.axvline(pd.Timestamp(self.splits.validation_start), color="black",
                       ls="--", lw=1, label="validation start")
            ax.axhline(0, color="black", lw=0.8, alpha=0.6)
            ax.set_title(f"Cumulative daily IC — best causal variant '{name}' (H={H})")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(self.output_dir / f"cumulative_ic_best_{name}_H{H}.png", dpi=140)
            plt.close(fig)
