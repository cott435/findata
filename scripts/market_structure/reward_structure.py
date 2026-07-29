#!/usr/bin/env python
"""Reward-structure explorer — how do the mode transforms alter cumulative
per-ticker reward paths?

Samples N tickers, decomposes the panel into global market mode -> sector mode
-> idiosyncratic residual (market_structure.decompose_modes), builds the
transform-comparison layers (raw / kalman / demean / minus_global_pc1 /
minus_sector_pc1), and visualizes with a shared encoding:

    COLOR = transform,  SHADE + DASH = ticker (darkest = most spaced dashes)

--mode interactive   serve the TransformExplorer dashboard (select/unselect
                     tickers, toggle transforms, 'focus' steps through them,
                     mode panel shows global vs sector vs compounded, legend
                     click mutes, minimap zooms)
--mode static        save all plots as PNGs (quick look)

Data (sampled layer panels, mode series, loadings, sample meta) is ALWAYS
saved, whichever mode runs. The feature is selectable: --indicator log_return
(default) | rsi | cci | willr | mfi | cmf | obv_vel (OBV EMA-velocity z-score).
"""
import pandas as pd

from _common import OUT_ROOT, build_parser, load_panel, ms


def obv_velocity(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """OBV EMA-velocity, rolling z-scored (volume.py's zvel construction)."""
    from findata.preprocess.calculators.technical import ema, obv
    vel = ema(obv(df), fast) - ema(obv(df), slow)
    z = (vel - vel.rolling(slow * 2).mean()) / vel.rolling(slow * 2).std()
    return z.rename("obv_vel")


CUSTOM_INDICATORS = {"obv_vel": obv_velocity}
TRANSFORM_PLOT_LAYERS = ("raw", "kalman", "minus_global_pc1", "minus_sector_pc1")
COMPONENT_PLOT_LAYERS = ("raw", "combined_modes", "residual")


def parse_args():
    p = build_parser(__doc__)
    p.add_argument("--n", type=int, default=4, help="tickers to sample for display")
    p.add_argument("--sector", default=None, help="restrict the sample to one sector")
    p.add_argument("--mode", choices=("interactive", "static"), default="interactive")
    return p.parse_args()


def main():
    args = parse_args()
    out_name = args.indicator

    # load the FULL panel — modes must be estimated on the whole universe, the
    # sample is only what gets displayed/saved
    custom = CUSTOM_INDICATORS.get(args.indicator)
    if custom is not None:
        from findata import get_all_tickers
        tickers = get_all_tickers()
        if args.tickers:
            tickers = tickers[:args.tickers]
        wide, meta = ms.indicator_panel(tickers, start=args.start, end=args.end,
                                        indicator=custom)
        print(f"panel: T={len(wide)} x N={wide.shape[1]} ({out_name})")
    else:
        wide, meta = load_panel(args)
    labels = meta["sector"]

    decomp = ms.decompose_modes(wide, labels)
    tlayers = ms.transform_layers(wide, labels)
    layers = {**tlayers, **{k: v for k, v in decomp["panels"].items() if k != "raw"}}

    sample = ms.sample_tickers(meta, n=args.n, sector=args.sector, seed=args.seed)
    print("sample:", {t: labels.get(t) for t in sample})

    out = OUT_ROOT / "reward_structure" / out_name
    out.mkdir(parents=True, exist_ok=True)

    # ---- data is ALWAYS saved ------------------------------------------- #
    for name, frame in layers.items():
        frame[sample].to_parquet(out / f"layer_{name}.parquet")
    decomp["modes"].to_parquet(out / "modes.parquet")
    decomp["loadings"].to_csv(out / "loadings.csv")
    meta.loc[sample].to_csv(out / "sample_meta.csv")
    print(f"data saved -> {out}")

    if args.mode == "static":
        from findata.analysis import save_layer_plot, save_modes_plot

        save_layer_plot({"raw": layers["raw"]}, sample,
                        out / "1_reward_structure_raw.png",
                        title=f"Sampled reward structure — cumulative {out_name} (standardized)")
        save_layer_plot({k: layers[k] for k in TRANSFORM_PLOT_LAYERS}, sample,
                        out / "2_transform_comparison.png",
                        title="Transform comparison (color = transform, shade/dash = ticker)")
        save_layer_plot({k: (layers[k] if k in layers else decomp["panels"][k])
                         for k in COMPONENT_PLOT_LAYERS}, sample,
                        out / "3_decomposition_components.png",
                        title="Decomposition: raw vs combined modes vs residual")
        for sector in sorted({labels.get(t) for t in sample if pd.notna(labels.get(t))}):
            members = [t for t in sample if labels.get(t) == sector]
            save_modes_plot(decomp["modes"], sector, decomp["panels"]["residual"],
                            members, out / f"4_modes_{sector.replace(' ', '_')}.png")
        print(f"\nstatic plots -> {out}")
    else:
        from findata.analysis import TransformExplorer

        explorer = TransformExplorer(
            layers={name: frame[sample] for name, frame in layers.items()},
            modes=decomp["modes"], tickers=sample)
        explorer.serve()


if __name__ == "__main__":
    main()

