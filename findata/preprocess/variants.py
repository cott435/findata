"""Named end-to-end processing variants for the forecasting project.

Each entry is a dataset the forecasting hyperparameter search can select via
its ``data.dataset`` axis (see forecasting/search/space.py). ``None`` means
scale-only (build_features' default). All mode variants need ``ticker_meta``
passed to :func:`findata.build_features`.

    base          scale-only features (control)
    global_resid  global market mode removed, residual transformed back
    resid         global + sector modes removed, residual transformed back
    resid_comp    residual PLUS per-ticker component columns
                  ('{col}_gmode', '{col}_smode') — nothing discarded
    resid_white   residual, then ticker-space ZCA on the MP-denoised
                  residual correlation ("whitening after denoising")
"""
from __future__ import annotations

from findata.preprocess.transforms import ProcessingConfig


def forecast_variants() -> dict[str, ProcessingConfig | None]:
    return {
        "base": None,
        "global_resid": ProcessingConfig(
            "global_resid", steps=[("modes", {"group_stage": False})]),
        "resid": ProcessingConfig(
            "resid", steps=[("modes", {})]),
        "resid_comp": ProcessingConfig(
            "resid_comp", steps=[("modes", {"output": "residual+components"})]),
        "resid_white": ProcessingConfig(
            "resid_white", steps=[("modes", {}), ("cs_whiten", {})]),
    }
