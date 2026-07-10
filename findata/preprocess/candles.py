"""Candle feature group — feature engineering ported from v1.

One fix vs v1: ``gap_log`` uses the previous close **per ticker**. v1 computed
``C.shift(1)`` on the stacked (ticker, date) frame, so each ticker's first row
gapped against another ticker's last close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from findata.preprocess.base import FeatureGroup, ScaleRule


class Candle(FeatureGroup):
    """
    Features:
        1. body_size: |C-O|/(H-L)   (bounded [0, 1])
        2. body_log:  log(C/O)
        3. range_log: log(H/L)
        4. skew_log:  log(H/max(O,C)) - log(min(O,C)/L)
        5. gap_log:   log(O/prev_close)
    Scaling: log features robust + arcsinh; body_size mapped to [-1, 1].
    """
    name = "candle"
    default_scaler = "robust"
    default_arcsinh = True
    eps = 1e-7

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        O, H, L, C = data["open"], data["high"], data["low"], data["close"]
        R = (H - L) + self.eps
        body = C - O
        prev_close = C.groupby(level="ticker").shift(1)
        return pd.DataFrame(
            {
                'body_size': (body / R).abs(),
                "body_log": np.log(C / O),
                "range_log": np.log(H / L),
                'skew_log': np.log(H / np.maximum(O, C)) - np.log(np.minimum(O, C) / L),
                'gap_log': np.log(O / prev_close),
            },
            index=data.index,
        )

    def scale_rules(self, columns) -> list[ScaleRule]:
        columns = list(columns)
        log_cols = [c for c in columns if 'log' in c]
        bounded = [c for c in columns if 'log' not in c]
        rules = [ScaleRule(columns=log_cols, scaler=self.default_scaler, arcsinh=True)]
        if bounded:
            rules.append(ScaleRule(columns=bounded, kind="affine", mul=2.0, add=-1.0))
        return rules
