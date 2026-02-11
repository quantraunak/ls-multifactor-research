from __future__ import annotations

import pandas as pd


def turnover_cost(prev_w: pd.Series, new_w: pd.Series, commission_bps: float, slippage_bps: float) -> float:
    prev_w = prev_w.reindex(new_w.index).fillna(0.0)
    turnover = (new_w - prev_w).abs().sum()
    total_bps = commission_bps + slippage_bps
    return turnover * total_bps / 10000.0

