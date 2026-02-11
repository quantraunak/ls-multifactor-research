from __future__ import annotations

import pandas as pd


def compute_forward_returns(prices: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    fwd = prices.shift(-horizon_days) / prices - 1.0
    return fwd

