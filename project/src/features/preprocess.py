from __future__ import annotations

import numpy as np
import pandas as pd


def _winsorize(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    q_low = df.quantile(lower, axis=1)
    q_high = df.quantile(upper, axis=1)
    return df.clip(q_low, q_high, axis=0)


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0.0, np.nan)
    return (df.sub(mean, axis=0)).div(std, axis=0)


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True) - 0.5


def build_feature_matrix(
    factors: dict[str, pd.DataFrame],
    winsorize_limits: tuple[float, float] = (0.01, 0.99),
    normalize: str = "zscore",
) -> pd.DataFrame:
    frames = []
    for name, df in factors.items():
        x = df.copy()
        if winsorize_limits is not None:
            x = _winsorize(x, *winsorize_limits)
        if normalize == "zscore":
            x = _zscore(x)
        elif normalize == "rank":
            x = _rank(x)
        stacked = x.stack(dropna=False).rename(name)
        frames.append(stacked)

    features = pd.concat(frames, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    features.index.names = ["date", "ticker"]
    return features

