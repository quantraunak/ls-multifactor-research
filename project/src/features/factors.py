from __future__ import annotations

import numpy as np
import pandas as pd


def _shift_by_calendar_days(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")
    target_dates = prices.index - pd.Timedelta(days=days)
    shifted = prices.reindex(target_dates, method="ffill")
    shifted.index = prices.index
    return shifted


def _momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    price_t_minus_21 = _shift_by_calendar_days(prices, 21)
    price_t_minus_252 = _shift_by_calendar_days(prices, 252)
    return price_t_minus_21 / price_t_minus_252 - 1.0


def _momentum_n_1(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    price_t_minus_21 = _shift_by_calendar_days(prices, 21)
    price_t_minus_n = _shift_by_calendar_days(prices, lookback_days)
    return price_t_minus_21 / price_t_minus_n - 1.0


def compute_factors(prices: pd.DataFrame, volumes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prices = prices.sort_index()
    returns = prices.pct_change()

    mom_12_1 = _momentum_12_1(prices)
    mom_6_1 = _momentum_n_1(prices, 126)
    mom_3_1 = _momentum_n_1(prices, 63)

    low_vol = -returns.rolling(252).std()

    trend = prices / prices.rolling(200).mean() - 1.0

    dollar_volume = prices * volumes
    liquidity = np.log1p(dollar_volume.rolling(20).mean())

    rolling_mean = returns.rolling(252).mean()
    rolling_std = returns.rolling(252).std()
    quality_proxy = rolling_mean / rolling_std.replace(0.0, np.nan)

    return {
        "mom_12_1": mom_12_1,
        "mom_6_1": mom_6_1,
        "mom_3_1": mom_3_1,
        "low_vol": low_vol,
        "trend": trend,
        "liquidity": liquidity,
        "quality_proxy": quality_proxy,
    }

