from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    total = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return total ** (1 / years) - 1 if years > 0 else 0.0


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    return (returns.mean() / downside.std()) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd.min()


def hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return (returns > 0).mean()


def summarize(returns: pd.Series, turnovers: list[float], holdings: pd.DataFrame) -> dict:
    return {
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "avg_gross_leverage": float(holdings.abs().sum(axis=1).mean()) if not holdings.empty else 0.0,
        "avg_net_exposure": float(holdings.sum(axis=1).mean()) if not holdings.empty else 0.0,
    }

