from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class OptimizerConfig:
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.1
    solver: str | None = "CLARABEL"


def optimize_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    betas: pd.Series,
    prev_weights: Optional[pd.Series],
    constraints,
    config: OptimizerConfig,
) -> pd.Series:
    mu = mu.dropna()

    # Defensive: if mu has a MultiIndex, keep only the ticker level
    if isinstance(mu.index, pd.MultiIndex):
        mu = mu.copy()
        mu.index = mu.index.get_level_values(-1)
        mu = mu[~mu.index.duplicated(keep="first")]

    tickers = mu.index.tolist()

    # Ensure tickers are plain strings (guard against tuple indices)
    if tickers and isinstance(tickers[0], tuple):
        tickers = [t[-1] if isinstance(t, tuple) else t for t in tickers]
        mu.index = tickers

    cov = cov.loc[tickers, tickers]
    betas = betas.reindex(tickers).fillna(0.0)

    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)

    cov = cov.fillna(0.0).values
    cov = cov + np.eye(n) * 1e-6

    w = cp.Variable(n)
    # Auxiliary variable t_i >= |w_i| to express gross leverage as a linear sum
    t = cp.Variable(n, nonneg=True)

    obj = mu.values @ w - config.risk_aversion * cp.quad_form(w, cov)

    if prev_weights is not None and len(prev_weights) == n:
        obj -= config.turnover_penalty * cp.norm1(w - prev_weights.values)

    L = constraints.gross_leverage
    cons = [
        cp.sum(w) == 0.0,
        t >= w,                                       # t_i >= w_i
        t >= -w,                                      # t_i >= -w_i  =>  t_i >= |w_i|
        cp.sum(t) >= 0.98 * L,                        # min deployment
        cp.sum(t) <= L,                               # max gross leverage
        t <= constraints.max_weight,                   # per-name cap
        cp.abs(betas.values @ w) <= constraints.beta_tolerance,
    ]

    problem = cp.Problem(cp.Maximize(obj), cons)
    try:
        if config.solver:
            problem.solve(solver=config.solver, verbose=False)
        else:
            problem.solve(verbose=False)
    except Exception:
        problem.solve(verbose=False)

    if w.value is None:
        half = n // 2
        out = np.zeros(n)
        if half > 0:
            out[:half] = 1.0 / half
            out[half:] = -1.0 / (n - half)
        return pd.Series(out, index=tickers)

    return pd.Series(w.value, index=tickers)
