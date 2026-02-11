from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.features.factors import compute_factors
from src.features.preprocess import build_feature_matrix
from src.labels.forward_returns import compute_forward_returns
from src.models.linear_ranker import LinearRanker
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.costs import turnover_cost
from src.portfolio.optimizer import OptimizerConfig, optimize_weights


@dataclass
class BacktestConfig:
    start: str
    end: str
    rebalance_freq: str
    train_lookback_months: int
    horizon_days: int
    long_short_quantile: float
    beta_window: int
    cov_window: int
    min_train_samples: int
    risk_aversion: float
    turnover_penalty: float
    commission_bps: float
    slippage_bps: float
    max_weight: float
    gross_leverage: float
    beta_tolerance: float
    solver: str | None
    normalize: str
    winsorize_limits: tuple[float, float]
    model_type: str
    model_alpha: float
    model_l1_ratio: float


def _estimate_betas(returns: pd.DataFrame, market: pd.Series) -> pd.Series:
    aligned = returns.join(market.rename("market"), how="inner")
    if aligned.empty:
        return pd.Series(0.0, index=returns.columns)
    market = aligned["market"]
    betas = {}
    for col in returns.columns:
        series = aligned[col].dropna()
        common = series.index.intersection(market.dropna().index)
        if common.empty:
            betas[col] = 0.0
            continue
        cov = series.loc[common].cov(market.loc[common])
        var = market.loc[common].var()
        if var == 0.0 or pd.isna(var):
            betas[col] = 0.0
        else:
            betas[col] = cov / var
    return pd.Series(betas)


def run_backtest(prices: pd.DataFrame, volumes: pd.DataFrame, config: BacktestConfig) -> Dict:
    prices = prices.loc[config.start : config.end].dropna(how="all", axis=1)
    volumes = volumes.loc[config.start : config.end].reindex(prices.index)

    returns = prices.pct_change()

    factors = compute_factors(prices, volumes)
    features = build_feature_matrix(
        factors,
        winsorize_limits=config.winsorize_limits,
        normalize=config.normalize,
    )

    fwd = compute_forward_returns(prices, config.horizon_days).stack()
    fwd.index.names = ["date", "ticker"]

    rebal_dates = prices.resample(config.rebalance_freq).last().index
    rebal_dates = rebal_dates[(rebal_dates >= prices.index.min()) & (rebal_dates <= prices.index.max())]

    holdings = {}
    daily_port_returns = []
    turnovers = []

    prev_weights = pd.Series(dtype=float)

    for i, dt in enumerate(rebal_dates[:-1]):
        next_dt = rebal_dates[i + 1]
        train_start = dt - pd.DateOffset(months=config.train_lookback_months)

        train_mask = (features.index.get_level_values(0) >= train_start) & (
            features.index.get_level_values(0) < dt
        )
        test_mask = features.index.get_level_values(0) == dt

        X_train = features.loc[train_mask]
        y_train = fwd.reindex(X_train.index).dropna()

        X_train = X_train.loc[y_train.index]

        X_test = features.loc[test_mask]
        if X_test.empty:
            continue

        if len(y_train) < config.min_train_samples:
            pred = X_test.mean(axis=1)
        else:
            model = LinearRanker(
                model_type=config.model_type,
                alpha=config.model_alpha,
                l1_ratio=config.model_l1_ratio,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        pred = pred.dropna()
        if pred.empty:
            continue

        n = len(pred)
        q = max(1, int(n * config.long_short_quantile))
        longs = pred.nlargest(q)
        shorts = pred.nsmallest(q)
        selected = pd.concat([longs, shorts])

        # Flatten MultiIndex (date, ticker) -> ticker-only index
        if isinstance(selected.index, pd.MultiIndex):
            ticker_idx = selected.index.get_level_values("ticker")
            selected = selected.copy()
            selected.index = ticker_idx

        # Deduplicate (a ticker may appear in both nlargest and nsmallest for tiny n)
        selected = selected[~selected.index.duplicated(keep="first")]

        sel_tickers = selected.index.tolist()

        ret_window = returns.loc[:dt].iloc[-config.cov_window:]
        cov = ret_window[sel_tickers].cov()

        market = returns["SPY"].loc[:dt].iloc[-config.beta_window:]
        asset_returns = returns[sel_tickers].loc[:dt].iloc[-config.beta_window:]
        betas = _estimate_betas(asset_returns, market)

        constraints = PortfolioConstraints(
            gross_leverage=config.gross_leverage,
            max_weight=config.max_weight,
            beta_tolerance=config.beta_tolerance,
        )

        opt_config = OptimizerConfig(
            risk_aversion=config.risk_aversion,
            turnover_penalty=config.turnover_penalty,
            solver=config.solver,
        )

        prev_subset = prev_weights.reindex(sel_tickers).fillna(0.0)
        w = optimize_weights(selected, cov, betas, prev_subset, constraints, opt_config)

        holdings[dt] = w
        turnover = (w - prev_subset).abs().sum()
        turnovers.append(float(turnover))

        cost = turnover_cost(prev_subset, w, config.commission_bps, config.slippage_bps)

        pnl_window = returns.loc[dt:next_dt].iloc[1:]
        if pnl_window.empty:
            continue

        pnl = pnl_window[sel_tickers].mul(w, axis=1).sum(axis=1)
        pnl.iloc[0] -= cost
        daily_port_returns.append(pnl)

        prev_weights = w

    if daily_port_returns:
        daily_returns = pd.concat(daily_port_returns).sort_index()
    else:
        daily_returns = pd.Series(dtype=float)

    holdings_df = pd.DataFrame(holdings).T.sort_index()
    return {
        "daily_returns": daily_returns,
        "holdings": holdings_df,
        "turnovers": turnovers,
    }

