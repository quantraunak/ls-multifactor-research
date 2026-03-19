# Multi-Factor Long/Short U.S. Equity Strategy

Systematic long/short equity backtest: walk-forward factor ranking, mean-variance optimization, dollar- and beta-neutral construction.

## Results

| Metric | Value |
|--------|-------|
| CAGR | — |
| Sharpe | — |
| Max Drawdown | — |
| Volatility (ann.) | — |
| Avg Turnover | — |
| Beta (vs SPY) | — |

> 2010–2024 · Monthly rebalance · 2× gross leverage · Dollar-neutral

![Equity Curve](project/reports/latest/equity_curve.png)
![Drawdown](project/reports/latest/drawdown.png)

## Why This Is Credible

- **No lookahead bias.** Walk-forward training — the model only sees data available before each rebalance date.
- **Monthly rebalancing.** Weights fixed between rebalance dates; no daily re-optimization.
- **Transaction costs included.** Commission (1 bp) and slippage (2 bp) deducted at every rebalance.
- **Dollar-neutral and beta-neutral.** Zero net exposure, near-zero beta vs SPY.
- **Realistic portfolio construction.** Mean-variance optimizer with per-name caps, leverage limits, and turnover penalty.

## Strategy Overview

A linear model ranks stocks cross-sectionally on momentum, volatility, trend, and liquidity factors derived from price and volume. Top and bottom deciles are passed to a constrained optimizer that builds a dollar-neutral, beta-neutral portfolio rebalanced monthly.

**Universe.** Russell 1000 PIT membership is not freely available. The system uses `universe.csv` if provided, otherwise falls back to an S&P 500 scrape or a static large-cap list. See Key Limitations.

**Factors.** `mom_12_1`, `mom_6_1`, `mom_3_1`, `low_vol`, `trend`, `liquidity`, `quality_proxy` — all from price and volume.

**Model.** Ridge regression (configurable to ElasticNet), trained on rolling 36-month windows with 21-day forward returns as labels.

## Methodology

**Rebalancing.** Monthly, last trading day. Weights held fixed until the next rebalance.

**Training.** At each rebalance date, the model trains on the trailing 36 months of cross-sectional features and forward returns. No future data enters training.

**Selection.** Model scores all tickers; top and bottom 10% form the long/short baskets.

**Optimization.** CVXPY mean-variance (CLARABEL solver) with:
- `sum(w) = 0` — dollar-neutral
- `sum(|w|) ≤ 2.0` — gross leverage cap
- `|w_i| ≤ 0.02` — per-name cap
- `|β'w| ≤ 0.05` — beta-neutral vs SPY
- Explicit turnover penalty in the objective

Equal-weight fallback if the solver fails.

**Costs.** 1 bp commission + 2 bp slippage, applied proportionally to turnover at each rebalance.

## Key Limitations

- **Survivorship-biased universe.** PIT Russell 1000 membership is not freely available. The proxy overstates investable universe quality.
- **Price/volume factors only.** No fundamental data (earnings, book value, ROE) enters the production signal.
- **Sample covariance without shrinkage.** Trailing 252-day sample covariance with no shrinkage or factor structure; noisy for large universes.
- **Simplified execution model.** Fixed-bps cost with no market-impact function or dependence on position size or ADV.

## Next Steps

1. **Point-in-time universe.** Replace the survivorship-biased proxy with true historical Russell 1000 membership.
2. **Improved risk model.** Covariance shrinkage (Ledoit-Wolf) or a statistical factor model to stabilize optimization.
3. **Paper trading / live pipeline.** Wire signal and optimizer to a broker API for forward validation.

## How to Run

```bash
make install    # install dependencies
make test       # run tests
make run        # run full backtest
```

Or directly:

```bash
python project/run_backtest.py --config project/configs/default.yaml
```

Outputs go to `project/reports/latest/<timestamp>/`: equity curve, holdings, performance summary, tear sheet, factor IC analysis.

Configuration: edit `project/configs/default.yaml` (dates, leverage, quantiles, model type, costs).
