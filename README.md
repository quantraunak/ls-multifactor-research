# Multi-Factor Long/Short U.S. Equity Strategy

## Results

| Metric | Value |
|--------|-------|
| CAGR | — |
| Sharpe | — |
| Max Drawdown | — |
| Volatility (ann.) | — |
| Avg Turnover | — |
| Beta (vs SPY) | — |

> Backtest period: 2010-01-01 to 2024-12-31. Monthly rebalance, 2× gross leverage, dollar-neutral.
> Run `make run` to generate results and update this table.

### Equity Curve

![Equity Curve](project/reports/latest/equity_curve.png)

### Drawdown

![Drawdown](project/reports/latest/drawdown.png)

## Why This Is Credible

- **No lookahead bias.** Walk-forward training — the model only sees data available before each rebalance date.
- **Monthly rebalancing.** Weights are fixed between rebalance dates; no daily re-optimization or curve-fitting.
- **Transaction costs included.** Commission (1 bp) and slippage (2 bp) are deducted on every turnover event.
- **Dollar-neutral and beta-neutral.** Portfolio is constrained to zero net exposure and near-zero beta vs SPY.
- **Realistic portfolio construction.** Mean-variance optimizer with per-name caps, gross leverage limits, and an explicit turnover penalty.

---

## Quick Start

```bash
make install
make test
make run
```

Outputs are saved under:
```
project/reports/latest/<timestamp>/
```
including:
- `equity_curve.csv`
- `holdings.csv`
- `performance_summary.json`
- `tear_sheet.html`
- `equity_curve.png`, `drawdown.png`
- `factor_ic.csv`, `factor_ic_summary.json`, `factor_ic.png`

## Data Limitations

Russell 1000 membership is **not freely available point-in-time**, so this repo uses a
**survivorship-biased proxy**:
- If `project/data/raw/universe.csv` exists, it is used directly.
- Otherwise, it scrapes the current S&P 500 list from Wikipedia (fallback).
- If scraping fails, a static large-cap list is used.

To use a true point-in-time universe, replace `project/data/raw/universe.csv` with your own
PIT universe history and update `src/data/universe.py`.

## Walk-Forward Backtest Design

- Factors computed only with data available up to each rebalance date
- Forward returns for training are aligned after the feature date
- Model trained on a rolling 36-month window
- Portfolio weights fixed between rebalances
- Transaction costs applied on turnover at rebalance

## Constraints

- Dollar neutral: `sum(w) = 0`
- Gross leverage: `sum(|w|) ≤ L`
- Max single-name weight: `|w_i| ≤ 0.02`
- Beta neutrality vs SPY: `|β'w| ≤ 0.05`
- Turnover penalty + linear costs (1 bp commission, 2 bp slippage)

## Key Limitations

- **Survivorship-biased universe.** True point-in-time Russell 1000 membership is not freely available. The backtest uses a proxy (S&P 500 scrape or static large-cap list), which overstates the investable universe quality.
- **Price/volume factors only.** The production pipeline derives all factors from price and volume data. No fundamental inputs (earnings, book value, ROE) enter the live signal.
- **Sample covariance without shrinkage.** The optimizer uses a trailing 252-day sample covariance matrix with no shrinkage estimator or factor-model structure, which can be noisy for large universes.
- **Simplified execution model.** Transaction costs are modeled as fixed basis points on turnover (1 bp commission + 2 bp slippage), with no market-impact function or dependence on position size or liquidity.

## Configuration

Edit `project/configs/default.yaml` to adjust:
- Start/end dates
- Quantiles, leverage, max weights
- Model type (ridge / elasticnet)
- Cost assumptions

## Next Steps

1. **Point-in-time universe construction.** Replace the survivorship-biased proxy with true historical Russell 1000 membership.
2. **Improved risk model.** Add covariance shrinkage (Ledoit-Wolf) or a statistical factor model to stabilize the optimizer.
3. **Paper trading / live pipeline.** Wire the signal and optimizer to a broker API for forward validation.

## End-to-End Run

```bash
python project/run_backtest.py --config project/configs/default.yaml
```
