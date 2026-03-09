# Multi-Factor Long/Short U.S. Equity Strategy

A realistic, local, reproducible multi-factor long/short US equity backtesting framework.

- Dollar-neutral, monthly rebalance
- Walk-forward training, no lookahead bias
- Mean-variance optimizer with beta-neutrality and turnover penalty
- Free data only (Yahoo Finance default; Stooq optional)

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

## Data Limitations (Important)

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
- Model trained on a rolling window (`train_lookback_months`)
- Portfolio weights fixed between rebalances
- Transaction costs applied on turnover at rebalance

## Constraints

- Dollar neutral: sum(w) = 0
- Gross leverage <= L
- Max |w_i| <= w_max
- Beta neutrality vs SPY (tolerance)
- Turnover penalty + linear costs in bps

## Configuration

Edit `project/configs/default.yaml` to adjust:
- start/end dates
- quantiles, leverage, max weights
- model type (ridge/elasticnet)
- cost assumptions

## End-to-End Run

```bash
python project/run_backtest.py --config project/configs/default.yaml
```

## Sample Outputs

See:
- `project/reports/latest/<timestamp>/tear_sheet.html`
- `project/reports/latest/<timestamp>/equity_curve.png`
- `project/reports/latest/<timestamp>/drawdown.png`
