import pandas as pd

from src.portfolio.optimizer import OptimizerConfig, optimize_weights
from src.portfolio.constraints import PortfolioConstraints
import numpy as np


def test_multiindex_score_yields_ticker_only():
    """Scores with a (date, ticker) MultiIndex should be flattened to ticker-only
    before reaching the optimizer."""
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    dt = pd.Timestamp("2020-06-30")
    idx = pd.MultiIndex.from_tuples([(dt, t) for t in tickers], names=["date", "ticker"])
    scores = pd.Series([0.05, 0.03, -0.02, -0.04], index=idx)

    # Simulate the engine flatten logic
    if isinstance(scores.index, pd.MultiIndex):
        ticker_idx = scores.index.get_level_values("ticker")
        scores = scores.copy()
        scores.index = ticker_idx

    assert list(scores.index) == tickers
    assert not isinstance(scores.index, pd.MultiIndex)


def test_optimizer_handles_multiindex_mu():
    """optimize_weights should work even if mu accidentally still has a MultiIndex."""
    tickers = ["A", "B", "C", "D"]
    dt = pd.Timestamp("2020-06-30")
    idx = pd.MultiIndex.from_tuples([(dt, t) for t in tickers], names=["date", "ticker"])
    mu = pd.Series([0.05, 0.03, -0.02, -0.04], index=idx)

    cov = pd.DataFrame(np.eye(4) * 0.05, index=tickers, columns=tickers)
    betas = pd.Series([1.0, 1.0, 1.0, 1.0], index=tickers)

    constraints = PortfolioConstraints(gross_leverage=2.0, max_weight=0.5, beta_tolerance=0.05)
    cfg = OptimizerConfig(risk_aversion=1.0, turnover_penalty=0.0)

    w = optimize_weights(mu, cov, betas, None, constraints, cfg)

    # Should return plain ticker index
    assert not isinstance(w.index, pd.MultiIndex)
    assert set(w.index) == set(tickers)
    assert abs(w.sum()) < 1e-6

