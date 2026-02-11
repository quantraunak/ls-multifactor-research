import numpy as np
import pandas as pd

from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.optimizer import OptimizerConfig, optimize_weights


def test_optimizer_neutrality():
    tickers = ["A", "B", "C", "D"]
    mu = pd.Series([0.05, 0.03, -0.02, -0.04], index=tickers)
    cov = pd.DataFrame(np.eye(4) * 0.05, index=tickers, columns=tickers)
    betas = pd.Series([1.1, 0.9, 1.0, 1.2], index=tickers)

    constraints = PortfolioConstraints(gross_leverage=2.0, max_weight=0.5, beta_tolerance=0.05)
    cfg = OptimizerConfig(risk_aversion=1.0, turnover_penalty=0.0)

    w = optimize_weights(mu, cov, betas, None, constraints, cfg)

    assert abs(w.sum()) < 1e-6
    assert (w.abs() <= constraints.max_weight + 1e-6).all()
    assert abs((betas * w).sum()) <= constraints.beta_tolerance + 1e-3

