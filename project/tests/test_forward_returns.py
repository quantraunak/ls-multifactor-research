import pandas as pd

from src.labels.forward_returns import compute_forward_returns


def test_forward_returns_alignment():
    prices = pd.DataFrame(
        {"AAA": [100, 110, 121]}, index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    )
    fwd = compute_forward_returns(prices, horizon_days=1)
    assert abs(fwd.loc["2020-01-01", "AAA"] - 0.10) < 1e-9
    assert abs(fwd.loc["2020-01-02", "AAA"] - 0.10) < 1e-9
    assert pd.isna(fwd.loc["2020-01-03", "AAA"])

