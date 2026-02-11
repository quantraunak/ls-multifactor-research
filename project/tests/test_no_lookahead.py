import pandas as pd

from src.features.factors import compute_factors


def test_no_lookahead_on_momentum():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    prices = pd.DataFrame({"AAA": range(300)}, index=idx).astype(float)
    vols = pd.DataFrame({"AAA": 1000.0}, index=idx)

    factors = compute_factors(prices, vols)
    base = factors["mom_12_1"].loc[idx[250], "AAA"]

    prices.loc[idx[299], "AAA"] = 1e9
    factors2 = compute_factors(prices, vols)
    mutated = factors2["mom_12_1"].loc[idx[250], "AAA"]

    assert base == mutated

