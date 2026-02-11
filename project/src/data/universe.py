from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

_STATIC_LARGE_CAPS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "BRK-B",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "PG",
    "HD",
    "MA",
    "LLY",
    "COST",
    "AVGO",
    "MRK",
    "CVX",
    "ABBV",
    "PEP",
    "KO",
    "ADBE",
    "CSCO",
    "PFE",
    "TMO",
    "WMT",
    "BAC",
    "NFLX",
    "CRM",
    "CMCSA",
    "ORCL",
    "ABT",
    "MCD",
    "QCOM",
    "NKE",
    "LIN",
    "TXN",
    "INTC",
    "AMD",
    "UPS",
    "LOW",
    "NEE",
    "PM",
    "RTX",
    "MS",
    "HON",
    "INTU",
    "UNP",
    "AMGN",
]


def _scrape_sp500() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return tickers


def load_universe(universe_path: Path, fallback: str = "sp500") -> List[str]:
    if universe_path.exists():
        df = pd.read_csv(universe_path)
        if "ticker" in df.columns:
            tickers = df["ticker"].astype(str).tolist()
        else:
            tickers = df.iloc[:, 0].astype(str).tolist()
        return sorted(set([t.upper().strip() for t in tickers if t]))

    if fallback == "sp500":
        try:
            return _scrape_sp500()
        except Exception:
            return _STATIC_LARGE_CAPS.copy()

    return _STATIC_LARGE_CAPS.copy()

