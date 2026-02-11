"""Robust price downloader with per-ticker isolation, stooq/yfinance fallback,
exponential-backoff retries, parquet caching, and a manifest log."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FetchConfig:
    source: str = "auto"            # "stooq" | "yfinance" | "auto"
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    force_refresh: bool = False
    min_history_days: int = 252
    min_tickers: int = 20
    retries: int = 3


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class PriceData:
    prices: pd.DataFrame   # index: date, columns: ticker (adj close)
    volumes: pd.DataFrame  # index: date, columns: ticker


# ---------------------------------------------------------------------------
# Stooq helpers
# ---------------------------------------------------------------------------

_STOOQ_TICKER_VARIANTS = {}  # runtime cache: canonical ticker -> working stooq sym


def _stooq_variants(ticker: str) -> List[str]:
    """Return candidate Stooq symbols for a US equity ticker."""
    base = ticker.upper()
    candidates = [f"{base}.US"]
    # BRK-B  ->  BRK.B.US  (Stooq convention for share classes)
    if "-" in base:
        candidates.append(f"{base.replace('-', '.')}.US")
    return candidates


def _download_stooq_single(
    ticker: str, start: str, end: str, retries: int
) -> Optional[pd.DataFrame]:
    from pandas_datareader import data as pdr

    # If we already know the working symbol, try that first
    variants = _stooq_variants(ticker)
    if ticker in _STOOQ_TICKER_VARIANTS:
        variants = [_STOOQ_TICKER_VARIANTS[ticker]] + [
            v for v in variants if v != _STOOQ_TICKER_VARIANTS[ticker]
        ]

    last_err: Optional[Exception] = None
    for sym in variants:
        for attempt in range(retries):
            try:
                df = pdr.DataReader(sym, "stooq", start, end)
                if df is not None and not df.empty:
                    df = df.sort_index()
                    df = df.rename(columns={
                        "Close": "adj_close",
                        "Volume": "volume",
                    })
                    df = df[["adj_close", "volume"]].copy()
                    df.index.name = "date"
                    _STOOQ_TICKER_VARIANTS[ticker] = sym
                    return df
            except Exception as exc:
                last_err = exc
                time.sleep(0.5 * (2 ** attempt))
    return None


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _download_yfinance_single(
    ticker: str, start: str, end: str, retries: int
) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        return None

    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                # yfinance may return MultiIndex columns for single ticker
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns={"Close": "adj_close", "Volume": "volume"})
                if "adj_close" not in df.columns and "Adj Close" in df.columns:
                    df = df.rename(columns={"Adj Close": "adj_close"})
                cols = [c for c in ["adj_close", "volume"] if c in df.columns]
                if "adj_close" not in cols:
                    return None
                df = df[cols].copy()
                df.index.name = "date"
                return df
        except Exception:
            time.sleep(0.5 * (2 ** attempt))
    return None


# ---------------------------------------------------------------------------
# Per-ticker cache
# ---------------------------------------------------------------------------

def _ticker_cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / "prices" / f"{ticker}.parquet"


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "prices" / "_manifest.json"


def _load_manifest(cache_dir: Path) -> dict:
    mp = _manifest_path(cache_dir)
    if mp.exists():
        return json.loads(mp.read_text())
    return {}


def _save_manifest(cache_dir: Path, manifest: dict):
    mp = _manifest_path(cache_dir)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_price_data(
    tickers: Iterable[str],
    cfg: FetchConfig,
    cache_dir: Path,
) -> PriceData:
    """Download / load cached prices for *tickers*.  Returns PriceData with
    adj-close and volume wide DataFrames."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    prices_dir = cache_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    tickers = sorted(set(t.upper().strip() for t in tickers if t))
    manifest = _load_manifest(cache_dir)

    succeeded: list[str] = []
    failed: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}

    for t in tickers:
        # --- try cache first ---
        cp = _ticker_cache_path(cache_dir, t)
        if cp.exists() and not cfg.force_refresh:
            try:
                df = pd.read_parquet(cp)
                if not df.empty:
                    frames[t] = df
                    succeeded.append(t)
                    continue
            except Exception:
                pass  # re-download

        # --- download ---
        df: Optional[pd.DataFrame] = None
        source_used: Optional[str] = None

        if cfg.source in ("stooq", "auto"):
            df = _download_stooq_single(t, cfg.start, cfg.end, cfg.retries)
            if df is not None:
                source_used = "stooq"

        if df is None and cfg.source in ("yfinance", "auto"):
            df = _download_yfinance_single(t, cfg.start, cfg.end, cfg.retries)
            if df is not None:
                source_used = "yfinance"

        if df is not None and not df.empty:
            df.to_parquet(cp)
            manifest[t] = {
                "source": source_used,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            frames[t] = df
            succeeded.append(t)
        else:
            failed[t] = "empty or no data from any source"

    _save_manifest(cache_dir, manifest)

    # --- assemble wide frames ---
    price_parts = []
    volume_parts = []
    for t, df in frames.items():
        df.index = pd.to_datetime(df.index)
        if "adj_close" in df.columns:
            price_parts.append(df["adj_close"].rename(t))
        if "volume" in df.columns:
            volume_parts.append(df["volume"].rename(t))

    if price_parts:
        prices = pd.concat(price_parts, axis=1).sort_index()
    else:
        prices = pd.DataFrame()

    if volume_parts:
        volumes = pd.concat(volume_parts, axis=1).sort_index()
    else:
        volumes = pd.DataFrame()

    # --- filter insufficient history ---
    dropped_history: list[str] = []
    if not prices.empty:
        counts = prices.count()
        short = counts[counts < cfg.min_history_days].index.tolist()
        if short:
            prices = prices.drop(columns=short)
            volumes = volumes.drop(columns=[c for c in short if c in volumes.columns])
            dropped_history = short

    # --- summary log ---
    print(f"[fetch_prices] attempted={len(tickers)}  succeeded={len(succeeded)}  "
          f"failed={len(failed)}  dropped_short_history={len(dropped_history)}")
    if failed:
        shown = list(failed.items())[:10]
        for t, msg in shown:
            print(f"  FAIL {t}: {msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more failures")

    # --- final checks ---
    remaining = prices.shape[1] if not prices.empty else 0
    if remaining < cfg.min_tickers:
        raise RuntimeError(
            f"Only {remaining} tickers survived (min_tickers={cfg.min_tickers}). "
            f"Check network or universe list."
        )

    return PriceData(prices=prices, volumes=volumes)
