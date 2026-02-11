from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_tear_sheet(returns: pd.Series, output_dir: Path, benchmark: pd.Series | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import quantstats as qs

        qs.reports.html(
            returns,
            benchmark=benchmark,
            output=str(output_dir / "tear_sheet.html"),
            title="Multi-Factor Long/Short Tear Sheet",
        )
    except Exception:
        html_path = output_dir / "tear_sheet.html"
        html_path.write_text("<html><body><h1>Tear Sheet</h1><p>Quantstats not available.</p></body></html>")

    equity = (1 + returns).cumprod()
    plt.figure(figsize=(10, 5))
    equity.plot()
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve.png")
    plt.close()

    drawdown = equity / equity.cummax() - 1
    plt.figure(figsize=(10, 4))
    drawdown.plot()
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown.png")
    plt.close()

