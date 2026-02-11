from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortfolioConstraints:
    gross_leverage: float = 2.0
    max_weight: float = 0.02
    beta_tolerance: float = 0.05

