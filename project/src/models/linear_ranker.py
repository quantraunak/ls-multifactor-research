from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge


@dataclass
class LinearRanker:
    model_type: str = "ridge"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    random_state: int = 7

    def _build(self):
        if self.model_type == "elasticnet":
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state)
        return Ridge(alpha=self.alpha, random_state=self.random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        model = self._build()
        model.fit(X.values, y.values)
        self.model_ = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model_.predict(X.values)
        return pd.Series(preds, index=X.index, name="prediction")

