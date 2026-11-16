"""The approaches. Each one maps a feature frame -> P(up), and can be swapped in with one line.

  kind      name          needs                     notes
  baseline  momentum      -                         sign of vol-normalised recent return. If you can't beat this, you have nothing.
  baseline  ema_cross     -                         EMA 9 vs 21
  linear    logreg        scikit-learn              scaled logistic regression; strong on small/noisy data
  tree      hist_gb       scikit-learn              sklearn's LightGBM-style histogram boosting; always available
  tree      lightgbm      pip install lightgbm      the recommended primary model
  tree      catboost      pip install catboost      good when you add categorical context (symbol, session)
  tree      xgboost       pip install xgboost       benchmark against lightgbm
  (not here) h2o          existing ml.py            still trained via /api/ml/train

Optional libraries import lazily: a missing one shows up as available=false, never as an import error at startup.
"""
from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np
import pandas as pd


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


class Approach:
    name = ""
    label = ""
    kind = "tree"
    needs: str | None = None       # importable module name, None = always available
    fitted = True                  # False for rule-based baselines

    @classmethod
    def available(cls) -> bool:
        return cls.needs is None or _has(cls.needs)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Approach":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # P(up), shape (n,)
        raise NotImplementedError


# ---------------- baselines ----------------
class Momentum(Approach):
    name, label, kind, fitted = "momentum", "Momentum baseline", "baseline", False

    def predict_proba(self, X):
        z = X["ret_5"] / (X["rv_15"] * np.sqrt(5)).replace(0, np.nan)
        return (0.5 + 0.5 * np.tanh(z / 2)).fillna(0.5).to_numpy()


class EmaCross(Approach):
    name, label, kind, fitted = "ema_cross", "EMA 9/21 crossover baseline", "baseline", False

    def predict_proba(self, X):
        return np.where(X["ema_9_21"].fillna(0) > 0, 0.6, 0.4)


# ---------------- learned ----------------
class LogReg(Approach):
    name, label, kind = "logreg", "Logistic regression", "linear"

    def fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class _Sk(Approach):
    """Wrapper for anything with a scikit-learn API. Subclasses set `make`."""
    make: Callable

    def fit(self, X, y):
        self.m = self.make()
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class HistGB(_Sk):
    name, label = "hist_gb", "Histogram gradient boosting (sklearn)"

    @staticmethod
    def make():
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
                                              l2_regularization=1.0, early_stopping=False, random_state=1)


class LightGBM(_Sk):
    name, label, needs = "lightgbm", "LightGBM", "lightgbm"

    @staticmethod
    def make():
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=1, verbose=-1)


class CatBoost(_Sk):
    name, label, needs = "catboost", "CatBoost", "catboost"

    @staticmethod
    def make():
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, l2_leaf_reg=5, random_seed=1, verbose=0)


class XGBoost(_Sk):
    name, label, needs = "xgboost", "XGBoost", "xgboost"

    @staticmethod
    def make():
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=20, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss", random_state=1, verbosity=0)


REGISTRY: dict[str, type[Approach]] = {c.name: c for c in (Momentum, EmaCross, LogReg, HistGB, LightGBM, CatBoost, XGBoost)}
DEFAULT_COMPARE = ["momentum", "ema_cross", "logreg", "hist_gb", "lightgbm", "catboost", "xgboost"]


def make(name: str) -> Approach:
    if name not in REGISTRY:
        raise KeyError(f"unknown approach {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]()


def catalog() -> list[dict]:
    return [dict(name=c.name, label=c.label, kind=c.kind, available=c.available(), needs=c.needs, fitted=c.fitted,
                 install=(f"pip install {c.needs}" if c.needs and not c.available() else None)) for c in REGISTRY.values()]
