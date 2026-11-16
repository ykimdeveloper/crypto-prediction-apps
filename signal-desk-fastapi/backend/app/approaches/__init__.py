"""Pluggable direction-model "approaches" that share ONE feature set, ONE label definition and ONE purged walk-forward harness,
so the numbers they report are comparable. See README section "Model approaches".

  features.py    candle-only features, past data only (tested for look-ahead)
  labels.py      cost- and volatility-aware label with a neutral zone (no more "close > close" coin flips)
  validation.py  expanding walk-forward folds with a purge gap >= the label horizon
  vol.py         volatility forecasters (rolling, EWMA, HAR, optional GARCH) used for gating / sizing
  models.py      the approaches: baselines, logistic regression, HistGB, LightGBM, CatBoost, XGBoost (optional ones import lazily)
  evaluate.py    out-of-fold metrics + non-overlapping trade simulation net of costs
  service.py     compare / train / predict entry points used by the FastAPI routes
"""
