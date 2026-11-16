"""quant15 - a 15-minute probability + volatility layer that sits NEXT TO the original contract algorithm (strategy.py).

  bars.py       CCXT backfill (REST) -> SQLite, 1m -> 15m bars aligned to the contract windows (:00/:15/:30/:45)
  features15.py causal 15m features (returns, realized vol, EWMA, range, RSI/EMA, time, BTC<->ETH cross-asset)
  vol15.py      HAR (variance form, fit on train only), optional GARCH via `arch`, volatility regimes
  model15.py    label = "does the NEXT 15m window settle UP", LightGBM/CatBoost/XGBoost (hist_gb fallback), purged walk-forward
  replay.py     replays the ORIGINAL rev/mom/end strategies over stored 1m candles, minute by minute, with paper.py's rules
  combine.py    original vs model-only vs gated (original AND model agree) + sigma calibration + a verdict
  service.py    compare / train / decide entry points (blocking; the routes run them in worker threads)
  routes.py     FastAPI router, mounted from main.py with two lines

Nothing here changes strategy.py, paper.py, hub.py or the h2o / approaches paths. The combined decision is opt-in and
defaults to "original" until a compare run says otherwise.
"""
