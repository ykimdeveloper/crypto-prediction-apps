# cryptoquant – 15-min BTC/ETH probability + volatility service

Pipeline: CryptoDataDownload CSV -> 15m bars -> features (returns, realized vol, HAR, EGARCH, cross-asset)
-> LightGBM + CatBoost + XGBoost (P(up over next 4 bars)) -> ensemble average -> vol-regime-adjusted signal.

```
pip install -r requirements.txt
python -m cryptoquant.train --symbols BTC ETH      # downloads, trains, prints test metrics
uvicorn cryptoquant.api:app --reload
curl localhost:8000/predict/BTC
```

Design notes
- Chronological split with a purge gap of HORIZON bars; early stopping uses the validation block only.
- GARCH/HAR are fit on TRAIN only; GARCH is then filtered with fixed params (causal, no look-ahead).
- Tune `EDGE` thresholds in config.py on validation data, and include fees/slippage before trusting any PnL.
- CDD data is a periodic file, not a live feed. For live trading, swap `data.get_bars` for an exchange API.
