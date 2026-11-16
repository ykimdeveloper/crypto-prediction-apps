# Signal Desk backend (FastAPI + CCXT Pro + ta + h2o + SQLite)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000        # add --reload while developing
```
Then `npm run dev` for the front end (vite.config.js proxies `/api` and `/ws` to :8000) and use the **Data source** switch in the header.

Tests (no network / no ccxt needed): `python -m unittest backend.tests.test_core backend.tests.test_parity` from the project root
(the parity test runs your original JS engine with Node and compares).

## Environment variables
| var | default | meaning |
|---|---|---|
| `SIGNALDESK_DB` | `backend/data/signaldesk.db` | SQLite file |
| `INDICATOR_BACKEND` | `ta` | `ta`, or `pandas` (bit-exact port of `indicators.js`; auto-fallback if `ta` is missing) |
| `ATR_SMOOTHING` | `sma` | `sma` = what the JS does; `wilder` = `ta.AverageTrueRange` |
| `CCXT_USE_WS` | `1` | ccxt.pro `watch_*` where available, REST polling otherwise |
| `CCXT_POLL_S` | `3` | REST fallback interval |
| `ALWAYS_ON` | – | keep recording with no browser open, e.g. `binance:BTC,binance:ETH,kraken:BTC:USD` |
| `IDLE_STOP_S` | `120` | stop an unwatched feed after this long |
| `ML_ENABLED`, `ML_HORIZON`, `H2O_MAX_MEM` | `1`, `15`, `2G` | h2o probability column |
| `CORS_ORIGINS` | localhost:5173 | only needed if the front end doesn't go through the Vite proxy |

## SQLite tables
`candles`, `indicators` (BB / StochRSI / ATR / `ml_p_up`, one row per closed candle), `ticker_snapshots` (one per symbol per minute),
`signals`, `paper_trades`, `ml_models`. Browse them with `GET /api/db/<table>?exchange=binance&symbol=BTC&limit=100`.

## ML (optional)
Let a feed run until `candles` has a few thousand rows, then
`POST /api/ml/train {"exchange":"binance","symbols":["BTC/USDT","ETH/USDT"]}` and watch `GET /api/ml/status`.
The result is a probability shown on each card as **ML ↑**. It is informational; it never gates a strategy.

pip install fastapi uvicorn ccxt ccxtpro


python3 -m venv fastapi-env
source fastapi-env/bin/activate
pip install -r requirements.txt


uvicorn main:app --reload
uvicorn app.main:app --reload                                              ─╯


## Model approaches (`app/approaches/`)
Several direction models behind one feature set, one label, and one purged walk-forward test, so their numbers are comparable.
The h2o path above is unchanged and still feeds the **ML ↑** chip; everything here is additive and informational.

| approach | needs | role |
|---|---|---|
| `momentum`, `ema_cross` | – | baselines: if a model can't beat these it has no edge |
| `logreg` | scikit-learn | strong on small/noisy data |
| `hist_gb` | scikit-learn | LightGBM-style boosting that is always available |
| `lightgbm` / `catboost` / `xgboost` | `pip install …` | primary model / alternative / benchmark (skipped, not fatal, if missing) |
| volatility: `rv15`, `ewma`, `har`, `garch` | `arch` for garch | scored against realized vol; used for the trade gate, expected move and regime |

**Label:** `y=1` if `ln(close[t+H]/close[t]) > cost + k·σ·√H`, `0` if below the negative of that, otherwise neutral (not trained on).
**Validation:** expanding folds split on timestamps; a training row is used only if its label window ends before the test block starts.
**Scoring:** out-of-fold AUC / Brier / log-loss, plus a trade simulation net of `cost_bps` (one position per symbol at a time), with and without a volatility gate.

```bash
# 1. compare everything on what you've recorded (runs in the background; poll GET /api/ml/compare)
curl -X POST localhost:8000/api/ml/compare -H 'content-type: application/json' \
  -d '{"exchange":"okx","symbols":["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT"],"cost_bps":8,"k":0.5}'
curl localhost:8000/api/ml/compare
# 2. train + calibrate the one you pick (newest 20% of data is held out for isotonic calibration)
curl -X POST localhost:8000/api/ml/approaches/train -H 'content-type: application/json' \
  -d '{"approach":"lightgbm","exchange":"okx","symbols":["BTC/USDT","ETH/USDT"]}'
# 3. ask for a decision: probability, predicted vol, regime, long/short/no_trade + reason
curl 'localhost:8000/api/ml/predict?exchange=okx&symbol=BTC/USDT'
```
Tests: `python -m unittest backend.tests.test_approaches` (look-ahead, gaps, purge, planted-signal-found, noise-not-found, decision layer).

**Read before trusting any number:** candles here are 1-minute and the horizon is 15 bars, so neighbouring labels share 14/15 of their
window. `warnings` in the compare result reports `labeled rows / horizon` as the honest sample size. A few hours of recording is a
handful of independent samples: expect "no detectable edge" until you have weeks (use `ALWAYS_ON`). `heuristic_edge_bps` in `/predict`
is `(2p-1)·0.8·σ√H`, a sizing heuristic, not a forecast.
