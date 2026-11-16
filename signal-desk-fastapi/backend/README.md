# Signal Desk backend (FastAPI + CCXT Pro + ta + h2o + SQLite)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000        # add --reload while developing

pip install fastapi uvicorn ccxt ccxtpro

python3 -m venv fastapi-env
source fastapi-env/bin/activate
pip install -r requirements.txt


uvicorn main:app --reload
uvicorn app.main:app --reload       
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

### Original vs. 15-minute model, live (quant15)

Header → **ML · FastAPI** on → **Signals: Original | Gated**. One global setting on the server (persisted in `models/quant15_live_policy.json`).
`Original` changes nothing. `Gated` keeps a rev/mom/end BUY only if the trained 15m model's P(up) for the current window backs its direction by
the margin (default 0.02); otherwise it shows WAIT with the reason. It runs before paper trading, so the ledger, alerts and log follow the policy.
The model is asked once per window in a worker thread; while it is pending BUYs are held; if no model exists for the desk's exchange label the
signals pass through unchanged and the card says why. Train on the **same exchange label the desk uses** (e.g. `binance`, not `agg:...`).

### History from CryptoDataDownload (binance only)

`POST /api/quant15/compare` and `/train` first top up missing history from CryptoDataDownload (`auto_fetch`, default true).
`POST /api/quant15/backfill` takes `source`: `auto` (CDD for binance 1m, then a CCXT top-up of the last day), `cdd`, or `ccxt`.
The CDD file is downloaded once (cached 12 h in `backend/data/cdd`) and only the newest `days` are imported. Manual route:
`python -m app.quant15.csvimport --exchange binance --symbol BTC/USDT --days 60 FILE.csv`. Override the address with `CDD_URL_TEMPLATE`.
Tests: `python -m unittest backend.tests.test_live_cdd backend.tests.test_csvimport backend.tests.test_quant15`.




## data curl
Routes: /api/quant15/* is now mounted in backend/app/main.py, and scipy is in backend/requirements.txt.
Your cryptoquant server: I didn't run it alongside. quant15 already contains its algorithms: the same features, the three-model average (LightGBM, CatBoost, XGBoost), HAR volatility, and optional GARCH. I ported the one thing it had that quant15 lacked, the CryptoDataDownload download. It also predicts four bars ahead, not your one-window contract, so I left it standalone.
Missing data: when you call /compare or /train with exchange: "binance", it now pulls missing history from CryptoDataDownload first. /backfill takes source: auto|cdd|ccxt, and auto uses CryptoDataDownload with a CCXT top-up of the last day. If a download fails, /status shows the error.

curl -s localhost:8000/api/quant15/status
curl -X POST localhost:8000/api/quant15/backfill -H 'content-type: application/json' -d '{"exchange":"coinbase","symbols":["BTC/USDT","ETH/USDT"],"days":10,"source":"ccxt"}'



## The switch
In the header, with ML · FastAPI on, there's a Signals: Original | Gated control. It's one global setting on the server.

Gated turns a BUY into WAIT unless the trained 15-minute model backs its direction. That happens before paper trading, so the ledger, alerts and log all follow it. Cards show the model's P(up) and any vetoes.

Honest states: the model is asked once per window in the background, and BUYs are held while it answers. If no model exists for the desk's exchange, signals pass through unchanged and the card says why. The control also warns you when the last compare run didn't recommend gating.
Model-only isn't offered live. A paper entry needs a real market price, and that policy only has a modeled one. It stays available in /compare and /decide.



## Testing
34 backend tests pass, including the new ones for the policy switch and the CryptoDataDownload download (against a fake server). I couldn't run these:
The FastAPI routes and the hub change, which are only compile-checked because this sandbox has no fastapi or ccxt.
The front end, which I could only compile-check.
The CryptoDataDownload address (.../cdd/Binance_BTCUSDT_minute.csv), which is a best guess. If it's wrong or behind a login, the error tells you, and CDD_URL_TEMPLATE or the manual csvimport command covers it.


## scratch notes
curl -X POST localhost:8000/api/quant15/train

curl -X POST localhost:8000/api/quant15/status

curl -s localhost:8000/api/quant15/status

curl -X POST localhost:8000/api/quant15/train \
-H 'content-type: application/json' \
-d '{"exchange":"coinbase","symbols":["BTC/USDT","ETH/USDT"]}'

curl -X POST localhost:8000/api/quant15/train \
-H 'content-type: application/json' \
-d '{"exchange":"binance","symbols":["BTC/USDT","ETH/USDT"]}'

curl -X POST localhost:8000/api/quant15/train -H 'content-type: application/json' \                                                     ─╯
-d '{"exchange":"coinbase","symbols":["BTC/USDT","ETH/USDT"],"days":45}'

POST /api/quant15/backfill with {"exchange":"coinbase","symbols":["BTC/USDT","ETH/USDT"],"days":10,"source":"ccxt"}

python -m app.quant15.probe --exchange coinbase --symbol BTC/USDT


## direction
curl -X POST localhost:8000/api/quant15/train -H 'content-type: application/json' \
-d '{"exchange":"coinbase","symbols":["BTC/USDT","ETH/USDT"],"days":45}'

curl -s localhost:8000/api/quant15/status     # repeat until train.state is "done" or "error"



# How to read the numbers
Direction model: AUC 0.554 over 8,612 windows. That is small but real, and "the previous bar's direction" scored 0.48. Only the scikit-learn fallback ran, so run pip install lightgbm catboost xgboost for the intended three-model average.
Gated vs original: hit rate is 80.8% vs 80.1%, a gap of 0.7 points. The cents look bigger (4.4¢ vs 3.5¢) but they're at modeled prices, not real quotes.
The "99% hit" on End-Mkt is almost certainly because those trades enter when the outcome is nearly settled, where a real ask would sit around 95–99¢. The 7¢ average is not something you'd collect.
Verdict: it recommends gated, but honestly the evidence is a modest, model-priced edge on old data. I'd treat it as a reason to test on fresh data, not to switch it on.
Volatility: re-scaling your ATR to 0.7× or 0.9× (HAR) lowers Brier from 0.183 to about 0.171. Your book's ATR × 1.4 overprices moves compared with settlements, and that improvement is bigger than what the direction model adds. It's also from 2022, so I'd confirm it on recent data.
Data gap: a