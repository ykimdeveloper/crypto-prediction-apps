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
