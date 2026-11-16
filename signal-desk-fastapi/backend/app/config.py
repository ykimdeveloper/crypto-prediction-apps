"""Runtime settings, all overridable with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("SIGNALDESK_DB", str(ROOT / "data" / "signaldesk.db"))
MODEL_DIR = Path(os.getenv("SIGNALDESK_MODELS", str(ROOT / "models")))
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o]

# "ta" = the `ta` package; "pandas" = a line-for-line port of src/lib/indicators.js (used automatically if `ta` is missing)
INDICATOR_BACKEND = os.getenv("INDICATOR_BACKEND", "ta")
# The JS app uses a simple moving average of true range (period 15). "wilder" switches to ta's AverageTrueRange (Wilder smoothing).
ATR_SMOOTHING = os.getenv("ATR_SMOOTHING", "sma")

USE_WS = os.getenv("CCXT_USE_WS", "1") == "1"       # ccxt.pro watch_* where the exchange supports it, REST polling otherwise
POLL_S = float(os.getenv("CCXT_POLL_S", "3"))        # REST fallback interval
PUSH_S = float(os.getenv("PUSH_INTERVAL_S", "1"))    # how often /ws/desk pushes a fresh state
STALE_MS = int(os.getenv("STALE_MS", "20000"))       # JS uses 6000 for REST polls; watch_* on quiet pairs can be slower
MAX_CANDLES = 400
IDLE_STOP_S = float(os.getenv("IDLE_STOP_S", "120"))  # stop a feed this long after the last client leaves

# Feeds that keep running (and keep writing to SQLite) with nobody connected: "binance:BTC,binance:ETH:USDT,kraken:BTC:USD"
ALWAYS_ON = [tuple(x.strip().split(":")) for x in os.getenv("ALWAYS_ON", "").split(",") if ":" in x]

# Optional: feed the h2o model probability into the card/DB (never gates a strategy)
ML_ENABLED = os.getenv("ML_ENABLED", "1") == "1"
ML_HORIZON = int(os.getenv("ML_HORIZON", "15"))  # candles ahead the model predicts (15 x 1m = one contract)

# ---- multi-exchange consolidation (CCXT mode) ----
# Quote currencies treated as "one dollar" when merging exchanges, in preference order. KRW/AUD/EUR venues are skipped.
AGG_QUOTES = tuple(q.strip().upper() for q in os.getenv("AGG_QUOTES", "USDT,USD,USDC").split(",") if q.strip())
# vwap = volume-weighted O/H/L/C across venues (default); median = per-field median (more robust to one bad venue)
AGG_MODE = os.getenv("AGG_MODE", "vwap")
# a venue whose close is further than this fraction from the cross-venue median (needs 3+ venues) is ignored for that minute
AGG_MAX_DEV = float(os.getenv("AGG_MAX_DEV", "0.03"))
MAX_AGG_EXCHANGES = int(os.getenv("MAX_AGG_EXCHANGES", "25"))
