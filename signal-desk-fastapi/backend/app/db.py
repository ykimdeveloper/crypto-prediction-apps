"""SQLite persistence (stdlib sqlite3, WAL). One row per fact, natural primary keys so re-ingesting is idempotent."""
from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path

from . import config

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- closed 1-minute candles straight from the exchange (via CCXT)
CREATE TABLE IF NOT EXISTS candles (
  exchange TEXT NOT NULL, symbol TEXT NOT NULL, tf TEXT NOT NULL, ts INTEGER NOT NULL,   -- ts = candle open, epoch ms
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
  PRIMARY KEY (exchange, symbol, tf, ts)
) WITHOUT ROWID;

-- Bollinger / Stochastic RSI / ATR computed with `ta`, one row per closed candle (NULL while warming up)
CREATE TABLE IF NOT EXISTS indicators (
  exchange TEXT NOT NULL, symbol TEXT NOT NULL, tf TEXT NOT NULL, ts INTEGER NOT NULL,
  bb_mid REAL, bb_upper REAL, bb_lower REAL, bb_state TEXT,
  stoch_k REAL, stoch_d REAL, stoch_delta REAL, stoch_traj TEXT, stoch_state TEXT,
  atr REAL, atr_floor REAL, atr_state TEXT,
  ml_p_up REAL,                                  -- h2o probability that close[t+horizon] > close[t] (NULL if no model)
  backend TEXT,                                  -- 'ta' or 'pandas'
  PRIMARY KEY (exchange, symbol, tf, ts)
) WITHOUT ROWID;

-- one ticker snapshot per symbol per minute (last/bid/ask/24h stats)
CREATE TABLE IF NOT EXISTS ticker_snapshots (
  exchange TEXT NOT NULL, symbol TEXT NOT NULL, ts INTEGER NOT NULL,       -- ts = minute start, epoch ms
  last REAL, bid REAL, ask REAL, high_24h REAL, low_24h REAL, base_volume_24h REAL, quote_volume_24h REAL, pct_change_24h REAL,
  source TEXT,                                                              -- 'ticker' or 'candle' (fallback)
  PRIMARY KEY (exchange, symbol, ts)
) WITHOUT ROWID;

-- every visible entry / exit cue the strategies produced
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL, exchange TEXT NOT NULL, symbol TEXT NOT NULL, tf TEXT NOT NULL,
  strategy TEXT NOT NULL,                        -- rev | mom | end
  signal TEXT NOT NULL,                          -- BUY UP | BUY DOWN | STOP LEVEL | TAKE PROFIT
  confidence INTEGER, ask_cents REAL, lean REAL, story TEXT, price REAL, strike REAL, contract_start INTEGER, reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_signals_lookup ON signals (exchange, symbol, ts);

-- paper trades (open -> closed | abandoned). Money is in contract cents, gross of the fee setting.
CREATE TABLE IF NOT EXISTS paper_trades (
  id TEXT PRIMARY KEY, exchange TEXT NOT NULL, symbol TEXT NOT NULL, tf TEXT NOT NULL, strategy TEXT NOT NULL,
  direction TEXT NOT NULL, confidence INTEGER, strike REAL, entry_cents REAL, tp_cents REAL,
  contract_start INTEGER, contract_end INTEGER, shadow INTEGER NOT NULL DEFAULT 0, opened_ts INTEGER,
  status TEXT NOT NULL,                          -- open | closed | abandoned
  how TEXT,                                      -- tp | settle
  pnl_cents REAL, settle_pnl_cents REAL, close_price REAL, path_json TEXT, closed_ts INTEGER
);
CREATE INDEX IF NOT EXISTS ix_paper_lookup ON paper_trades (exchange, tf, strategy, status);

-- trained h2o models (metadata; the binary lives in MODEL_DIR)
CREATE TABLE IF NOT EXISTS ml_models (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts INTEGER NOT NULL, exchange TEXT, symbols TEXT, tf TEXT,
  horizon INTEGER, n_rows INTEGER, algo TEXT, auc REAL, logloss REAL, bundle_path TEXT, active INTEGER NOT NULL DEFAULT 1
);
"""

TABLES = {"candles", "indicators", "ticker_snapshots", "signals", "paper_trades", "ml_models"}

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def init(path: str | None = None) -> sqlite3.Connection:
    global _conn
    path = path or config.DB_PATH
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        _conn.execute("UPDATE paper_trades SET status='abandoned' WHERE status='open'")  # open positions don't survive a restart
        _conn.commit()
    return _conn


def close():
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def _x(sql, args=(), many=False):
    with _lock:
        cur = _conn.executemany(sql, args) if many else _conn.execute(sql, args)
        _conn.commit()
        return cur


def _f(v):
    """NaN/inf -> NULL, numpy scalars -> python."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return v
    return None if (math.isnan(v) or math.isinf(v)) else v


def _s(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


# ---------------- writers ----------------
def upsert_candles(exchange, symbol, tf, candles):
    if candles:
        _x("INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?,?)",
           [(exchange, symbol, tf, int(c["t"]), c["o"], c["h"], c["l"], c["c"], c["v"]) for c in candles], many=True)


def upsert_indicators(exchange, symbol, tf, frame, backend, only_ts=None):
    """frame = app.indicators.indicator_frame(...). Keeps an existing ml_p_up if the new row doesn't carry one."""
    rows = []
    for r in frame.itertuples(index=False):
        if only_ts is not None and int(r.t) not in only_ts:
            continue
        rows.append((exchange, symbol, tf, int(r.t), _f(r.bb_mid), _f(r.bb_upper), _f(r.bb_lower), _s(r.bb_state),
                     _f(r.stoch_k), _f(r.stoch_d), _f(r.stoch_delta), _s(r.stoch_traj), _s(r.stoch_state),
                     _f(r.atr), _f(r.atr_floor), _s(r.atr_state), backend))
    if rows:
        _x("""INSERT INTO indicators (exchange,symbol,tf,ts,bb_mid,bb_upper,bb_lower,bb_state,stoch_k,stoch_d,stoch_delta,
                stoch_traj,stoch_state,atr,atr_floor,atr_state,backend) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
              ON CONFLICT(exchange,symbol,tf,ts) DO UPDATE SET bb_mid=excluded.bb_mid, bb_upper=excluded.bb_upper,
                bb_lower=excluded.bb_lower, bb_state=excluded.bb_state, stoch_k=excluded.stoch_k, stoch_d=excluded.stoch_d,
                stoch_delta=excluded.stoch_delta, stoch_traj=excluded.stoch_traj, stoch_state=excluded.stoch_state,
                atr=excluded.atr, atr_floor=excluded.atr_floor, atr_state=excluded.atr_state, backend=excluded.backend""",
           rows, many=True)


def set_ml_p_up(exchange, symbol, tf, ts, p):
    _x("UPDATE indicators SET ml_p_up=? WHERE exchange=? AND symbol=? AND tf=? AND ts=?", (_f(p), exchange, symbol, tf, int(ts)))


def insert_ticker(exchange, symbol, ts, t: dict, source="ticker"):
    _x("INSERT OR REPLACE INTO ticker_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
       (exchange, symbol, int(ts), _f(t.get("last")), _f(t.get("bid")), _f(t.get("ask")), _f(t.get("high")), _f(t.get("low")),
        _f(t.get("base_volume")), _f(t.get("quote_volume")), _f(t.get("percentage")), source))


def insert_signal(exchange, symbol, tf, strategy, signal, *, ts, conf=None, ask=None, lean=None, story=None, price=None,
                  strike=None, contract_start=None, reason=None):
    _x("""INSERT INTO signals (ts,exchange,symbol,tf,strategy,signal,confidence,ask_cents,lean,story,price,strike,contract_start,reason)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
       (int(ts), exchange, symbol, tf, strategy, signal, conf, _f(ask), _f(lean), story, _f(price), _f(strike), contract_start, reason))


def save_trade(exchange, symbol, r: dict, status: str, ts: int):
    """r = paper record (see app.paper). Called on open and again on close/settle."""
    _x("""INSERT INTO paper_trades (id,exchange,symbol,tf,strategy,direction,confidence,strike,entry_cents,tp_cents,contract_start,
            contract_end,shadow,opened_ts,status,how,pnl_cents,settle_pnl_cents,close_price,path_json,closed_ts)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
          ON CONFLICT(id) DO UPDATE SET status=excluded.status, how=excluded.how, pnl_cents=excluded.pnl_cents,
            settle_pnl_cents=excluded.settle_pnl_cents, close_price=excluded.close_price, path_json=excluded.path_json,
            closed_ts=excluded.closed_ts""",
       (f"{exchange}|{r['id']}", exchange, symbol, r["tf"], r["strat"], r["dir"], r.get("conf"), _f(r["strike"]), _f(r["entry"]),
        _f(r.get("tp")), r["start"], r["end"], int(bool(r.get("shadow"))), r.get("opened"), status, r.get("how"), _f(r.get("pnl")),
        _f(r.get("settlePnl")), _f(r.get("close")), json.dumps(r["path"]) if r.get("path") is not None else None,
        ts if status != "open" else None))


def save_model(**kw):
    _x("UPDATE ml_models SET active=0 WHERE 1")
    _x("""INSERT INTO ml_models (created_ts,exchange,symbols,tf,horizon,n_rows,algo,auc,logloss,bundle_path,active)
          VALUES (?,?,?,?,?,?,?,?,?,?,1)""",
       (kw["created_ts"], kw.get("exchange"), kw.get("symbols"), kw.get("tf"), kw.get("horizon"), kw.get("n_rows"), kw.get("algo"),
        _f(kw.get("auc")), _f(kw.get("logloss")), kw.get("bundle_path")))


# ---------------- readers ----------------
def load_closed_trades(exchange, limit=1500) -> list[dict]:
    with _lock:
        rows = _conn.execute("SELECT * FROM paper_trades WHERE exchange=? AND status='closed' ORDER BY closed_ts DESC LIMIT ?",
                             (exchange, limit)).fetchall()
    out = []
    for r in reversed(rows):
        out.append(dict(id=r["id"].split("|", 1)[1], sym=r["symbol"].split("/")[0], tf=r["tf"], strat=r["strategy"], dir=r["direction"],
                        conf=r["confidence"], strike=r["strike"], entry=r["entry_cents"], tp=r["tp_cents"], start=r["contract_start"],
                        end=r["contract_end"], shadow=bool(r["shadow"]), pnl=r["pnl_cents"], how=r["how"], close=r["close_price"],
                        path=json.loads(r["path_json"]) if r["path_json"] else [], settlePnl=r["settle_pnl_cents"]))
    return out


def query(table: str, *, exchange=None, symbol=None, tf=None, since=None, limit=500, order="DESC") -> list[dict]:
    if table not in TABLES:
        raise ValueError(f"unknown table {table}")
    ts_col = {"paper_trades": "opened_ts", "ml_models": "created_ts"}.get(table, "ts")
    filters = {"exchange": exchange, "symbol": symbol, "tf": tf}
    cols = {"ticker_snapshots": ("exchange", "symbol"), "ml_models": ("exchange",)}.get(table, ("exchange", "symbol", "tf"))
    where, args = [], []
    for col in cols:
        if filters[col] is not None:
            where.append(f"{col}=?")
            args.append(filters[col])
    if since is not None:
        where.append(f"{ts_col}>=?")
        args.append(int(since))
    sql = f"SELECT * FROM {table}" + (" WHERE " + " AND ".join(where) if where else "")
    sql += f" ORDER BY {ts_col} {'ASC' if order.upper() == 'ASC' else 'DESC'} LIMIT ?"
    args.append(max(1, min(int(limit), 5000)))
    with _lock:
        return [dict(r) for r in _conn.execute(sql, args).fetchall()]


def dataset(exchange, symbol, tf="1m") -> list[dict]:
    """candles joined with their indicator row, oldest first — the ML training input."""
    with _lock:
        rows = _conn.execute("""SELECT c.ts, c.open, c.high, c.low, c.close, c.volume, i.bb_mid, i.bb_upper, i.bb_lower, i.stoch_k,
                                       i.stoch_d, i.stoch_delta, i.atr
                                FROM candles c LEFT JOIN indicators i USING (exchange, symbol, tf, ts)
                                WHERE c.exchange=? AND c.symbol=? AND c.tf=? ORDER BY c.ts""", (exchange, symbol, tf)).fetchall()
    return [dict(r) for r in rows]


def active_model():
    with _lock:
        r = _conn.execute("SELECT * FROM ml_models WHERE active=1 ORDER BY id DESC LIMIT 1").fetchone()
    return dict(r) if r else None
