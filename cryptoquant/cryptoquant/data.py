import pandas as pd
import requests
from . import config

_cache = {}


def download(symbol, force=False):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / f"{symbol}_raw.csv"
    if force or not path.exists():
        r = requests.get(config.SOURCES[symbol], headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def read_cdd(path):
    """CDD files start with a notice line before the header; newer ones may not."""
    with open(path, "r", errors="ignore") as f:
        first = f.readline().lower()
    skip = 0 if first.startswith(("unix", "date")) else 1
    df = pd.read_csv(path, skiprows=skip)
    df.columns = [c.strip().lower() for c in df.columns]
    if "unix" in df:
        u = pd.to_numeric(df["unix"], errors="coerce")
        med = u.dropna().median()
        unit = "us" if med > 1e14 else "ms" if med > 1e11 else "s"
        idx = pd.to_datetime(u, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(df["date"], utc=True)
    vol_col = next(c for c in df.columns if c.startswith("volume"))
    out = pd.DataFrame(
        {"open": df["open"], "high": df["high"], "low": df["low"],
         "close": df["close"], "volume": df[vol_col]}
    )
    out.index = idx
    out.index.name = "ts"
    out = out[out.index.notna()]
    out = out[~out.index.duplicated()].sort_index()
    return out.apply(pd.to_numeric, errors="coerce").dropna()


def resample(df, rule=config.BAR):
    bars = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return bars.dropna(subset=["close"])


def get_bars(symbol, refresh=False):
    pkl = config.DATA_DIR / f"{symbol}_{config.BAR}.pkl"
    if refresh:
        _cache.pop(symbol, None)
    if not refresh and pkl.exists():
        mtime = pkl.stat().st_mtime
        if symbol in _cache and _cache[symbol][0] == mtime:
            return _cache[symbol][1]
        bars = pd.read_pickle(pkl)
    else:
        bars = resample(read_cdd(download(symbol, force=refresh)))
        bars.to_pickle(pkl)
        mtime = pkl.stat().st_mtime
    _cache[symbol] = (mtime, bars)
    return bars
