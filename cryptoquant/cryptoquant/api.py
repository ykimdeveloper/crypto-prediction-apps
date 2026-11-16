from fastapi import FastAPI, HTTPException
from . import config, predict
from .data import get_bars

app = FastAPI(title="cryptoquant 15m")


def _check(symbol):
    s = symbol.upper()
    if s not in config.SOURCES:
        raise HTTPException(404, f"unknown symbol; use one of {list(config.SOURCES)}")
    return s


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/predict/{symbol}")
def predict_symbol(symbol: str):
    s = _check(symbol)
    try:
        return predict.predict_latest(s)
    except FileNotFoundError:
        raise HTTPException(503, "no trained model; run `python -m cryptoquant.train` first")


@app.post("/refresh/{symbol}")
def refresh(symbol: str):
    """Re-download CryptoDataDownload data (files update with a delay, not live)."""
    s = _check(symbol)
    bars = get_bars(s, refresh=True)
    return {"symbol": s, "last_bar": str(bars.index[-1]), "bars": len(bars)}
