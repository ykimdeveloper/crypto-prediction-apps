from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

# CryptoDataDownload file URLs. Verify these on cryptodatadownload.com/data/
# (they publish minute/hourly/daily; the loader resamples anything finer than 15m).
SOURCES = {
    "BTC": "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_minute.csv",
    "ETH": "https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_minute.csv",
}
OTHER = {"BTC": "ETH", "ETH": "BTC"}  # cross-asset features

BAR = "15min"
HORIZON = 4            # predict P(up) over next 4 bars = 1 hour
TRAIN_FRAC, VALID_FRAC = 0.70, 0.15   # remainder is the test set
TAIL_BARS = 6000       # history used at serve time (must exceed longest HAR window)

# Minimum |p - 0.5| required to act, by volatility regime. Tune on validation data.
EDGE = {"low": 0.02, "mid": 0.03, "high": 0.04}
