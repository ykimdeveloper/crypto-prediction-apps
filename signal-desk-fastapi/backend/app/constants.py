"""1:1 port of src/lib/constants.js (only what the server needs)."""
MIN_MS = 60_000
CONTRACT_MS = 15 * MIN_MS

TIMING = dict(midEntryMinRemaining=5, midManageUntil=1, eomMaxRemaining=8, eomMinRemaining=2)
PRICE = dict(minEntry=35, mmrBand=(35, 50), mmcBand=(50, 78), eomBand=(85, 98), tpMmr=15, tpMmc=20)
BOOK = dict(sigmaFudge=1.4, halfSpread=0.5)
EOM_CONF_SIGMA = 2.0

BB_PERIOD = 20
STOCH_PERIOD = 14
RSI_PERIOD = 14
ATR_PERIOD = 15
ATR_FLOOR_BPS_PER_MIN = 0.0007

STRAT_KEYS = ("rev", "mom", "end")
STRAT_LABEL = {"rev": "MID REV", "mom": "MID MOM", "end": "END-MKT"}

PAPER = dict(
    minConf=80,
    gateMinN=5,
    gates=dict(
        mid=dict(holdSpanMin=60, holdBelow=-2, releaseSpanMin=30, releaseAtLeast=2.5),
        end=dict(holdSpanMin=120, holdBelow=-2, releaseSpanMin=60, releaseAtLeast=0.0001),
    ),
    slGrid=[None, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    exitLagSamples=0,
    tpGrid=[None, 5, 10, 15, 20, 30],
    optimizerMinTrades=8,
    pathSampleMs=5000,
)
