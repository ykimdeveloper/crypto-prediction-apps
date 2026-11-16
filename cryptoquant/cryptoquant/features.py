import numpy as np
import pandas as pd
from . import vol


def build_features(bars, other, garch_params, har_b):
    c = bars.close
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=bars.index)

    for k in (1, 2, 4, 8, 16, 32, 96):
        f[f"ret_{k}"] = lc.diff(k)
    for w in vol.WINS:
        f[f"rv_{w}"] = vol.rv(r, w)
    f["rv_ratio_4_96"] = f["rv_4"] / f["rv_96"]
    f["rv_ratio_96_672"] = f["rv_96"] / f["rv_672"]

    f["garch_sigma"] = vol.garch_sigma(r, garch_params).reindex(bars.index)
    f["har_vol"] = vol.har_forecast(r, har_b)
    f["garch_over_har"] = f["garch_sigma"] / f["har_vol"]
    f["z_ret_1"] = f["ret_1"] / f["garch_sigma"].shift(1)
    f["z_ret_4"] = f["ret_4"] / (f["garch_sigma"].shift(1) * 2)

    hl = np.log(bars.high / bars.low)
    f["range"] = hl
    f["range_z"] = hl / hl.rolling(96).mean()
    f["close_pos"] = (c - bars.low) / (bars.high - bars.low).replace(0, np.nan)
    lv = np.log1p(bars.volume)
    f["vol_z"] = (lv - lv.rolling(672).mean()) / lv.rolling(672).std()

    hod = bars.index.hour + bars.index.minute / 60
    f["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    f["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    f["dow"] = bars.index.dayofweek

    if other is not None:
        lo = np.log(other.close.reindex(bars.index))
        f["x_ret_1"] = lo.diff(1)
        f["x_ret_4"] = lo.diff(4)
        f["x_rv_96"] = vol.rv(lo.diff(), 96)
        f["rel_ret_4"] = f["ret_4"] - f["x_ret_4"]

    return f.replace([np.inf, -np.inf], np.nan)


def make_label(bars, horizon):
    lc = np.log(bars.close)
    fwd = lc.shift(-horizon) - lc
    y = (fwd > 0).astype(float).where(fwd.notna())
    return y, fwd
