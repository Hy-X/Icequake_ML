#!/usr/bin/env python3
"""Raw-waveform proof for sp_differential_column_test.py: overlay two events the catalog puts far
apart in depth, aligned on P, and see whether their S arrivals line up.

Independent of dt.cc: windows are read straight from the day archive, filtered, and the P
alignment is a fresh full-window cross-correlation on Z. If the sources are co-located, S
overlays exactly at every station; if the catalog's depth gap were real, S of the second event
would sit at the dashed "catalog-predicted" line instead.

Usage (from the repository root):
    python full_catalog_pipeline/06_depth/plot_column_pair_overlay.py --array T1 --pair 836 950
"""

# --- pipeline path bootstrap (added by the 2026-09 reorganization) ---------------
import os as _os, sys as _sys
_PIPE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_PIPE] + [_os.path.join(_PIPE, _s) for _s in sorted(_os.listdir(_PIPE))
                     if _os.path.isdir(_os.path.join(_PIPE, _s))
                     and not _s.startswith(('.', 'artifacts', '__'))]:
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# --- end bootstrap ---------------------------------------------------------------

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Trace

import catalog_paths
import config
from diagnose_cc_p_deficit import CC_FMAX, CC_FMIN, load_phase_dat
from hypodd_tune import RELOC_COLS
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window
from sp_differential_column_test import load_grids, local_xy, sp

PRE, POST = 0.3, 0.6   # seconds around P and S


def window(day_index, sta, t0, t1):
    cache = RollingDayCache(day_index, sta)
    data, _ = extract_window(cache, {"primary_date": t0.date.isoformat()}, t0, t1)
    if data is None:
        return None
    out = {}
    for ch in ("Z", "N", "E"):
        tr = Trace(np.asarray(data[ch], float))
        tr.stats.sampling_rate = config.SAMPLE_RATE_HZ
        tr.detrend("demean")
        tr.taper(0.05)
        tr.filter("bandpass", freqmin=CC_FMIN, freqmax=CC_FMAX, zerophase=True)
        out[ch] = tr.data
    return out


def xcorr_lag(a, b, maxlag):
    """Lag (samples) of b relative to a, by full normalised cross-correlation."""
    best, best_k = -2, 0
    for k in range(-maxlag, maxlag + 1):
        x = a[maxlag:len(a) - maxlag]
        y = b[maxlag + k:len(b) - maxlag + k]
        c = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-20)
        if c > best:
            best, best_k = c, k
    return best_k, best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", required=True, choices=["T1", "T2"])
    ap.add_argument("--pair", type=int, nargs=2, required=True)
    args = ap.parse_args()
    A, (e1, e2) = args.array, args.pair
    sr = config.SAMPLE_RATE_HZ

    ph = load_phase_dat(catalog_paths.phase_dat(A))
    cat = pd.read_csv(catalog_paths.reloc(A), sep=r"\s+", header=None, names=RELOC_COLS)
    sta = pd.read_csv(catalog_paths.station_sel(A), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    lat0, lon0 = cat.lat.mean(), cat.lon.mean()
    cat["ex"], cat["ey"] = local_xy(cat.lat, cat.lon, lat0, lon0)
    sx, sy = local_xy(sta.lat, sta.lon, lat0, lon0)
    pos = {s.split(".")[-1]: (x, y) for s, x, y in zip(sta.sta, sx, sy)}
    c = cat.set_index("id")
    grids = load_grids(A)
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)

    stations = [k[0] for k in ph[e1] if isinstance(k, tuple) and k[1] == "P"
                and (k[0], "S") in ph[e1] and (k[0], "P") in ph[e2] and (k[0], "S") in ph[e2]]
    fig, axes = plt.subplots(len(stations), 1, figsize=(12, 2.6 * len(stations)), squeeze=False)
    for ax, s in zip(axes[:, 0], stations):
        tp1, tp2 = ph[e1]["origin"] + ph[e1][(s, "P")], ph[e2]["origin"] + ph[e2][(s, "P")]
        sp1 = ph[e1][(s, "S")] - ph[e1][(s, "P")]
        span = sp1 + POST + 0.2
        w1, w2 = window(day_index, s, tp1 - PRE, tp1 + span), window(day_index, s, tp2 - PRE, tp2 + span)
        if w1 is None or w2 is None:
            ax.set_title(f"{s}: no data"); continue
        n = min(len(w1["Z"]), len(w2["Z"]))
        ip = int((PRE + 0.15) * sr)
        k, cc = xcorr_lag(w1["Z"][:ip], w2["Z"][:ip], int(0.05 * sr))  # refine P alignment
        t = np.arange(n) / sr - PRE
        h1 = np.hypot(w1["N"][:n], w1["E"][:n])
        h2 = np.roll(np.hypot(w2["N"], w2["E"]), -k)[:n]
        z1, z2 = w1["Z"][:n], np.roll(w2["Z"], -k)[:n]
        ax.plot(t, z1 / np.abs(z1).max() + 1.2, color="#2a78d6", lw=0.8, label=f"ev {e1} Z")
        ax.plot(t, z2 / np.abs(z2).max() + 1.2, color="#eb6834", lw=0.8, alpha=0.8, label=f"ev {e2} Z")
        ax.plot(t, h1 / h1.max() - 0.9, color="#2a78d6", lw=0.8, label=f"ev {e1} |H|")
        ax.plot(t, h2 / h2.max() - 0.9, color="#eb6834", lw=0.8, alpha=0.8, label=f"ev {e2} |H|")
        r1 = np.hypot(c.ex[e1] - pos[s][0], c.ey[e1] - pos[s][1])
        r2 = np.hypot(c.ex[e2] - pos[s][0], c.ey[e2] - pos[s][1])
        sp_c1 = float(sp(grids, np.array([c.depth[e1]]), np.array([r1]))[0])
        sp_c2 = float(sp(grids, np.array([c.depth[e2]]), np.array([r2]))[0])
        ax.axvline(sp1, color="#0b0b0b", lw=1, label=f"ev {e1} S pick")
        ax.axvline(sp1 + (sp_c2 - sp_c1), color="#eb6834", lw=1.5, ls="--",
                   label=f"ev {e2} S if catalog depths were right ({1000*(sp_c2-sp_c1):+.0f} ms)")
        ax.set_xlim(-PRE, span - 0.1)
        ax.set_yticks([])
        ax.set_title(f"{s}: aligned on P (Z CC {cc:.2f}); top Z, bottom horizontal envelope",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1, 0].set_xlabel("time after P (s)")
    fig.suptitle(f"{A} events {e1} / {e2}: catalog depths {c.depth[e1]:.3f} / {c.depth[e2]:.3f} km "
                 f"(dz {1000*(c.depth[e1]-c.depth[e2]):+.0f} m)", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(catalog_paths.work_dir(A), f"{A.lower()}_column_pair_overlay_{e1}_{e2}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
