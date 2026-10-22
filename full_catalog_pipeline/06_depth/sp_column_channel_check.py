#!/usr/bin/env python3
"""Is the collapsed S-P differential in sp_differential_column_test.py real, or a channel artifact?

regenerate_dtcc.py measures S lags on S_CHANNELS = ["Z", "E", "N"] and keeps the FIRST channel
that passes CC >= 0.4, so most production S lags come from the VERTICAL. At T1/T2's steep
incidence S is mostly horizontal; a Z-channel "S" window can be dominated by P coda or
conversions phase-locked to the P arrival, and a lag measured on that would copy the P lag
(dtS == dtP) whatever the true separation -- the collapse the column test reported.

This re-measures a sample of within-cluster pairs with the production windowing, upsampling and
filter, but keeps the S lag on EACH channel separately, plus each channel's S-window energy
relative to the 0.25 s before the S pick (P coda). Readouts per channel:
  * Theil-Sen slope of observed S-P differential vs the catalog-predicted one
    (1 = real column, 0 = co-located);
  * median S/pre-S RMS ratio (is there S in the window at all?).

Usage (from the repository root):
    python full_catalog_pipeline/06_depth/sp_column_channel_check.py --array T1 --pairs-per-cluster 300
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
import warnings

import numpy as np
import pandas as pd
from obspy.signal.cross_correlation import xcorr_pick_correction
from sklearn.cluster import DBSCAN

import catalog_paths
import config
from diagnose_cc_p_deficit import (CC_MAXLAG, CC_MIN_COEFF, CC_TIME_AFTER, CC_TIME_BEFORE,
                                   load_phase_dat)
from hypodd_tune import RELOC_COLS
from lib.day_volume_index import load_day_file_index_csv
import regenerate_dtcc
from regenerate_dtcc import PAD_BEFORE, prefetch
from sp_differential_column_test import (EPS_KM, MIN_CLUSTER, MIN_SAMPLES, load_grids,
                                         local_xy, sp, theil_sen)

PRE_S = (0.30, 0.05)   # P-coda reference window, seconds before the S pick


def lag(w1, w2):
    t1 = w1.stats.starttime + PAD_BEFORE
    t2 = w2.stats.starttime + PAD_BEFORE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            corr, coeff = xcorr_pick_correction(t1, w1, t2, w2, t_before=CC_TIME_BEFORE,
                                                t_after=CC_TIME_AFTER, cc_maxlag=CC_MAXLAG,
                                                filter=None, plot=False)
        except Exception:
            return np.nan, np.nan
    return float(corr), float(coeff)


def s_ratio(tr):
    sr = tr.stats.sampling_rate
    i0 = int(PAD_BEFORE * sr)
    s = tr.data[i0 - int(CC_TIME_BEFORE * sr): i0 + int(CC_TIME_AFTER * sr)]
    pre = tr.data[i0 - int(PRE_S[0] * sr): i0 - int(PRE_S[1] * sr)]
    return float(np.sqrt(np.mean(s ** 2)) / (np.sqrt(np.mean(pre ** 2)) + 1e-20))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", required=True, choices=["T1", "T2"])
    ap.add_argument("--pairs-per-cluster", type=int, default=300)
    ap.add_argument("--upsample", type=int, default=2)
    args = ap.parse_args()
    A = args.array
    rng = np.random.default_rng(1)

    cat = pd.read_csv(catalog_paths.reloc(A), sep=r"\s+", header=None, names=RELOC_COLS)
    sta = pd.read_csv(catalog_paths.station_sel(A), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    lat0, lon0 = cat.lat.mean(), cat.lon.mean()
    cat["ex"], cat["ey"] = local_xy(cat.lat, cat.lon, lat0, lon0)
    sx, sy = local_xy(sta.lat, sta.lon, lat0, lon0)
    # phase.dat station names carry no network prefix in load_phase_dat's keys? normalise both
    stx = {s.split(".")[-1]: x for s, x in zip(sta.sta, sx)}
    sty = {s.split(".")[-1]: y for s, y in zip(sta.sta, sy)}
    cat["hc"] = DBSCAN(eps=EPS_KM, min_samples=MIN_SAMPLES).fit_predict(cat[["ex", "ey"]].values)
    c = cat.set_index("id")
    grids = load_grids(A)

    ph = load_phase_dat(catalog_paths.phase_dat(A))
    work, needed = [], {}
    for hc, ev in cat[cat.hc >= 0].groupby("hc"):
        if len(ev) < MIN_CLUSTER:
            continue
        ids = ev.id.values
        cand = set()
        tries = 0
        while len(cand) < args.pairs_per_cluster and tries < 50 * args.pairs_per_cluster:
            a, b = rng.choice(ids, 2, replace=False)
            tries += 1
            if a in ph and b in ph:
                cand.add((min(a, b), max(a, b)))
        for a, b in cand:
            stas = [k[0] for k in ph[a] if isinstance(k, tuple) and k[1] == "P"
                    and (k[0], "S") in ph[a] and (k[0], "P") in ph[b] and (k[0], "S") in ph[b]]
            for s in stas:
                work.append((hc, a, b, s))
                for e in (a, b):
                    for p in ("P", "S"):
                        needed[(e, s, p)] = ph[e]["origin"] + ph[e][(s, p)]
    print(f"{A}: {len(work)} (pair, station) items, {len(needed)} windows")
    win = prefetch(needed, load_day_file_index_csv(config.DAY_FILE_INDEX_CSV), args.upsample,
                   config.SAMPLE_RATE_HZ)

    rows = []
    for hc, a, b, s in work:
        kp1, kp2, ks1, ks2 = (a, s, "P"), (b, s, "P"), (a, s, "S"), (b, s, "S")
        if not all(k in win for k in (kp1, kp2, ks1, ks2)):
            continue
        if "Z" not in win[kp1] or "Z" not in win[kp2]:
            continue
        cP, wP = lag(win[kp1]["Z"], win[kp2]["Z"])
        if not np.isfinite(cP) or wP < CC_MIN_COEFF:
            continue
        ttp = ph[a][(s, "P")] - ph[b][(s, "P")]
        tts = ph[a][(s, "S")] - ph[b][(s, "S")]
        dtP = ttp - cP
        key = s.split(".")[-1]
        r_a = np.hypot(c.ex[a] - stx[key], c.ey[a] - sty[key])
        r_b = np.hypot(c.ex[b] - stx[key], c.ey[b] - sty[key])
        pred = float(sp(grids, np.array([c.depth[a]]), np.array([r_a]))[0]
                     - sp(grids, np.array([c.depth[b]]), np.array([r_b]))[0])
        row = dict(cluster=hc, i=a, j=b, sta=s, pred=pred, dz_cat=c.depth[a] - c.depth[b],
                   d_pick=tts - ttp, wP=wP)
        for ch in ("Z", "E", "N"):
            if ch in win[ks1] and ch in win[ks2]:
                cS, wS = lag(win[ks1][ch], win[ks2][ch])
                row[f"d_{ch}"] = (tts - cS) - dtP if np.isfinite(cS) else np.nan
                row[f"w_{ch}"] = wS
                row[f"ratio_{ch}"] = np.sqrt(s_ratio(win[ks1][ch]) * s_ratio(win[ks2][ch]))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(catalog_paths.work_dir(A), f"{A.lower()}_sp_column_channel_check.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")

    def summarise(g, label):
        line = [f"{label:>10} n={len(g):5d}"]
        line.append(f"pick-only slope {theil_sen(g.pred.values, g.d_pick.values):+.2f}")
        for ch in ("Z", "E", "N"):
            ok = g[f"w_{ch}"] >= CC_MIN_COEFF
            if ok.sum() < 20:
                line.append(f"{ch}: n/a")
                continue
            gg = g[ok]
            line.append(f"{ch}: n={ok.sum()} slope {theil_sen(gg.pred.values, gg[f'd_{ch}'].values):+.2f} "
                        f"S/preS {gg[f'ratio_{ch}'].median():.2f}")
        print("  ".join(line))

    summarise(df, "ALL")
    for hc, g in df.groupby("cluster"):
        summarise(g, f"cl {hc}")


if __name__ == "__main__":
    main()
