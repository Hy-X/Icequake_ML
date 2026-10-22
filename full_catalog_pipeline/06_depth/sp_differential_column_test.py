#!/usr/bin/env python3
"""Are the vertical "chimneys" in the relocated catalog real columns, or repeating sources
smeared vertically?

The observable is the cross-correlation S-P DIFFERENTIAL. For a pair (i, j) seen at station k
with both a P and an S CC lag,

    d_k = dtS_k - dtP_k = [tS_i - tS_j] - [tP_i - tP_j]

The origin-time difference cancels (it shifts P and S equally), and so does the station term
(same station, both phases' static cancels in the pair difference). What is left depends only on
where the two sources are relative to each other, mostly on their depth difference: for steep
rays d ~ dz * cos(i) * (1/Vs - 1/Vp), ~75 ms per 300 m in ice -- 50-100x the CC lag precision.
Before v6 the dt.cc was ~98% S, so this could not be formed; the x2 upsampled v6 dt.cc is ~54% P.

  * Real column: d_k tracks the catalog's dz (pair-level slope ~1).
  * Smeared repeater: d_k ~ 0 whatever the catalog's dz (slope ~0), and the S-P-only depth
    inversion collapses the column.

Three readouts, per horizontally-defined cluster (DBSCAN on EPICENTRES only, so a column is
not split by the very depth spread being tested):
  1. pair-level: median-over-stations observed d vs the catalog-predicted d, Theil-Sen slope.
  2. z-only S-P inversion: re-solve every event's depth from d alone (epicentres fixed at the
     catalog), robust IRLS, mean depth pinned to the catalog's. Compare depth spread.
     Controls: POSITIVE (synthetic d predicted from the catalog + matched noise -- must recover
     the catalog spread) and NEGATIVE (all events at the cluster's median depth + same noise --
     must collapse). Real data are only interpretable between those two.
  3. CC coefficient vs catalog vertical separation for horizontally-close pairs (< 50 m apart):
     waveforms of sources more than ~a quarter wavelength apart (~50-100 m for S at 10-30 Hz)
     cannot stay near-identical, so mean CC flat out to hundreds of metres means repeaters.

Caveat: hypoDD saw these same dt.cc data (alongside dt.ct and damping), so readout 1 is not
independent of the relocation. Readout 2 drops dt.ct entirely and readout 3 does not use lags.

Usage (from the repository root):
    python full_catalog_pipeline/06_depth/sp_differential_column_test.py --array T1
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
from pyrocko import cake
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from sklearn.cluster import DBSCAN

import catalog_paths
from hypodd_tune import RELOC_COLS
from test_sp_absolute_depth import build_cake_model, first_arrival_grid, interp
from vels1d_model import ICE_THICKNESS_KM

EPS_KM, MIN_SAMPLES = 0.15, 12      # same DBSCAN scale as the basal-cluster scripts
MIN_CLUSTER = 30
CLOSE_H_KM = 0.05                   # "horizontally co-located" for readout 3
DZ_FD_KM = 0.01                     # finite-difference step for d/dz
N_IRLS = 6
RNG = np.random.default_rng(0)


def load_grids(array):
    cache = os.path.join(catalog_paths.scratch_dir("sp_column_test"), f"{array}_ttgrid.npz")
    if os.path.exists(cache):
        g = np.load(cache)
        return g["zs"], g["rs"], g["gp"], g["gs"]
    model = build_cake_model(array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, zs=zs, rs=rs, gp=gp, gs=gs)
    return zs, rs, gp, gs


def load_dtcc(path):
    rows, i, j = [], None, None
    for line in open(path):
        if line[0] == "#":
            _, a, b = line.split()[:3]
            i, j = int(a), int(b)
            continue
        sta, dt, w, ph = line.split()
        rows.append((i, j, sta, float(dt), float(w), ph))
    df = pd.DataFrame(rows, columns=["i", "j", "sta", "dt", "w", "ph"])
    p = df[df.ph == "P"].set_index(["i", "j", "sta"])
    s = df[df.ph == "S"].set_index(["i", "j", "sta"])
    both = p[["dt", "w"]].join(s[["dt", "w"]], lsuffix="P", rsuffix="S", how="inner").reset_index()
    both["d"] = both.dtS - both.dtP
    return both


def local_xy(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111.195 * np.cos(np.radians(lat0)), (lat - lat0) * 111.195)


def sp(grids, z, r):
    zs, rs, gp, gs = grids
    return interp(zs, rs, gs, z, r) - interp(zs, rs, gp, z, r)


def invert_z(obs, ev_index, z0, ex, ey, stx, sty, grids, d):
    """Robust z-only inversion of S-P differentials; epicentres fixed, mean depth pinned."""
    z = z0.copy()
    z_ref = z0.mean()
    ii = ev_index[obs.i.values]
    jj = ev_index[obs.j.values]
    sx, sy = stx[obs.sta.values].values, sty[obs.sta.values].values
    ri = np.hypot(ex[ii] - sx, ey[ii] - sy)
    rj = np.hypot(ex[jj] - sx, ey[jj] - sy)
    n, m = len(z), len(obs)
    w = np.ones(m)
    for _ in range(N_IRLS):
        pred = sp(grids, z[ii], ri) - sp(grids, z[jj], rj)
        gi = (sp(grids, z[ii] + DZ_FD_KM, ri) - sp(grids, z[ii] - DZ_FD_KM, ri)) / (2 * DZ_FD_KM)
        gj = (sp(grids, z[jj] + DZ_FD_KM, rj) - sp(grids, z[jj] - DZ_FD_KM, rj)) / (2 * DZ_FD_KM)
        res = d - pred
        scale = 1.4826 * np.median(np.abs(res)) + 1e-4
        u = np.abs(res) / (1.345 * scale)
        w = np.where(u <= 1, 1.0, 1.0 / u)              # Huber
        sw = np.sqrt(w)
        rows = np.concatenate([np.arange(m), np.arange(m), np.full(n, m)])
        cols = np.concatenate([ii, jj, np.arange(n)])
        # mean-dz = 0 row. S-P differentials carry no absolute-depth information, so this row
        # alone fixes the common shift; it must outweigh the data (a 1 m mean shift costs ~1).
        vals = np.concatenate([gi * sw, -gj * sw, np.full(n, 1000.0 / n)])
        A = coo_matrix((vals, (rows, cols)), shape=(m + 1, n)).tocsr()
        b = np.concatenate([res * sw, [0.0]])
        # Real damping: an event with few observations is otherwise nearly free and takes a
        # huge step that the per-event clip below then makes asymmetric, walking the whole
        # cluster off as a block. Re-centre after clipping for the same reason.
        dz = lsqr(A, b, damp=0.05, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
        z = np.clip(z + np.clip(dz, -0.3, 0.3), 0.05, 4.9)
        z += z_ref - z.mean()
    pred = sp(grids, z[ii], ri) - sp(grids, z[jj], rj)
    return z, d - pred


def theil_sen(x, y, nmax=3000):
    if len(x) > nmax:
        k = RNG.choice(len(x), nmax, replace=False)
        x, y = x[k], y[k]
    a, b = np.triu_indices(len(x), 1)
    dx = x[b] - x[a]
    ok = np.abs(dx) > 1e-3
    return float(np.median((y[b] - y[a])[ok] / dx[ok]))


def spread(z):
    return np.percentile(z, 75) - np.percentile(z, 25), np.percentile(z, 95) - np.percentile(z, 5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", required=True, choices=["T1", "T2"])
    args = ap.parse_args()
    A = args.array
    work = catalog_paths.work_dir(A)
    indir = catalog_paths.input_dir(A)
    ccpath = os.path.join(indir, "dt.cc.authoritative")
    if not os.path.exists(ccpath):
        ccpath = catalog_paths.dt_cc(A)
    print(f"{A}: reloc {catalog_paths.reloc(A)}\n{A}: dt.cc {ccpath}")

    cat = pd.read_csv(catalog_paths.reloc(A), sep=r"\s+", header=None, names=RELOC_COLS)
    sta = pd.read_csv(catalog_paths.station_sel(A), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    lat0, lon0 = cat.lat.mean(), cat.lon.mean()
    cat["ex"], cat["ey"] = local_xy(cat.lat, cat.lon, lat0, lon0)
    sx, sy = local_xy(sta.lat, sta.lon, lat0, lon0)
    stx, sty = dict(zip(sta.sta, sx)), dict(zip(sta.sta, sy))
    stx, sty = pd.Series(stx), pd.Series(sty)

    cat["hc"] = DBSCAN(eps=EPS_KM, min_samples=MIN_SAMPLES).fit_predict(cat[["ex", "ey"]].values)
    grids = load_grids(A)

    both = load_dtcc(ccpath)
    ids = set(cat.id)
    both = both[both.i.isin(ids) & both.j.isin(ids) & both.sta.isin(stx.index)].copy()
    print(f"P+S same-station CC observations on relocated events: {len(both)} "
          f"({both[['i','j']].drop_duplicates().shape[0]} pairs)")

    c = cat.set_index("id")
    zi, zj = c.depth.reindex(both.i).values, c.depth.reindex(both.j).values
    ri = np.hypot(c.ex.reindex(both.i).values - stx[both.sta].values,
                  c.ey.reindex(both.i).values - sty[both.sta].values)
    rj = np.hypot(c.ex.reindex(both.j).values - stx[both.sta].values,
                  c.ey.reindex(both.j).values - sty[both.sta].values)
    both["pred"] = sp(grids, zi, ri) - sp(grids, zj, rj)
    both["dz_cat"] = zi - zj
    both["dh_cat"] = np.hypot(c.ex.reindex(both.i).values - c.ex.reindex(both.j).values,
                              c.ey.reindex(both.i).values - c.ey.reindex(both.j).values)
    both["hc_i"] = c.hc.reindex(both.i).values
    both["hc_j"] = c.hc.reindex(both.j).values
    both["cc"] = np.sqrt(both.wP * both.wS)

    rows, fig_data = [], []
    for hc in sorted(set(cat.hc) - {-1}):
        ev = cat[cat.hc == hc]
        if len(ev) < MIN_CLUSTER:
            continue
        ob = both[(both.hc_i == hc) & (both.hc_j == hc)]
        if len(ob) < 50:
            continue
        # readout 1: pair level
        pr = ob.groupby(["i", "j"]).agg(d=("d", "median"), pred=("pred", "median"),
                                        dz=("dz_cat", "first"), dh=("dh_cat", "first"),
                                        cc=("cc", "mean"), n=("d", "size")).reset_index()
        slope = theil_sen(pr.pred.values, pr.d.values)
        r = np.corrcoef(pr.pred, pr.d)[0, 1]
        # readout 2: z-only inversion, real + controls
        evi = pd.Series(np.arange(len(ev)), index=ev.id.values)
        ex, ey, z0 = ev.ex.values, ev.ey.values, ev.depth.values
        z_real, res = invert_z(ob, evi, np.full(len(ev), np.median(z0)), ex, ey, stx, sty,
                               grids, ob.d.values)
        noise = 1.4826 * np.median(np.abs(res))
        z_pos, _ = invert_z(ob, evi, np.full(len(ev), np.median(z0)), ex, ey, stx, sty, grids,
                            ob.pred.values + RNG.normal(0, noise, len(ob)))
        z_neg, _ = invert_z(ob, evi, np.full(len(ev), np.median(z0)), ex, ey, stx, sty, grids,
                            RNG.normal(0, noise, len(ob)))
        # readout 3: CC vs vertical separation, horizontally-close pairs
        close = pr[pr.dh < CLOSE_H_KM]
        cc_near = close[close.dz.abs() < 0.05].cc.mean()
        cc_far = close[close.dz.abs() > 0.2].cc.mean()
        n_far = int((close.dz.abs() > 0.2).sum())
        row = dict(cluster=hc, n_ev=len(ev), n_pairs=len(pr), n_obs=len(ob),
                   ext_h_m=1000 * np.hypot(*np.ptp(np.c_[ex, ey], axis=0)) ,
                   cat_iqr_m=1000 * spread(z0)[0], cat_p90_m=1000 * spread(z0)[1],
                   sp_iqr_m=1000 * spread(z_real)[0], sp_p90_m=1000 * spread(z_real)[1],
                   pos_iqr_m=1000 * spread(z_pos)[0], neg_iqr_m=1000 * spread(z_neg)[0],
                   z_r=np.corrcoef(z0, z_real)[0, 1],
                   pair_slope=slope, pair_r=r, noise_ms=1000 * noise,
                   cc_close_dz_lt50m=cc_near, cc_close_dz_gt200m=cc_far, n_close_far=n_far,
                   depth_med_km=np.median(z0))
        rows.append(row)
        fig_data.append((hc, ev, pr, z0, z_real, z_pos, z_neg, close))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    out = pd.DataFrame(rows)
    csv = os.path.join(work, f"{A.lower()}_sp_differential_column_test.csv")
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")

    k = len(fig_data)
    fig, axes = plt.subplots(k, 3, figsize=(15, 3.6 * k), squeeze=False)
    for row_ax, (hc, ev, pr, z0, zr, zp, zn, close) in zip(axes, fig_data):
        a = row_ax[0]
        a.scatter(1000 * pr.pred, 1000 * pr.d, s=4, alpha=0.3, color="#2a78d6", linewidth=0)
        lim = np.percentile(np.abs(1000 * pr.pred), 99) * 1.1 + 5
        a.plot([-lim, lim], [-lim, lim], color="#0b0b0b", lw=1, ls="--", label="real column (1:1)")
        a.axhline(0, color="#eb6834", lw=1.2, label="co-located repeater (0)")
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_xlabel("catalog-predicted S-P differential (ms)")
        a.set_ylabel("observed CC S-P differential (ms)")
        a.set_title(f"cluster {hc} (n={len(ev)}): pair slope "
                    f"{theil_sen(pr.pred.values, pr.d.values):.2f}", fontsize=10)
        a.legend(fontsize=8, loc="upper left")
        a = row_ax[1]
        bins = np.linspace(min(z0.min(), zr.min()), max(z0.max(), zr.max()), 40) * 1000
        for z, lab, col in [(z0, "catalog (hypoDD)", "#8a8a86"), (zr, "S-P CC only", "#2a78d6"),
                            (zp, "+control: catalog synth", "#3f9b5c"),
                            (zn, "-control: one depth + noise", "#eb6834")]:
            a.hist(1000 * z, bins=bins, histtype="step", lw=1.6, color=col, label=lab)
        a.axvline(1000 * ICE_THICKNESS_KM[A], color="#0b0b0b", ls=":", lw=1, label="ice bed")
        a.set_xlabel("depth (m)"); a.set_ylabel("events"); a.legend(fontsize=7)
        a.set_title("depth re-solved from S-P CC differentials alone", fontsize=10)
        a = row_ax[2]
        a.scatter(1000 * close.dz.abs(), close.cc, s=5, alpha=0.4, color="#2a78d6", linewidth=0)
        a.set_xlabel(f"catalog |dz| of pairs < {int(CLOSE_H_KM*1000)} m apart horizontally (m)")
        a.set_ylabel("mean CC coefficient (P,S)")
        a.set_title("waveform similarity vs catalog vertical separation", fontsize=10)
    fig.suptitle(f"{A}: chimney test -- real vertical column or smeared repeating source?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png = os.path.join(work, f"{A.lower()}_sp_differential_column_test.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
