#!/usr/bin/env python3
"""Bed-reflection test: is a repeating-source cluster AT the bed or some height above it?

A source a height h above the ice bed sends a P wave down to the bed that reflects back up
(PbP) and reaches a surface station after the direct P. The delay grows with h (~2h/Vp near
vertical incidence) and shrinks with epicentral distance. A source AT the bed has no separate
reflection: it merges with the direct P. This uses only P (no S picks) and depends on velocity
only through the ice Vp, which the reflection profile measured.

Per cluster and station: read Z windows from the day archive, align on P by cross-correlation
against an iterated stack, then form a phase-weighted stack (PWS) to suppress incoherent
coda. The PWS envelope is scanned for the PbP moveout predicted for each trial source depth,
by 1D layered ray shooting (vels1d layers, reflector at the ice base). As a control, the
receiver-side surface-bed multiple (independent of source depth) is marked as well.

SENSITIVITY (measured with --inject, 2026-10-08, T2): the PWS is nonlinear and suppresses a
reflection weaker than the per-trace coda -- a 3% injected PbP stacks to a 0.005 envelope, 10%
survives at ~0.08. A linear stack does not help much: repeaters share a COHERENT P coda (median
2-4%, p95 7-10% of the P peak), which more stacking cannot reduce. So a null result here only
excludes reflections stronger than ~10% of P, and nothing inside the P-pulse blind zone.

Usage (from the repository root):
    python full_catalog_pipeline/06_depth/bed_reflection_stack.py --array T2 \
        --reloc <hypoDD.reloc> --clusters 1 2 3 4
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
from scipy.signal import find_peaks, hilbert

import catalog_paths
import config
import vels1d_model
from diagnose_cc_p_deficit import CC_FMAX, CC_FMIN, load_phase_dat
from hypodd_tune import RELOC_COLS, model_lines
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window
from sp_differential_column_test import local_xy

PRE, POST = 0.3, 1.6          # seconds around the P pick
P_WIN = (-0.05, 0.15)         # alignment window around P
MAXLAG = 0.025                # alignment search, seconds
MIN_CC = 0.6                  # traces below this after alignment are dropped
PWS_NU = 2.0
BLIND_LEVEL = 0.15            # P-pulse envelope above this hides any reflection ("blind zone")
MIN_PROM = 0.02               # a reflection candidate is an envelope peak this prominent (P peak = 1)
MATCH_S = 0.010               # and within this of the predicted arrival


def layers(array):
    tops, vps, _ = model_lines(array, "vels1d")
    return np.array(tops.split(), float), np.array(vps.split(), float)


def leg(tops, vps, z0, z1, p):
    """Horizontal distance and time for a ray of slowness p crossing depths z0..z1 (km)."""
    a, b = min(z0, z1), max(z0, z1)
    x = t = 0.0
    bounds = list(tops) + [1e9]
    for i, v in enumerate(vps):
        lo, hi = max(a, bounds[i]), min(b, bounds[i + 1])
        if hi <= lo:
            continue
        q = 1.0 - (p * v) ** 2
        if q <= 0:
            return np.nan, np.nan
        dz = hi - lo
        x += dz * p * v / np.sqrt(q)
        t += dz / (v * np.sqrt(q))
    return x, t


def shoot(tops, vps, segments, r):
    """Travel time of a ray through the given (z0, z1) segments arriving at distance r (km)."""
    vmax = max(vps[(tops < max(max(s) for s in segments))])
    lo, hi = 0.0, 1.0 / vmax - 1e-9
    for _ in range(60):
        p = 0.5 * (lo + hi)
        x = sum(leg(tops, vps, a, b, p)[0] for a, b in segments)
        if np.isnan(x) or x > r:
            hi = p
        else:
            lo = p
    return sum(leg(tops, vps, a, b, lo)[1] for a, b in segments)


def delays(tops, vps, H, z, r):
    """(PbP - P, receiver surface-bed multiple - P), seconds."""
    tp = shoot(tops, vps, [(z, 0.0)], r)
    tpbp = shoot(tops, vps, [(z, H), (H, 0.0)], r)
    tmult = shoot(tops, vps, [(z, 0.0), (0.0, H), (H, 0.0)], r)
    return tpbp - tp, tmult - tp


def read_z(day_index, sta, times, sr):
    cache = RollingDayCache(day_index, sta)
    out = []
    for t in times:
        data, _ = extract_window(cache, {"primary_date": t.date.isoformat()}, t - PRE, t + POST)
        if data is None:
            out.append(None)
            continue
        tr = Trace(np.asarray(data["Z"], float))
        tr.stats.sampling_rate = sr
        tr.detrend("demean")
        tr.taper(0.05)
        tr.filter("bandpass", freqmin=CC_FMIN, freqmax=CC_FMAX, zerophase=True)
        out.append(tr.data[: int((PRE + POST) * sr)])
    return out


def align_and_stack(traces, sr):
    n = min(len(x) for x in traces)
    X = np.array([x[:n] for x in traces])
    i0, i1 = int((PRE + P_WIN[0]) * sr), int((PRE + P_WIN[1]) * sr)
    X /= np.abs(X[:, i0:i1]).max(axis=1, keepdims=True) + 1e-20
    m = int(MAXLAG * sr)
    keep = np.ones(len(X), bool)
    for _ in range(3):
        ref = X[keep].mean(axis=0)[i0:i1]
        ccs = np.zeros(len(X))
        for k in range(len(X)):
            best, bk = -2, 0
            for s in range(-m, m + 1):
                seg = X[k, i0 + s:i1 + s]
                c = np.dot(seg, ref) / (np.linalg.norm(seg) * np.linalg.norm(ref) + 1e-20)
                if c > best:
                    best, bk = c, s
            X[k] = np.roll(X[k], -bk)
            ccs[k] = best
        keep = ccs >= MIN_CC
    Y = X[keep]
    lin = Y.mean(axis=0)
    phase = np.exp(1j * np.angle(hilbert(Y, axis=1)))
    coh = np.abs(phase.mean(axis=0)) ** PWS_NU
    pws = lin * coh
    env = np.abs(hilbert(pws))
    return env / env[i0:i1].max(), coh, int(keep.sum()), float(np.median(ccs[keep]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--reloc", default=None, help="hypoDD.reloc (default: authoritative)")
    ap.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--max-events", type=int, default=400)
    ap.add_argument("--tag", default="")
    ap.add_argument("--inject", type=float, default=0.0,
                    help="detection test: add each trace's own P pulse, scaled to this fraction of "
                         "its P peak, at the PbP delay predicted for the cluster's relocated depth "
                         "(+-2 ms random jitter per trace) BEFORE alignment and stacking")
    args = ap.parse_args()
    A = args.array
    sr = config.SAMPLE_RATE_HZ
    H = vels1d_model.ICE_THICKNESS_KM[A]
    tops, vps = layers(A)

    ph = load_phase_dat(catalog_paths.phase_dat(A))
    cat = pd.read_csv(args.reloc or catalog_paths.reloc(A), sep=r"\s+", header=None,
                      names=RELOC_COLS)
    sta = pd.read_csv(catalog_paths.station_sel(A), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    lat0, lon0 = cat.lat.mean(), cat.lon.mean()
    cat["ex"], cat["ey"] = local_xy(cat.lat, cat.lon, lat0, lon0)
    sx, sy = local_xy(sta.lat, sta.lon, lat0, lon0)
    pos = {s.split(".")[-1]: (x, y) for s, x, y in zip(sta.sta, sx, sy)}
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)

    zgrid = np.arange(0.10, H - 0.005, 0.01)
    rows = []
    fig, axes = plt.subplots(len(args.clusters), 2, figsize=(15, 4.2 * len(args.clusters)),
                             gridspec_kw=dict(width_ratios=[3, 1]), squeeze=False)
    for (ax, axs), c in zip(axes, args.clusters):
        ev = cat[cat.cid == c]
        cx, cy, zc = ev.ex.median(), ev.ey.median(), ev.depth.median()
        stacks = {}
        for s in sorted(pos):
            have = [e for e in ev.sort_values("nccp", ascending=False).id
                    if e in ph and (s, "P") in ph[e]][: args.max_events]
            if len(have) < 30:
                continue
            have = sorted(have, key=lambda e: ph[e]["origin"])
            times = [ph[e]["origin"] + ph[e][(s, "P")] for e in have]
            tr = [x for x in read_z(day_index, s, times, sr) if x is not None and len(x) > 0]
            if len(tr) < 30:
                continue
            if args.inject > 0:
                r_in = float(np.hypot(cx - pos[s][0], cy - pos[s][1]))
                dl_in = delays(tops, vps, H, zc, r_in)[0]
                rng = np.random.default_rng(0)
                a0, a1 = int((PRE - 0.02) * sr), int((PRE + 0.12) * sr)
                for x in tr:
                    pulse = x[a0:a1] / (np.abs(x[a0:a1]).max() + 1e-20) * np.abs(x[a0:a1]).max()
                    k = a0 + int(round((dl_in + rng.uniform(-0.002, 0.002)) * sr))
                    m = min(len(pulse), len(x) - k)
                    if m > 0:
                        x[k:k + m] += args.inject * pulse[:m]
            env, coh, n, mcc = align_and_stack(tr, sr)
            sp_obs = np.median([ph[e][(s, "S")] - ph[e][(s, "P")] for e in have if (s, "S") in ph[e]]
                               or [np.nan])
            r = float(np.hypot(cx - pos[s][0], cy - pos[s][1]))
            # Delays are measured from the P envelope PEAK (a reflection is the same pulse,
            # shifted), and nothing is scored until the P pulse has decayed below BLIND_LEVEL.
            ipk = int((PRE + P_WIN[0]) * sr) + int(np.argmax(env[int((PRE + P_WIN[0]) * sr):int((PRE + P_WIN[1]) * sr)]))
            ib = ipk + int(np.argmax(env[ipk:] < BLIND_LEVEL))
            tpk, tblind = ipk / sr - PRE, ib / sr - PRE
            iend = int((PRE + (sp_obs - 0.05 if np.isfinite(sp_obs) else POST - 0.1)) * sr)
            noise = float(np.median(env[ib:iend])) if iend > ib + 10 else np.nan
            stacks[s] = dict(env=env, n=n, mcc=mcc, r=r, sp=sp_obs, tpk=tpk, tblind=tblind,
                             noise=noise)
            print(f"cluster {c} {s}: r={r:.2f} km  stacked {n}/{len(tr)} (median cc {mcc:.2f})  "
                  f"S-P {sp_obs:.3f} s  P peak +{1000*tpk:.0f} ms, blind until +{1000*tblind:.0f} ms, "
                  f"noise floor {noise:.3f}")
        if not stacks:
            continue
        t = np.arange(len(next(iter(stacks.values()))["env"])) / sr - PRE

        # depth scan: mean normalised PWS envelope at the predicted PbP delay
        # score = fraction of eligible stations (prediction between blind zone and S) that have a
        # distinct envelope peak (prominence >= MIN_PROM) within MATCH_S of the predicted PbP.
        # A real reflector height should light up several stations at once along its moveout.
        for d in stacks.values():
            pk, pr = find_peaks(d["env"], prominence=MIN_PROM)
            d["peaks_t"] = pk / sr - PRE
            d["peaks_p"] = pr["prominences"]
        score = np.full(len(zgrid), np.nan)
        nsta = np.zeros(len(zgrid), int)
        for iz, z in enumerate(zgrid):
            vals = []
            for s, d in stacks.items():
                dl, _ = delays(tops, vps, H, z, d["r"])
                ta = d["tpk"] + dl
                if ta < d["tblind"] or (np.isfinite(d["sp"]) and ta > d["sp"] - 0.05):
                    continue
                k = int(round((PRE + ta) * sr))
                hit = np.abs(d["peaks_t"] - ta) <= MATCH_S
                vals.append(1.0 if hit.any() else 0.0)
            nsta[iz] = len(vals)
            if len(vals) >= 2:
                score[iz] = np.mean(vals)
        print("   candidate peaks (s after P, prominence): " + "; ".join(
            f"{s} " + ",".join(f"{t:.3f}({p:.2f})" for t, p in zip(d["peaks_t"], d["peaks_p"])
                               if d["tblind"] < t < (d["sp"] if np.isfinite(d["sp"]) else POST))
            for s, d in stacks.items()))
        ok = np.isfinite(score)
        zbest = zgrid[ok][np.nanargmax(score[ok])] if ok.any() else np.nan
        # deepest trial depth whose reflection is still outside the blind zone at >= 2 stations
        zvis = zgrid[nsta >= 2].max() if (nsta >= 2).any() else np.nan
        for s, d in stacks.items():
            _, dm = delays(tops, vps, H, H - 0.01, d["r"])
            k = int(round((PRE + d["tpk"] + dm) * sr))
            d["mult_snr"] = (d["env"][max(k - 3, 0):k + 4].max()
                             if k < len(d["env"]) else np.nan)   # amplitude rel. to P peak

        order = sorted(stacks, key=lambda s: stacks[s]["r"])
        for i, s in enumerate(order):
            d = stacks[s]
            y0 = -1.2 * i
            ax.plot(t, d["env"] + y0, color="#15616d", lw=0.9)
            ax.text(t[0], y0 + 0.4, f"{s}  r={d['r']:.1f} km  n={d['n']}", fontsize=8)
            for z, col, lab in [(zc, "#eb6834", f"reloc depth {zc:.2f} km"),
                                (zbest, "#2a78d6", f"scan max {zbest:.2f} km")]:
                if np.isfinite(z):
                    dl, _ = delays(tops, vps, H, z, d["r"])
                    ax.plot([d["tpk"] + dl] * 2, [y0, y0 + 1], color=col, lw=1.6,
                            label=lab if i == 0 else None)
            ax.axvspan(-0.1, d["tblind"], ymin=0, ymax=0, color="none")
            ax.fill_between([d["tpk"], d["tblind"]], y0, y0 + 1, color="#b9b8b2", alpha=0.35,
                            label="P-pulse blind zone" if i == 0 else None)
            dm = d["tpk"] + delays(tops, vps, H, H - 0.01, d["r"])[1]
            ax.plot([dm, dm], [y0, y0 + 1], color="#52514e", ls=":", lw=1.2,
                    label="receiver surface-bed multiple (control)" if i == 0 else None)
            if np.isfinite(d["sp"]):
                ax.plot([d["sp"], d["sp"]], [y0, y0 + 1], color="k", lw=1,
                        label="median S pick" if i == 0 else None)
            rows.append(dict(cluster=c, station=s, r_km=d["r"], n_stacked=d["n"],
                             median_cc=d["mcc"], sp_s=d["sp"],
                             pbp_delay_reloc_s=delays(tops, vps, H, zc, d["r"])[0],
                             pbp_delay_best_s=delays(tops, vps, H, zbest, d["r"])[0]
                             if np.isfinite(zbest) else np.nan,
                             multiple_delay_s=dm, p_peak_s=d["tpk"], blind_until_s=d["tblind"],
                             noise_floor=d["noise"], multiple_snr=d["mult_snr"]))
        ax.set_xlim(-0.1, POST)
        ax.set_yticks([])
        ax.set_xlabel("time after P (s)")
        ax.set_title(f"cluster {c} (n={len(ev)}): PWS envelope per station, sorted by distance",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        axs.plot(score, zgrid, color="#2a78d6")
        if np.isfinite(zvis):
            axs.axhspan(zvis, H, color="#b9b8b2", alpha=0.35, label="blind (<2 stations)")
        axs.axhline(zc, color="#eb6834", lw=1.2, label="relocated depth")
        axs.axhline(H, color="k", ls="--", lw=1, label="ice bed")
        axs.invert_yaxis()
        axs.set_xlabel("fraction of stations with a peak on the PbP moveout")
        axs.set_xlim(-0.05, 1.05)
        axs.set_ylabel("trial source depth (km)")
        axs.legend(fontsize=7)
        sc_reloc = score[np.argmin(np.abs(zgrid - zc))]
        print(f"cluster {c}: relocated depth {zc:.2f} km (score {sc_reloc:.2f}); scan max "
              f"{zbest:.2f} km (score {np.nanmax(score):.2f}); blind below {zvis:.2f} km "
              f"(= within {1000*(H-zvis):.0f} m of the bed); control multiple amplitude (P=1) "
              + ", ".join(f"{s} {d['mult_snr']:.3f}" for s, d in stacks.items()))
    fig.suptitle(f"{A}: bed-reflected P (PbP) search on stacked repeaters", fontweight="bold")
    fig.tight_layout()
    stem = os.path.join(catalog_paths.work_dir(A), f"{A.lower()}_bed_reflection_stack{args.tag}")
    fig.savefig(stem + ".png", dpi=120, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(stem + ".csv", index=False)
    print(f"wrote {stem}.png / .csv")


if __name__ == "__main__":
    main()
