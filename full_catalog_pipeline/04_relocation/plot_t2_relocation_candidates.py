#!/usr/bin/env python3
"""Decision figures for promoting a T2 relocation candidate (2026-10-08).

Figure 1, t2_reloc_candidates_comparison.png -- one row per candidate:
  * map in the shear-margin frame (across vs along), with stations;
  * depth section across the margin, with the ice bed;
  * per-event absolute fit: median |P residual| of each event's raw phase.dat picks at its
    relocated position (own origin-time shift, vels1d cake grid), against the same events at
    their starting (pyocto vels1d) positions. Double-difference data cannot see a rigid
    translation; this can.

Figure 2, t2_reloc_cluster_drift.png -- centroid of the two largest clusters through hypoDD's 32
iterations (per-iteration hypoDD.reloc.<cluster>.<iter> files), for v6 as built and for
clean_ctS0_Panchor, with the location the raw picks support for cluster 1 (grid search over a
common location, P+S, per-event origin times).

Usage (from the repository root):
    python full_catalog_pipeline/04_relocation/plot_t2_relocation_candidates.py
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

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter

import catalog_paths
from hypodd_tune import RELOC_COLS
from plot_margin_oriented_sections import (C_ACCENT, INK, MUTED, SMOOTH_SIGMA_PX, SURF,
                                           margin_axes, paths, style)
from reanchor_clusters import TO_PS, read_event_sel
from sp_differential_column_test import load_grids
from test_sp_absolute_depth import interp, load_picks

A = "T2"
C_V6, C_PANCHOR, C_VALID = "#2a78d6", "#eb6834", "#15616d"
ROOT = os.path.dirname(catalog_paths.run_dir(A, "x"))
SRC = os.path.join(ROOT, "hypodd_v6_allfix")
CANDIDATES = [
    ("A. v6 official",
     os.path.join(SRC, "output_files", "hypoDD.reloc")),
    ("B. v6, re-anchored",
     os.path.join(ROOT, "hypodd_v6_reanchored", "output_files", "hypoDD.reloc")),
    ("C. catalog S off + P anchor, re-anchored",
     os.path.join(ROOT, "hypodd_v6_ctS0_Panchor_reanchored", "output_files", "hypoDD.reloc")),
    ("D1. bad S picks removed (unjudged kept) + P anchor, re-anchored",
     os.path.join(ROOT, "hypodd_v6_clean_ctSclean30_Panchor_reanchored", "output_files",
                  "hypoDD.reloc")),
    ("D2. only CC-validated S picks + P anchor, re-anchored",
     os.path.join(ROOT, "hypodd_v6_clean_ctSvalid30_Panchor_reanchored", "output_files",
                  "hypoDD.reloc")),
]
DRIFT_RUNS = [
    # (label, dir holding per-iteration files, final hypoDD.reloc, colour)
    ("v6 official", os.path.join(SRC, "input_files"),
     os.path.join(SRC, "output_files", "hypoDD.reloc"), C_V6),
    ("catalog S off + P anchor", os.path.join(SRC, "tune", "clean_ctS0_Panchor", "prod"),
     os.path.join(SRC, "tune", "clean_ctS0_Panchor", "prod", "hypoDD.reloc"), C_PANCHOR),
    ("validated S + P anchor (D2)", os.path.join(SRC, "tune", "clean_ctSvalid30_Panchor", "prod"),
     os.path.join(SRC, "tune", "clean_ctSvalid30_Panchor", "prod", "hypoDD.reloc"), C_VALID),
]
# BASE schedule in hypodd_tune.py: 6,6,6,6,8 iterations; wt_ct / damping per stage
STAGE_ENDS = [6, 12, 18, 24, 32]
STAGE_LABELS = ["CT 0.6\ndamp 40", "CT 0.4\ndamp 30", "CT 0.2\ndamp 15", "CT 0.05\ndamp 8",
                "CT 0.05\ndamp 5"]
ACROSS_LIM, ALONG_LIM = (-6.5, 2.0), (-6.0, 6.0)


class Frame:
    """Margin frame (same construction as plot_margin_strain_profile.py)."""

    def __init__(self):
        P = paths(A)
        sta = pd.read_csv(P["station_sel"], sep=r"\s+", header=None,
                          names=["id", "lat", "lon", "elev"])
        sx, sy = TO_PS.transform(sta.lon.values, sta.lat.values)
        self.cx, self.cy = sx.mean(), sy.mean()
        with rasterio.open(P["vx"]) as ds:
            vx = ds.read(1).astype(float)
            bounds = tuple(ds.bounds)
        with rasterio.open(P["vy"]) as ds:
            vy = ds.read(1).astype(float)
        vx[vx == -32767] = np.nan
        vy[vy == -32767] = np.nan
        speed = gaussian_filter(np.nan_to_num(np.hypot(vx, vy)), sigma=SMOOTH_SIGMA_PX)
        self.across, self.along, *_ = margin_axes(speed, bounds, self.cx, self.cy)
        self.sta = sta.assign(code=sta.id.str.split(".").str[1], x=sx, y=sy)
        self.bed = P["ice_bed_km"]

    def xy(self, x, y):
        d = np.c_[np.asarray(x) - self.cx, np.asarray(y) - self.cy] / 1000.0
        return d @ self.across, d @ self.along

    def ll(self, lon, lat):
        return self.xy(*TO_PS.transform(np.asarray(lon), np.asarray(lat)))


class Picks:
    def __init__(self, frame):
        self.zs, self.rs, self.gp, self.gs = load_grids(A)
        pk = load_picks(catalog_paths.phase_dat(A, "hypodd_v6_allfix"))
        pos = dict(zip(frame.sta.id, zip(frame.sta.x, frame.sta.y)))
        pk["sx"] = [pos[s][0] for s in pk.sta]
        pk["sy"] = [pos[s][1] for s in pk.sta]
        self.pk = pk

    def _res(self, q, ex, ey, ez):
        r = np.hypot(q.sx.values - ex, q.sy.values - ey) / 1000.0
        g = np.where(q.phase.values == "P", 1, 0).astype(bool)
        pred = np.where(g, interp(self.zs, self.rs, self.gp, ez, r),
                        interp(self.zs, self.rs, self.gs, ez, r))
        res = q.tt.values - pred
        return res - pd.Series(res).groupby(q.id.values).transform("median").values

    def per_event_p(self, ev):
        """Median |P residual| (ms) per event at its own location."""
        x, y = TO_PS.transform(ev.lon.values, ev.lat.values)
        L = pd.DataFrame(dict(id=ev.id.values, ex=x, ey=y, ez=ev.depth.values))
        q = self.pk.merge(L, on="id")
        q["res"] = self._res(q, q.ex.values, q.ey.values, q.ez.values)
        return q[q.phase == "P"].groupby("id").res.apply(lambda v: np.median(np.abs(v))) * 1000

    def best_common(self, ids, x0, y0):
        """Grid-search one common location for a cluster's picks (P+S)."""
        q = self.pk[self.pk.id.isin(ids)]
        best = (np.inf, x0, y0, 1.5)
        for step, half, zz in [(125, 2500, (0.8, 1.2, 1.5, 1.8)), (25, 150, None)]:
            bx, by, bz = best[1:]
            zlist = zz or np.arange(max(0.1, bz - 0.4), bz + 0.41, 0.1)
            for dx in np.arange(-half, half + 1, step):
                for dy in np.arange(-half, half + 1, step):
                    for z in zlist:
                        n = len(q)
                        v = np.median(np.abs(self._res(q, np.full(n, bx + dx), np.full(n, by + dy),
                                                       np.full(n, z))))
                        if v < best[0]:
                            best = (v, bx + dx, by + dy, z)
        return best


def read_reloc(path):
    e = pd.read_csv(path, sep=r"\s+", header=None, names=RELOC_COLS)
    return e.apply(pd.to_numeric, errors="coerce").dropna(subset=["lat", "lon", "depth"])


def fig_comparison(frame, picks, sel):
    start = sel.reset_index()
    start_res = picks.per_event_p(start)
    bins = np.logspace(0, 3, 50)
    fig, axes = plt.subplots(len(CANDIDATES), 3, figsize=(17, 4.6 * len(CANDIDATES)),
                             gridspec_kw=dict(width_ratios=[1.05, 1.5, 1.0]))
    fig.patch.set_facecolor(SURF)
    sa, sl = frame.xy(frame.sta.x, frame.sta.y)
    for row, (label, path) in zip(axes, CANDIDATES):
        ev = read_reloc(path)
        a, l = frame.ll(ev.lon, ev.lat)
        res = picks.per_event_p(ev)
        s_res = start_res.reindex(res.index)
        ax = row[0]
        ax.scatter(a, l, s=3, color=C_ACCENT, alpha=0.35, linewidths=0, zorder=3)
        ax.scatter(sa, sl, marker="^", s=70, color=INK, zorder=4)
        for code, x, y in zip(frame.sta.code, sa, sl):
            ax.annotate(code, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7,
                        color=MUTED)
        ax.set_xlim(*ACROSS_LIM); ax.set_ylim(*ALONG_LIM); ax.set_aspect("equal")
        ax.set_xlabel("across margin (km)"); ax.set_ylabel("along margin (km)")
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        style(ax)
        ax = row[1]
        ax.scatter(a, ev.depth, s=3, color=C_ACCENT, alpha=0.35, linewidths=0, zorder=3)
        ax.axhline(frame.bed, color=INK, ls="--", lw=1)
        ax.text(ACROSS_LIM[1] - 0.1, frame.bed - 0.05, f"ice bed {frame.bed:.2f} km", ha="right",
                fontsize=8, color=MUTED)
        ax.set_xlim(*ACROSS_LIM); ax.set_ylim(3.0, 0.0)
        ax.set_xlabel("across margin (km)"); ax.set_ylabel("depth (km)")
        ax.set_title(f"depth section, n={len(ev)}; {100*(ev.depth > frame.bed).mean():.1f}% below bed",
                     loc="left", fontsize=10, color=INK)
        style(ax)
        ax = row[2]
        ax.hist(s_res.dropna(), bins=bins, histtype="step", color=MUTED, lw=1.5,
                label=f"same events at start: median {s_res.median():.0f} ms")
        ax.hist(res, bins=bins, color=C_ACCENT, alpha=0.8,
                label=f"this catalog: median {res.median():.0f} ms")
        ax.axvline(50, color=MUTED, lw=0.8, ls=":")
        ax.text(52, ax.get_ylim()[1] * 0.92, f"{100*(res > 50).mean():.0f}% of events\n> 50 ms",
                fontsize=8, color=INK, va="top")
        ax.set_xscale("log")
        ax.set_xlabel("per-event median |P residual| vs raw picks (ms)")
        ax.set_ylabel("events")
        ax.set_title("fit to raw catalog P picks", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        style(ax)
        print(f"{label}: n={len(ev)} median |P res| {res.median():.1f} ms (start "
              f"{s_res.median():.1f}), >50 ms {100*(res > 50).mean():.1f}%")
    fig.suptitle("T2 relocation candidates: where the events sit, and whether the raw P picks "
                 "agree", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    out = os.path.join(SRC, "t2_reloc_candidates_comparison.png")
    fig.savefig(out, dpi=110, facecolor=SURF)
    print(f"wrote {out}")


def centroid_track(d, cid, frame):
    pts = []
    for it in range(1, 40):
        f = os.path.join(d, f"hypoDD.reloc.{cid:03d}.{it:03d}")
        if not os.path.exists(f):
            break
        e = read_reloc(f)
        a, l = frame.ll(e.lon, e.lat)
        pts.append((it, np.median(a), np.median(l), np.median(e.depth)))
    return pd.DataFrame(pts, columns=["it", "across", "along", "depth"])


def fig_drift(frame, picks, sel):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw=dict(width_ratios=[1.7, 1]))
    fig.patch.set_facecolor(SURF)
    best_done = False
    end_labels = []
    for name, d, final_path, col in DRIFT_RUNS:
        final = read_reloc(final_path)
        for rank, ls in [(0, "-"), (1, "--")]:
            cid = final.cid.value_counts().index[rank]
            n = int((final.cid == cid).sum())
            ids = final.id[final.cid == cid]
            s = sel.loc[sel.index.intersection(ids)]
            sa, sl = frame.ll(s.lon, s.lat)
            a0, l0 = np.median(sa), np.median(sl)
            tr = centroid_track(d, cid, frame)
            disp = np.hypot(tr.across - a0, tr.along - l0) * 1000
            ax1.plot(np.r_[0, tr.it], np.r_[0, disp], color=col, ls=ls, lw=2,
                     marker="o", ms=3)
            end_labels.append((disp.iloc[-1], tr.it.iloc[-1],
                               f"{name}, cluster {rank+1} (n={n}): {disp.iloc[-1]:.0f} m"))
            ax2.plot(np.r_[a0, tr.across], np.r_[l0, tr.along], color=col, ls=ls, lw=2)
            ax2.scatter([tr.across.iloc[-1]], [tr.along.iloc[-1]], color=col, s=50, zorder=5,
                        edgecolors=SURF, linewidths=1.5)
            ax2.scatter([a0], [l0], marker="s", color=MUTED, s=40, zorder=5)
            if rank == 0 and not best_done:
                x0, y0 = TO_PS.transform(np.median(s.lon), np.median(s.lat))
                v, bx, by, bz = picks.best_common(set(ids), x0, y0)
                ba, bl = frame.xy([bx], [by])
                ax2.scatter(ba, bl, marker="*", s=260, color=INK, zorder=6,
                            label=f"cluster 1 location the raw picks support "
                                  f"(P+S grid search, {v*1000:.0f} ms)")
                best_done = True
                print(f"cluster 1 pick-supported location: across {ba[0]:+.2f} along "
                      f"{bl[0]:+.2f} km, depth {bz:.2f} km ({v*1000:.1f} ms)")
    ytop = ax1.get_ylim()[1]
    placed = []
    for y, x, lab in sorted(end_labels):
        yy = max([y] + [p + 0.045 * ytop for p in placed])
        placed.append(yy)
        ax1.annotate(lab, (x, y), xytext=(x + 1.0, yy), fontsize=8, color=INK, va="center",
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    prev = 0
    for e, lab in zip(STAGE_ENDS, STAGE_LABELS):
        ax1.axvline(e + 0.5, color=MUTED, lw=0.6, ls=":")
        ax1.text((prev + e) / 2 + 0.5, ytop * 0.98, lab, fontsize=7, color=MUTED,
                 ha="center", va="top")
        prev = e
    ax1.set_xlim(0, 48)
    ax1.set_xlabel("hypoDD iteration")
    ax1.set_ylabel("cluster centroid distance from its starting position (m)")
    ax1.set_title("drift through the inversion (stage settings = v6 schedule; the P-anchor run "
                  "holds catalog-P weight at 0.6 throughout)", loc="left", fontsize=9, color=INK)
    style(ax1)
    ax2.scatter([], [], marker="s", color=MUTED, s=40, label="starting centroid (pyocto)")
    ax2.plot([], [], color=C_V6, lw=2, label="v6 official track")
    ax2.plot([], [], color=C_PANCHOR, lw=2, label="catalog S off + P anchor track (C)")
    ax2.plot([], [], color=C_VALID, lw=2, label="validated S + P anchor track (D2)")
    ax2.plot([], [], color=INK, lw=1.5, ls="-", label="cluster 1 (solid) / cluster 2 (dashed)")
    sa, sl = frame.xy(frame.sta.x, frame.sta.y)
    ax2.scatter(sa, sl, marker="^", s=60, color=INK, zorder=4)
    for code, x, y in zip(frame.sta.code, sa, sl):
        ax2.annotate(code, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=MUTED)
    ax2.set_xlim(-4.7, -1.5); ax2.set_ylim(-2.0, 3.8); ax2.set_aspect("equal")
    ax2.set_xlabel("across margin (km)"); ax2.set_ylabel("along margin (km)")
    ax2.set_title("centroid tracks in map view", loc="left", fontsize=10, color=INK)
    ax2.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False,
               ncol=1)
    style(ax2)
    fig.suptitle("T2: whole clusters slide during hypoDD; the raw picks do not support the slide",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    out = os.path.join(SRC, "t2_reloc_cluster_drift.png")
    fig.savefig(out, dpi=110, facecolor=SURF)
    print(f"wrote {out}")


def main():
    frame = Frame()
    picks = Picks(frame)
    sel = read_event_sel(os.path.join(SRC, "input_files", "event.sel"))
    if "--drift-only" not in _sys.argv:
        fig_comparison(frame, picks, sel)
    fig_drift(frame, picks, sel)


if __name__ == "__main__":
    main()
