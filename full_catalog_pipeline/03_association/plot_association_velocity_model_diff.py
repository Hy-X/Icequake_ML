#!/usr/bin/env python3
"""The pyocto ASSOCIATION under two velocity models, compared directly.

Nothing downstream is involved -- no ph2dt, no hypoDD. The question is only: locate the
same picks with the two-layer model the association stage currently uses, versus the
measured vels1d reflection profile the relocation stage uses, and show how the resulting
event catalogues differ.

Both runs use the SAME picks and the SAME eikonal grid spacing (20 m), so every difference
shown is the velocity model and nothing else. (A separate control -- the two-layer model
rerun at 20 m against the 100 m production grid -- established that grid quantisation alone
moves epicentres ~30 m and depths ~96 m; that is the floor these differences sit on.)

Layout, one row per array:
    map view  |  depth section  |  depth distribution  |  per-event shift

Usage (from the repository root):
    python full_catalog_pipeline/03_association/plot_association_velocity_model_diff.py
"""
import os as _os, sys as _sys
_PIPE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_PIPE] + [_os.path.join(_PIPE, _s) for _s in sorted(_os.listdir(_PIPE))
                     if _os.path.isdir(_os.path.join(_PIPE, _s))
                     and not _s.startswith(('.', 'artifacts', '__'))]:
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vels1d_model as V

VM = "full_catalog_pipeline/artifacts/full_run/vmtest"
FIGS = "full_catalog_pipeline/figures"
STATIONS_JSON = "/scratch2/qm/t1/output/working_files/stations.json"
ARRAY_STATIONS = {
    "T1": set("DEEJ ELZA LILA LOUS OTIS SQIG TJTJ".split()),
    "T2": set("BAUM DRSC EPJZ FRST JULA OKGS WICH".split()),
}

INK, MUTED, GRID = "#0b0b0b", "#55534e", "#e4e3de"
C_TWO = "#2a78d6"    # two-layer -- the model association currently uses
C_VELS = "#eb6834"   # vels1d measured profile
C_REF = "#8a8a86"    # neutral: stations, reference bands, connector segments

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})


def load(d):
    return pd.read_csv(f"{VM}/{d}/pyocto_events.csv")


def match(a, b, tol=0.5):
    """Pair events between the two catalogues by origin time. They come from identical
    picks, so origin times agree to milliseconds and a 0.5 s window is unambiguous."""
    a = a.sort_values("time").reset_index(drop=True)
    b = b.sort_values("time").reset_index(drop=True)
    tb = b["time"].to_numpy()
    j = np.searchsorted(tb, a["time"].to_numpy())
    best = np.full(len(a), -1); bd = np.full(len(a), np.inf)
    for sh in (-1, 0):
        k = np.clip(j + sh, 0, len(tb) - 1)
        d = np.abs(tb[k] - a["time"].to_numpy())
        take = (d < tol) & (d < bd)
        best[take], bd[take] = k[take], d[take]
    ok = best >= 0
    return a[ok].reset_index(drop=True), b.iloc[best[ok]].reset_index(drop=True)


def station_latlon(array):
    st = json.load(open(STATIONS_JSON))
    lat, lon = [], []
    for k, v in st.items():
        if k.split(".")[1] in ARRAY_STATIONS[array]:
            lat.append(v["latitude"]); lon.append(v["longitude"])
    return np.array(lat), np.array(lon)


def to_local_km(lat, lon, ref_lat, ref_lon):
    """Flat-earth projection onto km east/north about a single reference point.

    Everything plotted -- both catalogues and the stations -- goes through this one
    function with the SAME reference, so the frames cannot drift apart. (pyocto's own x/y
    columns are a different projection with their own origin; mixing them with a
    locally-computed station projection is what put the stations in the wrong place.)
    """
    return ((lon - ref_lon) * 111.32 * np.cos(np.radians(ref_lat)),
            (lat - ref_lat) * 111.32)


def main():
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.6))
    for r, array in enumerate(["T1", "T2"]):
        two = load(f"{array}_twolayer_d02")
        vel = load(f"{array}_vels1d_d02")
        A, B = match(two, vel)
        marks = V.bed_markers(array)
        dh = np.hypot(B.x.to_numpy() - A.x.to_numpy(), B.y.to_numpy() - A.y.to_numpy()) * 1000
        dz = (B.depth - A.depth) * 1000

        slat, slon = station_latlon(array)
        ref_lat, ref_lon = float(slat.mean()), float(slon.mean())
        sx, sy = to_local_km(slat, slon, ref_lat, ref_lon)
        ax_, ay_ = to_local_km(A.latitude.to_numpy(), A.longitude.to_numpy(), ref_lat, ref_lon)
        bx_, by_ = to_local_km(B.latitude.to_numpy(), B.longitude.to_numpy(), ref_lat, ref_lon)

        # ---------------- map view
        ax = axes[r, 0]
        ax.scatter(ax_, ay_, s=5, color=C_TWO, alpha=0.40, lw=0,
                   label=f"two-layer  (n={len(two)})")
        ax.scatter(bx_, by_, s=5, color=C_VELS, alpha=0.40, lw=0,
                   label=f"vels1d  (n={len(vel)})")
        ax.scatter(sx, sy, marker="v", s=70, color=INK, zorder=6,
                   edgecolor="white", lw=0.8, label="stations")
        ax.set_aspect("equal")
        ax.set_xlabel("East of array centre (km)")
        ax.set_ylabel(f"{array}\n\nNorth of array centre (km)")
        ax.set_title(f"{array} — map view", fontsize=11)
        ax.legend(fontsize=8, loc="lower left", markerscale=2.2, framealpha=0.95)

        # ---------------- depth section
        ax = axes[r, 1]
        ax.scatter(ax_, A.depth, s=5, color=C_TWO, alpha=0.35, lw=0, label="two-layer")
        ax.scatter(bx_, B.depth, s=5, color=C_VELS, alpha=0.35, lw=0, label="vels1d")
        ax.axhspan(marks["ice_base"], marks["bedrock_top"], color=C_REF, alpha=0.20, lw=0)
        ax.annotate("ice–bed ramp", xy=(0.02, marks["ice_base"]), xycoords=("axes fraction", "data"),
                    xytext=(0, -4), textcoords="offset points", fontsize=8, color=MUTED, va="bottom")
        ax.set_ylim(5.2, -0.2)
        ax.set_xlabel("East of array centre (km)"); ax.set_ylabel("Depth (km)")
        ax.set_title(f"{array} — depth section", fontsize=11)
        ax.legend(fontsize=8, loc="lower right", markerscale=2.2, framealpha=0.95)

        # ---------------- depth distribution
        ax = axes[r, 2]
        bins = np.arange(0, 5.0, 0.08)
        ax.hist(A.depth, bins=bins, histtype="step", lw=2, color=C_TWO, label="two-layer")
        ax.hist(B.depth, bins=bins, histtype="step", lw=2, color=C_VELS, label="vels1d")
        ax.axvspan(marks["ice_base"], marks["bedrock_top"], color=C_REF, alpha=0.20, lw=0)
        for df, col in [(A, C_TWO), (B, C_VELS)]:
            ax.axvline(df.depth.median(), color=col, lw=1.4, ls="--")
        ax.set_xlabel("Depth (km)"); ax.set_ylabel("events")
        ax.set_title(f"{array} — depth distribution\n"
                     f"median {A.depth.median():.2f} → {B.depth.median():.2f} km", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper right")

        # ---------------- per-event shift
        ax = axes[r, 3]
        ax.scatter(dh, dz, s=5, color=C_VELS, alpha=0.30, lw=0)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xlim(0, 1400); ax.set_ylim(-1500, 900)
        ax.set_xlabel("Epicentral shift (m)"); ax.set_ylabel("Depth shift, vels1d − two-layer (m)")
        ax.set_title(f"{array} — per-event change", fontsize=11)
        txt = (f"epicentre  median {np.median(dh):5.0f} m   p95 {np.percentile(dh,95):5.0f} m\n"
               f"depth      median {np.median(dz):+5.0f} m   |Δ| med {np.median(np.abs(dz)):4.0f} m\n"
               f"grid-noise floor:  {30 if array=='T2' else 34} m / {96} m")
        ax.annotate(txt, xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8.5, family="monospace", color=MUTED,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))

    fig.suptitle("pyocto association under two velocity models — same picks, same 20 m grid, "
                 "nothing downstream\n"
                 "T1 barely moves; T2 shifts systematically shallower and its epicentres move "
                 "6× the grid-noise floor",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    out = f"{FIGS}/03_assoc_velocity_model_diff.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
