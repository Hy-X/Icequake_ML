#!/usr/bin/env python3
"""Does hypoDD's answer depend on the INITIAL locations pyocto hands it?

Motivation: the pyocto association stage still uses the v5 two-layer velocity model
while hypoDD now uses the measured vels1d profile. Re-associating under vels1d would
change pyocto's locations; this script asks whether that would actually change the
relocated catalog, by rerunning hypoDD with its starting locations swapped and
EVERYTHING else (dt.ct, dt.cc, hypoDD.inp, station.sel) held fixed.

That isolates one of the three pathways by which a pyocto rerun could propagate
downstream. The other two -- changed pick assignments (which would change phase.dat and
therefore dt.ct) and changed event pairing via ph2dt's MAXSEP (which would change which
pairs are cross-correlated) -- are NOT covered here and have to be argued separately
from the association-stage comparison.

Events are matched from event.sel to the new pyocto catalog by origin time. Events with
no match keep their original starting location, and the count is reported.

Usage:
    python full_catalog_pipeline/04_relocation/hypodd_initial_location_test.py \
        --base full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd_vels1d \
        --new-pyocto full_catalog_pipeline/artifacts/full_run/vmtest/T1_vels1d_d02 \
        --out full_catalog_pipeline/artifacts/full_run/vmtest/T1_hypodd_initloc
"""
import argparse
import datetime
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

HYPODD_BIN = "/home/jwalter/bin/hypoDD"


def parse_event_sel(path):
    """hypoDD event.sel -> DataFrame. Fixed format: DATE TIME LAT LON DEPTH MAG EH EZ RMS ID."""
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        f = line.split()
        date, tt = f[0], f[1]
        # TIME is HHMMSSss, written without leading zeros. The fields are added as a
        # timedelta rather than passed to the datetime constructor because hypoDD rounds
        # seconds and can emit SS=60 (two such events in T2's event.sel), which the
        # constructor rejects but which simply means the next minute.
        tt = tt.zfill(8)
        dt = (datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:8]),
                                tzinfo=datetime.timezone.utc)
              + datetime.timedelta(hours=int(tt[0:2]), minutes=int(tt[2:4]),
                                   seconds=int(tt[4:6]), milliseconds=int(tt[6:8]) * 10))
        rows.append({"raw": line.rstrip("\n"), "unix": dt.timestamp(),
                     "lat": float(f[2]), "lon": float(f[3]), "depth": float(f[4]),
                     "id": int(f[-1])})
    return pd.DataFrame(rows)


def rewrite_event_sel(df, out_path):
    """Write event.sel back, keeping every field except lat/lon/depth verbatim.

    The original columns are rewritten in place by slicing the source line, so the
    date/time/magnitude/error/ID fields cannot drift -- only the three numbers we mean
    to change are touched.
    """
    with open(out_path, "w") as f:
        for _, r in df.iterrows():
            g = r["raw"].split()
            g[2], g[3], g[4] = f"{r['new_lat']:.4f}", f"{r['new_lon']:.4f}", f"{r['new_depth']:.3f}"
            f.write(f"{g[0]:>8} {g[1]:>9} {g[2]:>9} {g[3]:>10} {g[4]:>10} "
                    f"{g[5]:>6} {g[6]:>7} {g[7]:>7} {g[8]:>6} {g[9]:>10}\n")


def read_reloc(path):
    cols = ["id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
            "yr", "mo", "dy", "hr", "mi", "sc", "mag", "nccp", "nccs", "nctp",
            "nvts", "rcc", "rct", "cid"]
    df = pd.read_csv(path, sep=r"\s+", names=cols)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="existing hypoDD run dir (has input_files/, output_files/)")
    ap.add_argument("--new-pyocto", required=True, help="dir with the new pyocto_events.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtcc", default="dt.cc",
                     help="which cross-correlation file in input_files to use. It is copied under "
                          "its own basename because hypoDD.inp names it explicitly (T1: dt.cc, "
                          "T2: dt.cc.authoritative)")
    ap.add_argument("--baseline-reloc", default=None,
                     help="hypoDD.reloc to compare against; default is <base>/output_files/hypoDD.reloc. "
                          "Point this at another run of THIS script to A/B two starting catalogs through "
                          "identical machinery")
    ap.add_argument("--tol", type=float, default=0.5, help="origin-time match tolerance (s)")
    args = ap.parse_args()

    src_in = os.path.join(args.base, "input_files")
    out_in = os.path.join(args.out, "input_files")
    os.makedirs(out_in, exist_ok=True)
    for name in ["dt.ct", "station.sel", "hypoDD.inp"]:
        shutil.copy(os.path.join(src_in, name), os.path.join(out_in, name))
    # copy the CC file under its OWN basename -- hypoDD.inp names it explicitly, and at T2
    # that name is "dt.cc.authoritative", not "dt.cc".
    shutil.copy(os.path.join(src_in, args.dtcc), os.path.join(out_in, os.path.basename(args.dtcc)))

    ev = parse_event_sel(os.path.join(src_in, "event.sel"))
    new = pd.read_csv(os.path.join(args.new_pyocto, "pyocto_events.csv")).sort_values("time")
    nt = new["time"].to_numpy()

    j = np.searchsorted(nt, ev["unix"].to_numpy())
    best = np.full(len(ev), -1)
    bestd = np.full(len(ev), np.inf)
    for shift in (-1, 0):
        k = np.clip(j + shift, 0, len(nt) - 1)
        d = np.abs(nt[k] - ev["unix"].to_numpy())
        take = (d < args.tol) & (d < bestd)
        best[take], bestd[take] = k[take], d[take]
    ok = best >= 0
    ev["new_lat"] = np.where(ok, new["latitude"].to_numpy()[np.clip(best, 0, len(nt)-1)], ev["lat"])
    ev["new_lon"] = np.where(ok, new["longitude"].to_numpy()[np.clip(best, 0, len(nt)-1)], ev["lon"])
    ev["new_depth"] = np.where(ok, new["depth"].to_numpy()[np.clip(best, 0, len(nt)-1)], ev["depth"])
    print(f"event.sel: {len(ev)} events, {ok.sum()} matched to new pyocto catalog "
          f"({len(ev)-ok.sum()} kept at their original start location)")
    dz = (ev["new_depth"] - ev["depth"])[ok] * 1000
    print(f"start-depth change: median {np.median(dz):+.1f} m, "
          f"median|.| {np.median(np.abs(dz)):.1f} m, p5/p95 {np.percentile(dz,5):+.1f}/{np.percentile(dz,95):+.1f} m")

    rewrite_event_sel(ev, os.path.join(out_in, "event.sel"))

    print(f"running hypoDD in {out_in} ...")
    with open(os.path.join(args.out, "hypoDD_log.txt"), "w") as log:
        rc = subprocess.call([HYPODD_BIN, "hypoDD.inp"], cwd=out_in, stdout=log, stderr=subprocess.STDOUT)
    print(f"hypoDD rc={rc}")

    out_files = os.path.join(args.out, "output_files")
    os.makedirs(out_files, exist_ok=True)
    for name in ["hypoDD.loc", "hypoDD.reloc", "hypoDD.res", "hypoDD.src", "hypoDD.sta"]:
        p = os.path.join(out_in, name)
        if os.path.exists(p):
            shutil.move(p, os.path.join(out_files, name))

    base_rel = read_reloc(args.baseline_reloc or
                          os.path.join(args.base, "output_files", "hypoDD.reloc"))
    new_rel = read_reloc(os.path.join(out_files, "hypoDD.reloc"))
    print(f"\nrelocated: baseline {len(base_rel)}  ->  new-start {len(new_rel)}")
    m = base_rel.merge(new_rel, on="id", suffixes=("_b", "_n"))
    print(f"common relocated events: {len(m)}")
    dz = (m["depth_n"] - m["depth_b"]) * 1000
    dx = (m["x_n"] - m["x_b"])
    dy = (m["y_n"] - m["y_b"])
    dh = np.hypot(dx, dy)
    print(f"\nFINAL hypoDD location change (new start minus baseline):")
    print(f"  epicentral: median {np.median(dh):.1f} m   p95 {np.percentile(dh,95):.1f} m")
    print(f"  depth:      median {np.median(dz):+.1f} m   median|.| {np.median(np.abs(dz)):.1f} m"
          f"   p5/p95 {np.percentile(dz,5):+.1f}/{np.percentile(dz,95):+.1f} m")
    for lab, d in [("baseline", base_rel["depth"]), ("new-start", new_rel["depth"])]:
        print(f"  depth dist {lab:<10} p5={np.percentile(d,5):.3f} med={np.median(d):.3f} "
              f"p95={np.percentile(d,95):.3f} mean={d.mean():.3f} km")
    for lab, r in [("baseline", base_rel), ("new-start", new_rel)]:
        print(f"  residuals {lab:<10} rct median={np.median(r['rct']):.4f}  rcc median={np.median(r['rcc']):.4f}")


if __name__ == "__main__":
    main()
