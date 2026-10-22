#!/usr/bin/env python3
"""Undo hypoDD's whole-cluster translation by re-anchoring each cluster to its starting centroid.

Double-difference data constrain where events sit RELATIVE to each other, not where a tight
cluster sits as a whole: the common translation is a near-null-space mode, and in LSQR it is held
only by damping. On T2 v6 it drifted steadily as wt_ct and damping were tapered, sliding cluster 1
(n=2017) 1.2 km and taking the median per-event absolute P residual from 22 ms to 103 ms.

This shifts every hypoDD cluster rigidly so that its median epicentre equals the median of the
SAME events' starting (event.sel) epicentres. Depths are left alone by default (--mode 3d also
re-anchors depth). Internal geometry is untouched, so the relative locations hypoDD is good at
survive while the absolute position comes back to what the raw picks support.

Writes <array>_v5/<out>/output_files/hypoDD.reloc, with input_files symlinked to the source run
so catalog_paths and every downstream script can use it through ICEQUAKE_RELOC=<out>.

Usage (from the repository root):
    python full_catalog_pipeline/04_relocation/reanchor_clusters.py --array T2 \\
        --src full_catalog_pipeline/artifacts/full_run/T2_v5/hypodd_v6_allfix/output_files/hypoDD.reloc \\
        --source-run hypodd_v6_allfix --out hypodd_v6_reanchored
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

import numpy as np
import pandas as pd
import pyproj

import catalog_paths
from hypodd_tune import RELOC_COLS

TO_PS = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
FROM_PS = pyproj.Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)


def read_event_sel(path):
    sel = pd.read_csv(path, sep=r"\s+", header=None)
    return sel.rename(columns={2: "lat", 3: "lon", 4: "depth", 9: "id"}).set_index("id")


def reanchor(reloc, sel, mode="h"):
    e = reloc.copy()
    x, y = TO_PS.transform(e.lon.values, e.lat.values)
    s = sel.loc[e.id]
    x0, y0 = TO_PS.transform(s.lon.values, s.lat.values)
    e["X"], e["Y"], e["X0"], e["Y0"], e["Z0"] = x, y, x0, y0, s.depth.values
    shifts = []
    for c, m in e.groupby("cid"):
        dx, dy = np.median(m.X0) - np.median(m.X), np.median(m.Y0) - np.median(m.Y)
        dz = np.median(m.Z0) - np.median(m.depth) if mode == "3d" else 0.0
        e.loc[m.index, "X"] += dx
        e.loc[m.index, "Y"] += dy
        e.loc[m.index, "depth"] += dz
        shifts.append(dict(cid=c, n=len(m), dx_m=dx, dy_m=dy, dz_km=dz))
    lon, lat = FROM_PS.transform(e.X.values, e.Y.values)
    e["lon"], e["lat"] = lon, lat
    return e[RELOC_COLS], pd.DataFrame(shifts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", required=True, choices=["T1", "T2"])
    ap.add_argument("--src", required=True, help="hypoDD.reloc to re-anchor")
    ap.add_argument("--source-run", required=True,
                    help="run dir (under <array>_v5/) whose input_files/event.sel holds the start")
    ap.add_argument("--out", required=True, help="new run dir name under <array>_v5/")
    ap.add_argument("--mode", choices=["h", "3d"], default="h")
    args = ap.parse_args()

    src_inputs = os.path.join(catalog_paths.run_dir(args.array, args.source_run), "input_files")
    sel = read_event_sel(os.path.join(src_inputs, "event.sel"))
    reloc = pd.read_csv(args.src, sep=r"\s+", header=None, names=RELOC_COLS)
    out, shifts = reanchor(reloc, sel, args.mode)

    run = catalog_paths.run_dir(args.array, args.out)
    os.makedirs(os.path.join(run, "output_files"), exist_ok=True)
    link = os.path.join(run, "input_files")
    if not os.path.lexists(link):
        os.symlink(os.path.relpath(src_inputs, run), link)
    out.to_csv(os.path.join(run, "output_files", "hypoDD.reloc"), sep=" ", header=False,
               index=False, float_format="%.6f")
    shifts.to_csv(os.path.join(run, "reanchor_shifts.csv"), index=False)
    with open(os.path.join(run, "PROVENANCE.md"), "w") as f:
        f.write(f"# {args.array} {args.out} -- CANDIDATE, not authoritative\n\n"
                f"Re-anchored copy of `{args.src}` by `04_relocation/reanchor_clusters.py "
                f"--mode {args.mode}`: each hypoDD cluster shifted rigidly so its median epicentre "
                f"equals the median of the same events' `{args.source_run}/input_files/event.sel` "
                f"epicentres. Per-cluster shifts: `reanchor_shifts.csv`.\n")
    big = shifts.sort_values("n", ascending=False).head(4)
    for r in big.itertuples():
        print(f"cluster {r.cid:3d} n={r.n:4d}  shift {np.hypot(r.dx_m, r.dy_m):6.0f} m")
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
