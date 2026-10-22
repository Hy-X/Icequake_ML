#!/usr/bin/env python3
"""Rebuild the hypoDD relocation from a pyocto association done under the vels1d model.

Background: the association stage located events with the v5 two-layer model while the
relocation stage used the measured vels1d profile. This rebuilds the chain so both stages
use the same model, per array.

Two arms are produced, because a rebuild necessarily pulls in two fixes that were staged
but never promoted, and bundling them would make the v5->v6 difference unattributable:

  vmonly  -- the ONLY change is the association velocity model. MAXSEP stays at each
             array's production value and dt.cc is regenerated WITHOUT upsampling, so the
             cross-correlation keeps production semantics (including its P deficit).
  upsonly -- vmonly PLUS the x2 upsampled dt.cc, MAXSEP still at the production value.
             Differencing this against vmonly isolates the upsampling exactly, since
             locations, pair set and every other setting are identical.
  allfix  -- upsonly PLUS the corrected geodetic MAXSEP. Differencing against upsonly
             isolates the MAXSEP change.

So vmonly -> upsonly -> allfix adds exactly one change at a time.

What is deliberately NOT rebuilt: phase.dat's PICKS. The vels1d association assigns
99.9% of the same picks to the same events, so production's phase.dat is reused with only
the header locations swapped. That keeps the event-id map stable and makes the arms differ
by exactly the intended quantities rather than by a handful of pick reassignments.

Network code: both arrays' phase.dat/station.sel use 7U. regenerate_dtcc.py defaults T1 to
2E, which would silently produce a dt.cc whose station names match nothing -- always pass
--network explicitly here.

Usage:
    python full_catalog_pipeline/04_relocation/rebuild_v6_from_vels1d_association.py \
        --array T1 --arm allfix
"""
import os as _os, sys as _sys
_PIPE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_PIPE] + [_os.path.join(_PIPE, _s) for _s in sorted(_os.listdir(_PIPE))
                     if _os.path.isdir(_os.path.join(_PIPE, _s))
                     and not _s.startswith(('.', 'artifacts', '__'))]:
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import argparse
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from ph2dt_pair_overlap_test import (
    read_phase_dat, header_time, write_phase_dat, compute_maxsep, run_ph2dt, read_pairs,
)
from hypodd_tune import filter_dtcc

ROOT = "full_catalog_pipeline/artifacts/full_run"
VM = f"{ROOT}/vmtest"
HYPODD_BIN = "/home/jwalter/bin/hypoDD"
NETWORK = "7U"                     # both arrays -- see module docstring

# Each array's production dt.cc convention, kept identical in BOTH arms so the arms differ
# only by MAXSEP and upsampling. T2's authoritative run filters dt.cc; T1's does not.
DTCC_FILTER = {"T1": None, "T2": {"max_dt_s": 0.2, "min_w": 0.5}}


def production_maxsep(src_in):
    """MAXSEP as the production ph2dt.inp actually used it (field 3 of the parameter line).

    The parameter line is the last NON-EMPTY line; production's file has no trailing
    newline, so indexing a plain split("\n") off the end lands on the phase.dat filename.
    """
    text = open(os.path.join(src_in, "ph2dt.inp")).read()
    last = [ln for ln in text.split("\n") if ln.strip()][-1]
    return float(last.split()[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", required=True, choices=["T1", "T2"])
    ap.add_argument("--arm", required=True, choices=["vmonly", "upsonly", "allfix"])
    ap.add_argument("--workers", type=int, default=8,
                    help="cross-correlation workers. Kept well below nproc by default: this "
                         "machine also runs live seismic monitoring")
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    A, arm = args.array, args.arm
    src_in = f"{ROOT}/{A}_v5/hypodd_vels1d/input_files"
    which = f"hypodd_v6_{arm}"
    out = f"{ROOT}/{A}_v5/{which}"
    out_in = os.path.join(out, "input_files")
    os.makedirs(out_in, exist_ok=True)
    log = lambda m: print(f"[{A}/{arm}] {m}", flush=True)

    # ---- 1. phase.dat with the vels1d-association locations ------------------------
    blocks = read_phase_dat(os.path.join(src_in, "phase.dat"))
    new = pd.read_csv(f"{VM}/{A}_vels1d_d02/pyocto_events.csv").sort_values("time")
    nt = new["time"].to_numpy()
    times = np.array([header_time(h) for h, _ in blocks])
    j = np.searchsorted(nt, times)
    new_loc, matched = {}, 0
    for i, t in enumerate(times):
        cands = [k for k in (j[i] - 1, j[i]) if 0 <= k < len(nt)]
        if not cands:
            continue
        k = min(cands, key=lambda k: abs(nt[k] - t))
        if abs(nt[k] - t) < args.tol:
            r = new.iloc[k]
            new_loc[i] = (r["latitude"], r["longitude"], r["depth"])
            matched += 1
    log(f"phase.dat {len(blocks)} events, {matched} relocated to the vels1d association "
        f"({len(blocks)-matched} keep their v5 location)")
    write_phase_dat(blocks, new_loc, os.path.join(out_in, "phase.dat"))
    for n in ["station.dat", "station.sel", "ph2dt.inp", "hypoDD.inp"]:
        shutil.copy(os.path.join(src_in, n), os.path.join(out_in, n))

    # ---- 2. ph2dt -> dt.ct, event.sel ----------------------------------------------
    prod_ms = production_maxsep(src_in)
    if arm in ("vmonly", "upsonly"):
        ms = prod_ms
        log(f"MAXSEP held at the production value {ms:.4f} km")
    else:
        lat = np.array([new_loc[i][0] for i in sorted(new_loc)])
        lon = np.array([new_loc[i][1] for i in sorted(new_loc)])
        dep = np.array([new_loc[i][2] for i in sorted(new_loc)])
        ms = compute_maxsep(lat, lon, dep)
        log(f"MAXSEP recomputed geodetically: {prod_ms:.4f} -> {ms:.4f} km")
    t0 = time.time()
    rc = run_ph2dt(out_in, ms)
    if rc != 0:
        raise SystemExit(f"ph2dt failed rc={rc}")
    npairs = len(read_pairs(os.path.join(out_in, "dt.ct")))
    log(f"ph2dt ok in {time.time()-t0:.0f}s -- {npairs:,} event pairs")

    # ---- 3. dt.cc ------------------------------------------------------------------
    upsample = 1 if arm == "vmonly" else 2
    raw = os.path.join(out_in, "dt.cc.regen")   # regenerate_dtcc refuses to write "dt.cc"
    env = {**os.environ, "ICEQUAKE_RELOC": which}
    t0 = time.time()
    cmd = ["python", "full_catalog_pipeline/05_crosscorrelation/regenerate_dtcc.py",
           "--array", A, "--network", NETWORK, "--upsample", str(upsample),
           "--workers", str(args.workers), "--out", raw]
    log(f"regenerating dt.cc (upsample x{upsample}, network {NETWORK}) ...")
    with open(os.path.join(out, "regen_dtcc.log"), "w") as f:
        rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"regenerate_dtcc failed rc={rc}; see {out}/regen_dtcc.log")
    log(f"dt.cc regenerated in {(time.time()-t0)/60:.1f} min")

    filt = DTCC_FILTER[A]
    if filt:
        use = os.path.join(out_in, "dt.cc.filtered")
        n_in, n_out = filter_dtcc(raw, use, **filt)
        log(f"dt.cc filtered (|dt|<={filt['max_dt_s']}s, w>={filt['min_w']}): "
            f"{n_out:,}/{n_in:,} observations kept ({100*n_out/max(n_in,1):.1f}%)")
    else:
        use = raw
        log("dt.cc left unfiltered (matches this array's production convention)")

    # ---- 4. hypoDD -----------------------------------------------------------------
    inp = os.path.join(out_in, "hypoDD.inp")
    lines = open(inp).read().split("\n")
    lines[1] = os.path.basename(use)
    open(inp, "w").write("\n".join(lines))
    log(f"hypoDD.inp now reads {lines[1]}")
    t0 = time.time()
    with open(os.path.join(out, "hypoDD_log.txt"), "w") as f:
        rc = subprocess.call([HYPODD_BIN, "hypoDD.inp"], cwd=out_in,
                             stdout=f, stderr=subprocess.STDOUT)
    out_files = os.path.join(out, "output_files")
    os.makedirs(out_files, exist_ok=True)
    for n in ["hypoDD.loc", "hypoDD.reloc", "hypoDD.res", "hypoDD.src", "hypoDD.sta"]:
        p = os.path.join(out_in, n)
        if os.path.exists(p):
            shutil.move(p, os.path.join(out_files, n))
    reloc = os.path.join(out_files, "hypoDD.reloc")
    n = sum(1 for _ in open(reloc)) if os.path.exists(reloc) else 0
    log(f"hypoDD rc={rc} in {(time.time()-t0)/60:.1f} min -- {n:,} events relocated")
    if n == 0:
        log("WARNING: zero events relocated -- inversion did not converge for this arm")


if __name__ == "__main__":
    main()
