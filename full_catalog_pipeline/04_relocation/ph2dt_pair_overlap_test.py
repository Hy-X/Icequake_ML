#!/usr/bin/env python3
"""How much of the event-PAIR set would change if pyocto's locations changed?

This closes the one pathway `hypodd_initial_location_test.py` cannot reach. ph2dt selects
event pairs by their INITIAL locations (MAXSEP, MAXNGH), and that pair set is what gets
cross-correlated -- the ~10 h step. So a pyocto rerun can force a dt.cc rebuild even when
the picks are unchanged.

Because the picks ARE unchanged (verified ~99.9% identical at both arrays by
compare_velocity_model_catalogs.py), ph2dt can be rerun in seconds on the same phase.dat
with only the header locations swapped. The pair-set overlap that comes back is a direct
upper bound on how much of dt.cc would have to be recomputed:

    pairs in common / pairs in union  ->  the fraction of dt.cc that carries over.

Two arms are run so the MAXSEP effect is separable from the location effect:
  * `samesep`  -- production ph2dt.inp verbatim, only the locations change;
  * `newsep`   -- MAXSEP additionally recomputed from the new catalog the way
                  hypodd_relocate.compute_maxsep does.

Usage:
    python full_catalog_pipeline/04_relocation/ph2dt_pair_overlap_test.py \
        --base full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd_vels1d \
        --new-pyocto full_catalog_pipeline/artifacts/full_run/vmtest/T1_vels1d_d02 \
        --out full_catalog_pipeline/artifacts/full_run/vmtest/T1_ph2dt_pairs
"""
import argparse
import datetime
import math
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

PH2DT_BIN = "/home/jwalter/bin/ph2dt"


def read_phase_dat(path):
    """Return (header_lines_parsed, raw_blocks). Each block is (header_str, [pick_strs])."""
    blocks = []
    cur_h, cur_p = None, []
    for line in open(path):
        if line.startswith("#"):
            if cur_h is not None:
                blocks.append((cur_h, cur_p))
            cur_h, cur_p = line.rstrip("\n"), []
        elif line.strip():
            cur_p.append(line.rstrip("\n"))
    if cur_h is not None:
        blocks.append((cur_h, cur_p))
    return blocks


def header_time(h):
    f = h.split()
    sec = float(f[6])
    return (datetime.datetime(int(f[1]), int(f[2]), int(f[3]),
                              tzinfo=datetime.timezone.utc)
            + datetime.timedelta(hours=int(f[4]), minutes=int(f[5]), seconds=sec)).timestamp()


def write_phase_dat(blocks, new_loc, path):
    """new_loc: dict block_index -> (lat, lon, depth). Untouched blocks keep their header."""
    with open(path, "w") as out:
        for i, (h, picks) in enumerate(blocks):
            if i in new_loc:
                f = h.split()
                lat, lon, dep = new_loc[i]
                f[7], f[8], f[9] = f"{lat:.6f}", f"{lon:.6f}", f"{dep:.4f}"
                h = " ".join(f)
            out.write(h + "\n")
            for p in picks:
                out.write(p + "\n")


def compute_maxsep(lat, lon, depth_km, percentile=5.0, sample=800, seed=0):
    """5th-percentile pairwise 3-D inter-event distance, geodetically correct in longitude
    (the ~4.5x cos(lat) correction at ~77 S -- see hypodd_relocate.compute_maxsep)."""
    rng = np.random.default_rng(seed)
    n = min(sample, len(lat))
    idx = rng.choice(len(lat), n, replace=False)
    la, lo, dz = lat[idx], lon[idx], depth_km[idx]
    coslat = math.cos(math.radians(float(np.mean(la))))
    x = lo * 111.0 * coslat
    y = la * 111.0
    d = np.sqrt((x[:, None] - x) ** 2 + (y[:, None] - y) ** 2 + (dz[:, None] - dz) ** 2)
    return float(np.percentile(d[np.triu_indices(n, 1)], percentile))


def read_pairs(dt_ct_path):
    """Set of (id1, id2) pairs, and obs count per pair."""
    pairs = {}
    cur = None
    with open(dt_ct_path) as f:
        for line in f:
            if line.startswith("#"):
                a, b = line.split()[1:3]
                cur = (int(a), int(b))
                pairs[cur] = 0
            elif cur is not None and line.strip():
                pairs[cur] += 1
    return pairs


def run_ph2dt(work, maxsep=None):
    if maxsep is not None:
        lines = open(os.path.join(work, "ph2dt.inp")).read().rstrip("\n").split("\n")
        f = lines[-1].split()
        f[2] = f"{maxsep:.10f}"
        lines[-1] = " ".join(f)
        open(os.path.join(work, "ph2dt.inp"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(work, "ph2dt_log.txt"), "w") as log:
        rc = subprocess.call([PH2DT_BIN, "ph2dt.inp"], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    return rc


def report(label, base_pairs, new_pairs):
    kb, kn = set(base_pairs), set(new_pairs)
    common, union = kb & kn, kb | kn
    print(f"\n  {label}")
    print(f"    pairs: baseline {len(kb)}  new {len(kn)}  common {len(common)}  "
          f"({100*len(common)/max(len(union),1):.1f}% of union)")
    print(f"    => {100*len(common)/max(len(kb),1):.1f}% of the EXISTING dt.cc pair set survives; "
          f"{len(kn - kb)} pairs are new (would need fresh cross-correlation)")
    if common:
        do = np.array([new_pairs[k] - base_pairs[k] for k in common])
        print(f"    observations per common pair: median change {np.median(do):+.1f}, "
              f"unchanged for {100*np.mean(do == 0):.1f}% of them")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new-pyocto", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    src_in = os.path.join(args.base, "input_files")
    blocks = read_phase_dat(os.path.join(src_in, "phase.dat"))
    print(f"phase.dat: {len(blocks)} events, {sum(len(p) for _, p in blocks)} picks")

    new = pd.read_csv(os.path.join(args.new_pyocto, "pyocto_events.csv")).sort_values("time")
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
    print(f"matched {matched}/{len(blocks)} events to the new pyocto catalog")

    # Baseline arm: production phase.dat + production ph2dt.inp, rerun so both arms come
    # from the same binary and the comparison cannot be confounded by a stale dt.ct.
    for tag, locs, maxsep in [("baseline", {}, None),
                              ("samesep", new_loc, None),
                              ("newsep", new_loc, "recompute")]:
        work = os.path.join(args.out, tag)
        os.makedirs(work, exist_ok=True)
        for n in ["station.dat", "ph2dt.inp"]:
            shutil.copy(os.path.join(src_in, n), os.path.join(work, n))
        write_phase_dat(blocks, locs, os.path.join(work, "phase.dat"))
        ms = None
        if maxsep == "recompute":
            lat = np.array([new_loc[i][0] for i in sorted(new_loc)])
            lon = np.array([new_loc[i][1] for i in sorted(new_loc)])
            dep = np.array([new_loc[i][2] for i in sorted(new_loc)])
            ms = compute_maxsep(lat, lon, dep)
            print(f"recomputed MAXSEP from the new catalog: {ms:.4f} km")
        rc = run_ph2dt(work, ms)
        print(f"ph2dt[{tag}] rc={rc}")

    base_pairs = read_pairs(os.path.join(args.out, "baseline", "dt.ct"))
    print(f"\nBaseline ph2dt rerun: {len(base_pairs)} pairs "
          f"(production dt.ct has {len(read_pairs(os.path.join(src_in, 'dt.ct')))} -- "
          f"these should agree; a mismatch means the production file was built differently)")
    for tag in ["samesep", "newsep"]:
        report(tag, base_pairs, read_pairs(os.path.join(args.out, tag, "dt.ct")))


if __name__ == "__main__":
    main()
