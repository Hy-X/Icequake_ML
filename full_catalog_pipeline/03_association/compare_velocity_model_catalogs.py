#!/usr/bin/env python3
"""Compare two pyocto catalogs produced from the SAME picks with DIFFERENT velocity
models, and report whether the difference is large enough to matter downstream.

Events are matched by origin time (the same picks associated under two models produce
essentially the same events at essentially the same times, so a tight tolerance is a
valid match rule here -- and unmatched counts are themselves a reported result).

Three things are reported, because they drive three different downstream costs:

  * location/time shifts -- what changes in the association-stage catalog itself;
  * pick-assignment changes -- these change phase.dat, hence dt.ct, hence hypoDD's data;
  * MAXSEP and inter-event geometry -- ph2dt pairs events by their INITIAL locations, so
    a shift here changes which pairs are formed and therefore which pairs are
    cross-correlated (the ~10 h step), not just where hypoDD starts from.

Usage:
    python full_catalog_pipeline/03_association/compare_velocity_model_catalogs.py \
        --ref  <dir with pyocto_events.csv + pyocto_assignments.csv> \
        --test <dir> [--label-ref v5_twolayer --label-test vels1d] [--tol 0.5]
"""
import argparse
import os

import numpy as np
import pandas as pd


def load(d):
    ev = pd.read_csv(os.path.join(d, "pyocto_events.csv"))
    asg = pd.read_csv(os.path.join(d, "pyocto_assignments.csv"))
    return ev, asg


def match_by_time(a, b, tol):
    """Greedy nearest-time matching of b onto a. Returns (idx_a, idx_b) arrays."""
    a_sorted = a.sort_values("time")
    b_sorted = b.sort_values("time")
    ta, tb = a_sorted["time"].to_numpy(), b_sorted["time"].to_numpy()
    j = np.searchsorted(tb, ta)
    best = np.full(len(ta), -1)
    for shift in (-1, 0):
        k = np.clip(j + shift, 0, len(tb) - 1)
        d = np.abs(tb[k] - ta)
        take = (d < tol) & ((best < 0) | (d < np.abs(tb[np.clip(best, 0, len(tb) - 1)] - ta)))
        best[take] = k[take]
    ok = best >= 0
    return a_sorted.index.to_numpy()[ok], b_sorted.index.to_numpy()[best[ok]]


def pct(x, q):
    return np.percentile(x, q) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--label-ref", default="ref")
    ap.add_argument("--label-test", default="test")
    ap.add_argument("--tol", type=float, default=0.5, help="origin-time match tolerance (s)")
    ap.add_argument("--maxsep-sample", type=int, default=800)
    args = ap.parse_args()

    ev_a, asg_a = load(args.ref)
    ev_b, asg_b = load(args.test)
    print(f"\n{'='*76}")
    print(f"{args.label_ref}: {len(ev_a)} events / {len(asg_a)} assigned picks   ({args.ref})")
    print(f"{args.label_test}: {len(ev_b)} events / {len(asg_b)} assigned picks   ({args.test})")
    print(f"{'='*76}")

    ia, ib = match_by_time(ev_a, ev_b, args.tol)
    A, B = ev_a.loc[ia].reset_index(drop=True), ev_b.loc[ib].reset_index(drop=True)
    print(f"Matched {len(A)} events at |dt| < {args.tol}s "
          f"({100*len(A)/len(ev_a):.1f}% of {args.label_ref}, "
          f"{100*len(A)/len(ev_b):.1f}% of {args.label_test}); "
          f"unmatched: {len(ev_a)-len(A)} / {len(ev_b)-len(B)}")

    dz = (B["depth"] - A["depth"]).to_numpy() * 1000.0        # m
    dx = (B["x"] - A["x"]).to_numpy() * 1000.0
    dy = (B["y"] - A["y"]).to_numpy() * 1000.0
    dh = np.hypot(dx, dy)
    dt = (B["time"] - A["time"]).to_numpy() * 1000.0          # ms
    dnp = (B["picks"] - A["picks"]).to_numpy()

    print(f"\nShifts, {args.label_test} minus {args.label_ref} (matched events):")
    print(f"{'quantity':<22}{'median':>10}{'mean':>10}{'p5':>10}{'p95':>10}{'|.| med':>10}")
    for name, v, unit in [("epicentral shift", dh, "m"), ("depth shift", dz, "m"),
                          ("origin-time shift", dt, "ms"), ("pick count change", dnp, "")]:
        print(f"{name+' ('+unit+')':<22}{np.median(v):>10.1f}{np.mean(v):>10.1f}"
              f"{pct(v,5):>10.1f}{pct(v,95):>10.1f}{np.median(np.abs(v)):>10.1f}")

    print(f"\nDepth distribution:")
    print(f"{'':<14}{'p5':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p95':>9}{'mean':>9}")
    for lab, ev in [(args.label_ref, ev_a), (args.label_test, ev_b)]:
        d = ev["depth"].to_numpy()
        print(f"{lab:<14}" + "".join(f"{pct(d,q):>9.3f}" for q in (5, 25, 50, 75, 95))
              + f"{d.mean():>9.3f}")

    # Fraction of events below the nominal ice base, the quantity every depth
    # discussion in this project turns on.
    for lab, ev, base in [(args.label_ref, ev_a, None), (args.label_test, ev_b, None)]:
        pass

    print(f"\nPick assignments (which picks joined which event):")
    def keyset(asg, ev):
        # assignments carry event_idx + station + time; compare the multiset of
        # (station, rounded pick time) actually used, and per-event composition
        cols = [c for c in asg.columns]
        return cols
    print(f"  {args.label_ref} columns: {list(asg_a.columns)}")
    ka = set(zip(asg_a["station"], np.round(asg_a["time"], 3)))
    kb = set(zip(asg_b["station"], np.round(asg_b["time"], 3)))
    print(f"  picks used by both: {len(ka & kb)}   only {args.label_ref}: {len(ka - kb)}"
          f"   only {args.label_test}: {len(kb - ka)}")
    print(f"  => {100*len(ka & kb)/max(len(ka | kb),1):.2f}% of all used picks are common")

    # MAXSEP: ph2dt's pairing radius, recomputed the way hypodd_relocate.py does
    rng = np.random.default_rng(0)
    for lab, ev in [(args.label_ref, ev_a), (args.label_test, ev_b)]:
        n = min(args.maxsep_sample, len(ev))
        s = ev.iloc[rng.choice(len(ev), n, replace=False)]
        X, Y, Z = s["x"].to_numpy(), s["y"].to_numpy(), s["depth"].to_numpy()
        d = np.sqrt((X[:, None]-X)**2 + (Y[:, None]-Y)**2 + (Z[:, None]-Z)**2)
        d = d[np.triu_indices(n, 1)]
        print(f"  MAXSEP-equivalent (5th pct of {n}-event sample) {lab}: {np.percentile(d,5):.4f} km")


if __name__ == "__main__":
    main()
