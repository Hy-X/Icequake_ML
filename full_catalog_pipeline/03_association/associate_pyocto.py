#!/usr/bin/env python3
"""Associate a combined picks CSV (station,time,phase,probability) into events
using pyocto, for the T1/T2 glacier array. Station coordinates come from the
QuakeMigrate run's stations.json.

Velocity model: two-layer (ice over bedrock), differentiated per array via
--array {T1,T2}. Ice layer Vp/Vs per array come from prior QuakeMigrate-era
work (T1: /home/jwalter/dr/time/input_files hypoDD velocity model; T2:
/scratch2/qm/t2/growclust/tests.inp + IN/vzmodel.txt). The ice-bed interface
depth (i.e. ice thickness) per array comes from BedMachine Antarctica v3 +
Bedmap2, queried at each array's centroid in qm_prelim.py: T1 ~3.24 km
(3209.66/3268 m average), T2 ~2.02 km (2010.82/2039 m average) -- these two
independent datasets agree well within each array. The bedrock layer below
the interface (Vp=5.1 km/s, Vp/Vs=1.8) is only independently measured for T1
(same hypoDD model); T2 reuses it for lack of an independent measurement --
flagged here since it is an assumption, not a measurement.

Usage:
    python full_catalog_pipeline/associate_pyocto.py \
        --picks-glob "full_catalog_pipeline/artifacts/pilot/picks_*_202007.csv" \
        --out-events full_catalog_pipeline/artifacts/pilot/pyocto_events.csv \
        --out-assignments full_catalog_pipeline/artifacts/pilot/pyocto_assignments.csv \
        --array T2
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyocto

# vels1d_model.py lives in 04_relocation/ (it was written for the hypoDD models); the
# same measured profile is what --velocity-source vels1d feeds to pyocto here.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "04_relocation"))
import vels1d_model

STATIONS_JSON = "/scratch2/qm/t1/output/working_files/stations.json"

BED_VP_KM_S = 5.1
BED_VPVS = 1.8
BED_VS_KM_S = BED_VP_KM_S / BED_VPVS

# Per-array ice layer velocity + ice-bed interface depth (see module docstring).
ARRAY_VELOCITY = {
    "T1": {"ice_vp": 3.85, "ice_vpvs": 2.0, "ice_thickness_km": 3.24},
    "T2": {"ice_vp": 3.841, "ice_vpvs": 1.949, "ice_thickness_km": 2.02},
}


def build_station_df(network="7U"):
    with open(STATIONS_JSON) as f:
        stations = json.load(f)
    rows = []
    for key, meta in stations.items():
        # key is like "2E.DRSC" in the QuakeMigrate metadata; our continuous
        # archive uses network code 7U for the same physical stations.
        code = key.split(".")[1]
        rows.append({
            "id": f"{network}.{code}.",
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "elevation": meta["elevation"],
        })
    return pd.DataFrame(rows)


def build_vels1d_model_df(array, max_depth_km, step_m=20.0):
    """pyocto model frame from the measured reflection profile (vels1d/), thickness-
    shifted to `array` exactly as the hypoDD models are -- see vels1d_model.py.

    Decimated to `step_m` because pyocto linearly interpolates velocity between
    consecutive rows onto its own grid; 20 m is finer than any grid we solve on, so the
    decimation is lossless in practice while keeping the frame small.
    """
    depth_m, vp_m_s, vs_m_s = vels1d_model.shifted_profile(array)
    keep = depth_m <= (max_depth_km + 1.0) * 1000.0
    depth_m, vp_m_s, vs_m_s = depth_m[keep], vp_m_s[keep], vs_m_s[keep]
    stride = max(1, int(round(step_m / (depth_m[1] - depth_m[0]))))
    idx = np.unique(np.r_[np.arange(0, len(depth_m), stride), len(depth_m) - 1])
    return pd.DataFrame({
        "depth": depth_m[idx] / 1000.0,
        "vp": vp_m_s[idx] / 1000.0,
        "vs": vs_m_s[idx] / 1000.0,
    })


def build_velocity_model(out_path, array, transition_halfwidth_km=0.01,
                         source="twolayer", delta=0.1, zdist=10.0):
    if source == "vels1d":
        model_df = build_vels1d_model_df(array, zdist)
        pyocto.VelocityModel1D.create_model(model_df, delta=delta, xdist=250, zdist=zdist,
                                            path=out_path)
        return pyocto.VelocityModel1D(out_path, tolerance=2.0)

    cfg = ARRAY_VELOCITY[array]
    ice_vp = cfg["ice_vp"]
    ice_vs = ice_vp / cfg["ice_vpvs"]
    interface = cfg["ice_thickness_km"]
    # Step at the ice-bed interface, over a transition zone of the given half-width
    # (default is a near-sharp 20 m step). An abrupt (near-zero-width) step creates a
    # travel-time degeneracy right at the interface depth that the grid-search location
    # can lock onto -- widening the transition removes that artifact.
    model_df = pd.DataFrame({
        "depth": [-1.0, interface - transition_halfwidth_km, interface + transition_halfwidth_km, 6.0],
        "vp": [ice_vp, ice_vp, BED_VP_KM_S, BED_VP_KM_S],
        "vs": [ice_vs, ice_vs, BED_VS_KM_S, BED_VS_KM_S],
    })
    pyocto.VelocityModel1D.create_model(model_df, delta=delta, xdist=250, zdist=zdist,
                                        path=out_path)
    return pyocto.VelocityModel1D(out_path, tolerance=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks-glob", required=True)
    ap.add_argument("--out-events", required=True)
    ap.add_argument("--out-assignments", required=True)
    ap.add_argument("--velocity-model-path", default="full_catalog_pipeline/artifacts/pilot/velocity_model")
    ap.add_argument("--array", choices=sorted(ARRAY_VELOCITY), required=True,
                     help="which array's ice-layer velocity + ice-bed interface depth to use")
    ap.add_argument("--n-p-and-s-picks", type=int, default=3,
                     help="min number of stations required to report BOTH P and S ('high-quality' stations)")
    ap.add_argument("--min-node-size-location", type=float, default=0.02,
                     help="octree leaf size (km) for the location refinement step; pyocto default is 1.5 km, "
                          "which quantizes reported depths into discrete bands -- lower for finer depths. "
                          "0.02 km (~14 m effective grid at location_split_depth=10 over a 7 km zlim) is well "
                          "below the method's ~200-300 m depth uncertainty vs QuakeMigrate, so pushing finer "
                          "than this buys little -- locations still snap to grid nodes, just imperceptibly so")
    ap.add_argument("--location-split-depth", type=int, default=10,
                     help="search depth for location splits; must increase alongside a smaller "
                          "--min-node-size-location to actually resolve finer minima (pyocto default is 6)")
    ap.add_argument("--location-split-return", type=int, default=7,
                     help="part of location_split_depth used to evenly sample the space rather than "
                          "descend; pyocto docs say this should increase alongside --exponential-edt")
    ap.add_argument("--no-exponential-edt", dest="exponential_edt", action="store_false",
                     help="disable the sharpened EDT loss surface (on by default). Without it, weakly-"
                          "constrained events default to a shared 'flat loss' grid node instead of truly "
                          "resolving -- verified this drops the fraction of events sharing one exact depth "
                          "from ~11%% to ~4%% on the T2 full-archive catalog with no precision/recall cost")
    ap.add_argument("--refinement-iterations", type=int, default=3,
                     help="number of localisation and pick-matching refinement iterations (pyocto default 3)")
    ap.add_argument("--transition-halfwidth-km", type=float, default=0.15,
                     help="half-width (km) of the ice-bed velocity transition zone. A near-zero width creates "
                          "a travel-time degeneracy right at the interface depth that pins ~10%% of events "
                          "there; 0.15 km cuts that to ~7%% with no further gain past ~0.3 km, and is itself a "
                          "physically defensible width for a gradational (not instantaneous) ice-bed contact")
    ap.add_argument("--velocity-source", choices=["twolayer", "vels1d"], default="twolayer",
                     help="'twolayer' (default, and what v5 used) is the per-array ice-over-bedrock "
                          "step described above. 'vels1d' is the measured reflection-seismology "
                          "profile from full_catalog_pipeline/vels1d/ -- the SAME model the hypoDD "
                          "relocations use -- thickness-shifted per array by vels1d_model.py. It has a "
                          "real firn gradient and a gradational ice-bed ramp, so --transition-halfwidth-km "
                          "does not apply to it")
    ap.add_argument("--model-delta", type=float, default=0.1,
                     help="eikonal grid spacing (km) for the travel-time table. v5 used pyocto's 0.1 km. "
                          "The vels1d firn gradient lives in the top 300 m, which 0.1 km cannot represent, "
                          "so use ~0.02 with --velocity-source vels1d (and rerun the twolayer model at the "
                          "same delta if you want the comparison to isolate the model, not the grid)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.picks_glob))
    print(f"Loading {len(files)} pick files: {files}")
    picks = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Total picks: {len(picks)}")

    stations = build_station_df()
    stations = stations[stations["id"].isin(picks["station"].unique())].reset_index(drop=True)
    print(f"Stations with picks: {len(stations)}")

    velocity_model = build_velocity_model(
        args.velocity_model_path, args.array, args.transition_halfwidth_km,
        source=args.velocity_source, delta=args.model_delta)
    print(f"Velocity model: source={args.velocity_source} delta={args.model_delta} km")

    lat_pad, lon_pad = 0.05, 0.1
    associator = pyocto.OctoAssociator.from_area(
        lat=(stations["latitude"].min() - lat_pad, stations["latitude"].max() + lat_pad),
        lon=(stations["longitude"].min() - lon_pad, stations["longitude"].max() + lon_pad),
        zlim=(-1, 6),
        velocity_model=velocity_model,
        time_before=30.0,
        n_picks=6,
        n_p_and_s_picks=args.n_p_and_s_picks,
        min_pick_fraction=0.2,
        min_node_size_location=args.min_node_size_location,
        location_split_depth=args.location_split_depth,
        location_split_return=args.location_split_return,
        exponential_edt=args.exponential_edt,
        refinement_iterations=args.refinement_iterations,
    )
    stations_xyz = associator.transform_stations(stations)

    events, assignments = associator.associate(picks, stations_xyz)
    print(f"Associated {len(events)} events from {len(assignments)} picks")

    if len(events) > 0:
        events = associator.transform_events(events)

    os.makedirs(os.path.dirname(args.out_events), exist_ok=True)
    events.to_csv(args.out_events, index=False)
    assignments.to_csv(args.out_assignments, index=False)
    print(f"Wrote {args.out_events}, {args.out_assignments}")


if __name__ == "__main__":
    main()
