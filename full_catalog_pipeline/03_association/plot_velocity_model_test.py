#!/usr/bin/env python3
"""Figures for the association-stage velocity-model test (2026-09-17).

Question tested: the pyocto association stage still locates events with the v5 two-layer
ice-over-bedrock model, while hypoDD relocates them with the measured vels1d reflection
profile. Would re-associating under vels1d change anything worth the cost of redoing the
association and then the relocation in sequence?

Three figures, one per link in that chain:

  1  ..._models_traveltimes.png  -- the two models, and the travel-time differences they
     imply. Establishes how large the input perturbation even is, before any association.
  2  ..._catalog_shifts.png      -- what actually changed in the pyocto catalogs, each
     against a NOISE FLOOR measured by rerunning the SAME model on a finer eikonal grid.
     Without that control a reader cannot tell a model effect from grid quantisation.
  3  ..._downstream.png          -- how far the change survives into hypoDD: the final
     relocation's response to changed starting locations, and the ph2dt pair-set overlap
     that bounds how much cross-correlation a redo would actually cost.

Usage (from the repository root):
    python full_catalog_pipeline/03_association/plot_velocity_model_test.py
"""
import os as _os, sys as _sys
_PIPE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_PIPE] + [_os.path.join(_PIPE, _s) for _s in sorted(_os.listdir(_PIPE))
                     if _os.path.isdir(_os.path.join(_PIPE, _s))
                     and not _s.startswith(('.', 'artifacts', '__'))]:
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyocto

import vels1d_model as V
from associate_pyocto import ARRAY_VELOCITY, BED_VP_KM_S, BED_VPVS, build_vels1d_model_df

VM = "full_catalog_pipeline/artifacts/full_run/vmtest"
FIGS = "full_catalog_pipeline/figures"
SCRATCH = "/tmp/claude-1000/-home-jwalter-Icequake-ML/84c22eb9-8117-4218-8587-6d282c2f537c/scratchpad/vmtest"

INK = "#0b0b0b"
MUTED = "#55534e"
GRID = "#e4e3de"
# Categorical slots, fixed order, validated with the dataviz palette checker
# (worst adjacent CVD dE 24.7, all six checks PASS against a light surface).
C_V5 = "#7d5bd6"        # v5 production (two-layer, 100 m eikonal grid)
C_TWO = "#2a78d6"       # two-layer re-run on the fine grid -- the control
C_VELS = "#eb6834"      # vels1d measured profile
C_REF = "#8a8a86"       # neutral: reference bands and annotations only, never a series

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.grid": True,
    "grid.color": GRID,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

DELTA, XDIST, ZDIST = 0.02, 40.0, 8.0


def twolayer_df(array, halfwidth=0.15):
    c = ARRAY_VELOCITY[array]
    i = c["ice_thickness_km"]
    return pd.DataFrame({
        "depth": [-1.0, i - halfwidth, i + halfwidth, 6.0],
        "vp": [c["ice_vp"]] * 2 + [BED_VP_KM_S] * 2,
        "vs": [c["ice_vp"] / c["ice_vpvs"]] * 2 + [BED_VP_KM_S / BED_VPVS] * 2,
    })


def tt_table(model_df, path):
    pyocto.VelocityModel1D.create_model(model_df, delta=DELTA, xdist=XDIST, zdist=ZDIST, path=path)
    with open(path, "rb") as f:
        nx, nz, _ = struct.unpack("iid", f.read(struct.calcsize("iid")))
        p = np.frombuffer(f.read(nx * nz * 8), dtype=np.float64).reshape(nx, nz)
        s = np.frombuffer(f.read(nx * nz * 8), dtype=np.float64).reshape(nx, nz)
    return p, s


def ecdf(ax, v, color, label, lw=2):
    v = np.sort(np.asarray(v))
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=lw, label=label)


def annotate_medians(ax, entries):
    """Mark each series' median on the ECDF at y=0.5.

    The two medians can sit within a few tens of metres of each other (they do at T1,
    which is the point of the panel), so the labels are staggered vertically and pushed to
    opposite sides rather than both placed to the lower right, where they would overprint.
    """
    entries = sorted(entries, key=lambda e: e[0])
    offsets = [(-8, 16), (8, -20)]
    aligns = ["right", "left"]
    for (v, col), (dx, dy), ha in zip(entries, offsets, aligns):
        ax.plot([v], [0.5], "o", color=col, ms=7, zorder=5)
        ax.annotate(f"{v:.0f} m", (v, 0.5), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9.5, color=col, fontweight="bold", ha=ha,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))


def load(d):
    return (pd.read_csv(f"{d}/pyocto_events.csv"),
            pd.read_csv(f"{d}/pyocto_assignments.csv"))


def match(a, b, tol=0.5):
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


# ---------------------------------------------------------------- figure 1
def figure_models():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    dists = [0.5, 1.0, 2.0, 4.0, 8.0]
    for r, array in enumerate(["T1", "T2"]):
        two, vel = twolayer_df(array), build_vels1d_model_df(array, ZDIST)
        thick = ARRAY_VELOCITY[array]["ice_thickness_km"]
        marks = V.bed_markers(array)

        for c, key, name in [(0, "vp", "Vp"), (1, "vs", "Vs")]:
            ax = axes[r, c]
            ax.step(two[key], two["depth"], where="post", color=C_TWO, lw=2,
                    label="two-layer (v5, used for association)")
            ax.plot(vel[key], vel["depth"], color=C_VELS, lw=2,
                    label="vels1d measured (used for hypoDD)")
            ax.axhspan(marks["ice_base"], marks["bedrock_top"], color=C_REF, alpha=0.16, lw=0)
            ax.axhline(thick, color=C_REF, lw=1, ls="--")
            ax.set_ylim(4.6, -0.1); ax.set_xlabel(f"{name} (km/s)")
            if c == 0:
                ax.set_ylabel(f"{array}\n\nDepth (km)")
            ax.set_title(f"{array} — {name}", fontsize=11)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower left", framealpha=0.95)

        pT, sT = tt_table(two, f"{SCRATCH}/fig_{array}_two.bin")
        pV, sV = tt_table(vel, f"{SCRATCH}/fig_{array}_vel.bin")
        ax = axes[r, 2]
        z = np.arange(0.2, 4.0, DELTA)
        iz = np.round(z / DELTA).astype(int)
        cmap = plt.cm.plasma(np.linspace(0.12, 0.8, len(dists)))
        for col, dd in zip(cmap, dists):
            ix = int(round(dd / DELTA))
            dsp = ((sV[ix, iz] - pV[ix, iz]) - (sT[ix, iz] - pT[ix, iz])) * 1000
            ax.plot(dsp, z, color=col, lw=2, label=f"{dd:g} km")
        ax.axvline(0, color=INK, lw=1)
        ax.axhspan(marks["ice_base"], marks["bedrock_top"], color=C_REF, alpha=0.16, lw=0)
        ax.set_ylim(4.6, -0.1); ax.set_xlim(-40, 260)
        ax.set_xlabel("Δ(S−P), vels1d − two-layer (ms)")
        ax.set_title(f"{array} — S−P change (depth control)", fontsize=11)
        ax.legend(fontsize=8, title="epicentral dist.", title_fontsize=8, loc="lower right")

    fig.suptitle("Velocity models compared, and the travel-time perturbation they imply\n"
                 "Shaded band = the reflection profile's gradational ice–bed ramp; "
                 "dashed line = BedMachine ice thickness",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = f"{FIGS}/03_assoc_vmtest_models_traveltimes.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- figure 2
def figure_catalogs():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    summary = {}
    for r, array in enumerate(["T1", "T2"]):
        v5, _ = load(f"full_catalog_pipeline/artifacts/full_run/{array}_v5")
        two, _ = load(f"{VM}/{array}_twolayer_d02")
        vel, _ = load(f"{VM}/{array}_vels1d_d02")

        a, b = match(v5, two)                 # grid-only change  -> the NOISE FLOOR
        noise_h = np.hypot((b.x - a.x), (b.y - a.y)) * 1000
        noise_z = np.abs(b.depth - a.depth) * 1000
        a2, b2 = match(two, vel)              # model change at matched grid
        mod_h = np.hypot((b2.x - a2.x), (b2.y - a2.y)) * 1000
        mod_z = np.abs(b2.depth - a2.depth) * 1000
        summary[array] = (np.median(noise_h), np.median(mod_h),
                          np.median(noise_z), np.median(mod_z))

        ax = axes[r, 0]
        ecdf(ax, noise_h, C_REF, "grid refinement only (noise floor)")
        ecdf(ax, mod_h, C_VELS, "velocity model changed")
        ax.set_xlim(0, 900); ax.set_xlabel("Epicentral shift (m)")
        ax.set_ylabel(f"{array}\n\nfraction of events ≤ x")
        ax.set_title(f"{array} — epicentre shift vs noise floor", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        annotate_medians(ax, [(np.median(noise_h), C_REF), (np.median(mod_h), C_VELS)])

        ax = axes[r, 1]
        ecdf(ax, noise_z, C_REF, "grid refinement only (noise floor)")
        ecdf(ax, mod_z, C_VELS, "velocity model changed")
        ax.set_xlim(0, 900); ax.set_xlabel("|Depth shift| (m)")
        ax.set_title(f"{array} — depth shift vs noise floor", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        annotate_medians(ax, [(np.median(noise_h), C_REF), (np.median(mod_h), C_VELS)])

        ax = axes[r, 2]
        bins = np.arange(0, 5.0, 0.1)
        for df, col, lab in [(v5, C_V5, "v5 production (two-layer, 100 m grid)"),
                             (two, C_TWO, "two-layer, 20 m grid (control)"),
                             (vel, C_VELS, "vels1d, 20 m grid")]:
            ax.hist(df.depth, bins=bins, histtype="step", lw=2, color=col, label=lab)
        marks = V.bed_markers(array)
        ax.axvspan(marks["ice_base"], marks["bedrock_top"], color=C_REF, alpha=0.16, lw=0)
        ax.set_xlabel("pyocto depth (km)"); ax.set_ylabel("events")
        ax.set_title(f"{array} — catalogue depth distribution", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("What the velocity model actually changed in the pyocto catalogues\n"
                 "Every model effect is plotted against a noise floor from rerunning the SAME "
                 "model on a finer grid — at T1 the two nearly coincide, at T2 they separate",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{FIGS}/03_assoc_vmtest_catalog_shifts.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)
    for k, v in summary.items():
        print(f"  {k}: epi noise {v[0]:.0f} m vs model {v[1]:.0f} m | "
              f"depth noise {v[2]:.0f} m vs model {v[3]:.0f} m")


# ---------------------------------------------------------------- figure 3
def read_reloc(path):
    cols = ["id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez", "yr", "mo", "dy",
            "hr", "mi", "sc", "mag", "nccp", "nccs", "nctp", "nvts", "rcc", "rct", "cid"]
    return pd.read_csv(path, sep=r"\s+", names=cols)


def figure_downstream():
    """hypoDD's response, and what a redo would cost.

    The framing matters here: hypoDD does NOT damp the starting perturbation -- final
    locations move about as far as the starting locations did (roughly 1:1). What makes
    the change harmless is different and is panel 2: once each cluster's centroid is
    removed, the WITHIN-cluster geometry barely moves, so the perturbation is close to a
    rigid translation per cluster rather than a distortion of the structure the science
    rests on.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    base = read_reloc("full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd_vels1d/output_files/hypoDD.reloc")
    new = read_reloc(f"{VM}/T1_hypodd_initloc/output_files/hypoDD.reloc")
    m = base.merge(new, on="id", suffixes=("_b", "_n"))

    # ---- panel 1: the perturbation is passed through, not absorbed
    v5, _ = load("full_catalog_pipeline/artifacts/full_run/T1_v5")
    vel, _ = load(f"{VM}/T1_vels1d_d02")
    a, b = match(v5, vel)
    start_dz = (b.depth.to_numpy() - a.depth.to_numpy()) * 1000
    final_dz = ((m.depth_n - m.depth_b) * 1000).to_numpy()
    ax = axes[0, 0]
    bins = np.arange(-800, 820, 40)
    ax.hist(start_dz, bins=bins, histtype="step", lw=2, color=C_VELS,
            label=f"pyocto STARTING depths   median |Δ| {np.median(np.abs(start_dz)):.0f} m")
    ax.hist(final_dz, bins=bins, histtype="step", lw=2, color=C_TWO,
            label=f"hypoDD FINAL depths        median |Δ| {np.median(np.abs(final_dz)):.0f} m")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Depth change (m)"); ax.set_ylabel("events")
    ax.set_title("T1 — hypoDD passes the change through ~1:1, it does not damp it",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

    # ---- panel 2: but it is a rigid translation per cluster, not a distortion
    ax = axes[0, 1]
    dh_abs = np.hypot(m.x_n - m.x_b, m.y_n - m.y_b).to_numpy()
    dz_abs = np.abs(final_dz)
    rel_h, rel_z = [], []
    for c, g in m.groupby("cid_b"):
        if len(g) < 20:
            continue
        rx = (g.x_n - g.x_n.mean()) - (g.x_b - g.x_b.mean())
        ry = (g.y_n - g.y_n.mean()) - (g.y_b - g.y_b.mean())
        rz = (g.z_n - g.z_n.mean()) - (g.z_b - g.z_b.mean())
        rel_h += list(np.hypot(rx, ry)); rel_z += list(np.abs(rz))
    ecdf(ax, dh_abs, C_V5, f"absolute epicentre  (median {np.median(dh_abs):.0f} m)")
    ecdf(ax, dz_abs, C_TWO, f"absolute depth  (median {np.median(dz_abs):.0f} m)")
    ecdf(ax, rel_h, C_VELS, f"within-cluster epicentre  (median {np.median(rel_h):.0f} m)")
    ecdf(ax, rel_z, C_REF, f"within-cluster depth  (median {np.median(rel_z):.0f} m)")
    ax.set_xlim(0, 400); ax.set_xlabel("Change in final hypoDD location (m)")
    ax.set_ylabel("fraction of events ≤ x")
    ax.set_title("T1 — cluster centroids move; cluster SHAPE does not", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")

    # ---- panel 3: the fit does not improve
    ax = axes[1, 0]
    ecdf(ax, base.rcc, C_V5, f"baseline starts  (median {np.median(base.rcc):.4f} s)")
    ecdf(ax, new.rcc, C_VELS, f"vels1d starts     (median {np.median(new.rcc):.4f} s)")
    ax.set_xlim(0, 0.06); ax.set_xlabel("Cross-correlation residual RCC (s)")
    ax.set_ylabel("fraction of events ≤ x")
    ax.set_title("T1 — fit quality is unchanged: no accuracy is gained", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.annotate(f"events relocated:  {len(base)} → {len(new)}",
                xy=(0.04, 0.12), xycoords="axes fraction", fontsize=9, color=MUTED)

    # ---- panel 4: what a redo would cost, per array
    ax = axes[1, 1]
    cost = [("T1", 92.1), ("T2", 45.1)]
    for i, (arr, surv) in enumerate(cost):
        ax.barh([i], [surv], color=C_TWO, height=0.45)
        ax.barh([i], [100 - surv], left=[surv], color=C_VELS, height=0.45)
        ax.annotate(f"{surv:.0f}% reusable", (surv / 2, i), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=11)
        ax.annotate(f"{100-surv:.0f}% to recompute", (100, i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5,
                    color=C_VELS, fontweight="bold")
    ax.set_yticks(range(len(cost)))
    ax.set_yticklabels([f"{a}\n({n:,} new pairs)" for (a, _), n in
                        zip(cost, [27833, 208659])])
    ax.set_xlim(0, 128); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of the ph2dt event-pair set (MAXSEP held fixed)")
    ax.set_title("Cost of a redo: T1 is cheap, T2 is not", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()

    fig.suptitle("How far the change survives downstream\n"
                 "The final catalogue does move ~100 m — but rigidly, per cluster, with no gain in fit",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{FIGS}/04_reloc_vmtest_downstream.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    figure_models()
    figure_catalogs()
    figure_downstream()
