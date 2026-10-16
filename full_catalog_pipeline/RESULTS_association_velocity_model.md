# Should the association stage use the vels1d velocity model?

> **Bottom line (2026-09-18, all four rebuild arms complete):** no, not for its own sake.
> The association velocity model changes nothing measurable at T1 and changes T2's
> epicentres without improving the fit. The exercise's real finding is that the **x2
> upsampled dt.cc** raises P cross-correlation observations 1800x at T1 and 213x at T2,
> improves the median rcc residual 0.0100 -> 0.0090 s at both, and relocates more events.
> **Neither v6 arm has been through the validation v5 received** — see "What is NOT done".

Run 2026-09-17. Answers the question: the pyocto association stage locates events with the
v5 two-layer ice-over-bedrock model, while hypoDD relocates them with the measured vels1d
reflection profile. Does making the two consistent change anything, and is the sequential
redo worth it?

Scripts: `03_association/associate_pyocto.py --velocity-source vels1d`,
`03_association/compare_velocity_model_catalogs.py`,
`03_association/plot_association_velocity_model_diff.py`,
`04_relocation/hypodd_initial_location_test.py`,
`04_relocation/ph2dt_pair_overlap_test.py`,
`04_relocation/rebuild_v6_from_vels1d_association.py`.

## The control that makes the numbers readable

Every comparison below is quoted against a **noise floor**: the SAME two-layer model rerun
on a 20 m eikonal grid instead of production's 100 m. That is pure numerical quantisation,
no physics. It moves epicentres ~30 m and depths ~96 m. Any effect near those values is not
a velocity-model effect.

The 20 m twolayer run also reproduces production exactly (2400 / 4277 events, 99.99% of pick
assignments identical), which validates the harness.

## 1. The association stage

Same picks, same 20 m grid, only the model differs.

| | events | epicentre med / p95 | *floor* | depth med / \|Δ\|med / p5 | *floor* |
|---|---|---|---|---|---|
| T1 | 2400 → 2399 | 75 / 725 m | *34 m* | −68 / 82 / −670 m | *96 m* |
| T2 | 4277 → 4277 | **185 / 1051 m** | *30 m* | **−219 / 260 / −1135 m** | *96 m* |

**T1's depth change (82 m) is below the 96 m numerical floor.** T2's is real: epicentres move
6× the floor and the whole depth distribution slides shallower (median 1.63 → 1.47 km, p5
0.97 → 0.14 km).

Unchanged on both arrays: event count, and **99.9% of pick-to-event assignments**. The vels1d
model relocates the same events; it does not detect or associate differently.

Why T2 and not T1: T1's old model already assumed Vp/Vs 2.0 against the measured 2.01, so only
the firn differs. T2 assumed 1.949, its ice is thinner so firn is a larger fraction of each
ray path, and the measured ice–bed ramp (2.0–2.5 km) sits exactly where T2's events are.

### Against QuakeMigrate — no gain either way

| | precision / recall | epicentral offset | depth offset |
|---|---|---|---|
| T1 two-layer | 0.752 / 0.172 | 0.24 km | 223 m |
| T1 vels1d | 0.752 / 0.172 | 0.29 km | 315 m |
| T2 two-layer | 0.763 / 0.222 | 0.18 km | 321 m |
| T2 vels1d | 0.763 / 0.222 | 0.25 km | 396 m |

Detection quality is bit-identical; agreement with the independent catalogue gets slightly
*worse*. QuakeMigrate carries its own velocity model, so "closer to QM" is not "more correct" —
but there is no external evidence that the vels1d association is better, only different.

## 2. Downstream, T1 — and the finding that actually matters

Both T1 arms rebuilt completely (new phase.dat locations, new ph2dt, new dt.cc, new hypoDD).

| catalogue | events | rcc median | **P cc-obs** | S cc-obs | depth p5/med/p95 (km) |
|---|---|---|---|---|---|
| v5 | 2164 | 0.0100 | **400** | 109,512 | 2.09 / 2.89 / 3.39 |
| v6/vmonly (velocity model only) | 2148 | 0.0100 | **318** | 112,404 | 2.05 / 2.87 / 3.39 |
| v6/allfix (+ MAXSEP + upsampling) | 2153 | **0.0090** | **568,528** | 272,910 | 2.34 / 2.81 / 3.28 |

| comparison | epicentre med | depth \|Δ\|med | within-cluster (centroid removed) |
|---|---|---|---|
| v5 → vmonly — *velocity model alone* | 239 m | 80 m | **29 / 20 m** |
| v5 → allfix — *all three changes* | 207 m | 159 m | 108 / 82 m |
| vmonly → allfix — *upsampling + MAXSEP* | 357 m | 167 m | **241 / 95 m** |

**The association velocity model is nearly irrelevant; the dt.cc upsampling fix is
transformative.** Changing the model alone moves cluster *shape* by 29/20 m and leaves the
median rcc residual at 0.0100 s — it translates each cluster rigidly without fitting better.
Adding the x2-upsampled dt.cc takes P cross-correlation observations from **400 to 568,528**
(~1800x — the inversion finally has real P constraint), improves the residual to 0.0090 s, and
tightens the depth distribution from both ends.

**This is why the isolated arm was worth running.** Bundled, the improvement would have looked
like a velocity-model win when it is almost entirely the P-deficit fix
(`05_crosscorrelation/regenerate_dtcc.py --upsample 2`).

## 2b. Downstream, T2 (full rebuild)

`hypodd_v6_vmonly` — association velocity model is the only intended change. 4 of 3516 rows
came out with NaN coordinates (0.1%, the usual air-quake dropout) and are excluded below.

| | v5 | v6/vmonly |
|---|---|---|
| events relocated (clean) | 3293 | **3512** |
| rcc median | 0.0100 s | 0.0100 s |
| P cc-obs | 11,558 | 14,304 |
| depth p5 / med / p95 | 1.06 / 1.66 / **2.49** km | 0.75 / 1.67 / **2.02** km |
| epicentre shift | — | median **441 m** |
| within-cluster (centroid removed) | — | 35 / 71 m |

T2 behaves as the association-stage test predicted and unlike T1: epicentres move **441 m**,
and this survives a full self-consistent rebuild rather than being a shortcut artifact. The
deep tail tightens sharply — p95 2.49 -> 2.02 km, i.e. events stop extending well below the
ice base, which is physically sensible under the measured model. The fit is unchanged
(rcc 0.0100 s), so this remains "different, not demonstrably better", but at T2 the difference
is large and has a physical rationale.

## 3. T2's inversion is fragile to a STALE PAIR SET (corrected)

An earlier draft of this document claimed T2's relocation collapses under a ~96 m
starting-location nudge. **The full rebuild refuted that and the claim is withdrawn.**

| inputs | events relocated |
|---|---|
| byte-identical production `event.sel` + `dt.ct` + `dt.cc.authoritative` | 3293 (reproduces production) |
| production `dt.ct`/`dt.cc`, `event.sel` swapped ~96 m (same velocity model) | **0** — `STOP >>> More than 1000 air quakes` |
| full self-consistent rebuild (new locations AND new `dt.ct` AND new `dt.cc`) | **3516** |

The divergence requires an `event.sel` inconsistent with the pair set `ph2dt` selected around
the OLD locations. It is not a function of perturbation size: the full rebuild moves locations
by *more* and converges, returning more events than production. The failing `event.sel` was
verified well-formed (identical 91-character lines and 10 fields throughout), so this is not a
file-format artifact.

**The methodological lesson is the durable part.** The "swap `event.sel`, keep `dt.ct`/`dt.cc`"
shortcut in `04_relocation/hypodd_initial_location_test.py` is **unsound at T2** — it worked at
T1 and silently returned a null at T2. Any future starting-location experiment must regenerate
`dt.ct` as well, i.e. use `04_relocation/rebuild_v6_from_vels1d_association.py`.

T2's recorded 41.5% LSQR non-convergence with the un-upsampled dt.cc remains a separate, real
concern — but it is not what this experiment demonstrated.

## Cost of a redo (measured, not estimated)

`regenerate_dtcc.py` replaces hypoDDpy's ~10 h cross-correlation with a ~23 min prefetch
rebuild, so a redo is ~30 min/arm at T1 and ~45 min at T2 — not an overnight job.

ph2dt pair-set overlap, if instead reusing dt.cc incrementally (MAXSEP held fixed):
T1 92.1% of pairs carry over (27,833 new); T2 only 45.1% (208,659 new). Wholesale regeneration
is simpler and faster than exploiting that reuse.

## Traps hit, worth not re-hitting

- `regenerate_dtcc.py` defaults T1's network code to **2E**, but both arrays' phase.dat and
  station.sel use **7U**. Always pass `--network 7U`; otherwise dt.cc station names match
  nothing and the run is silently void.
- Production `ph2dt.inp` has **no trailing newline**, so parsing the parameter line by a naive
  index off the end of `split("\n")` returns the phase.dat filename.
- T2's `event.sel` contains two events with hypoDD-rounded `SS=60`, which
  `datetime.datetime(...)` rejects; build the time with a `timedelta` instead.
- T1's production dt.ct was built with MAXSEP **0.9747 km**, the geodetically-wrong inflated
  value. The corrected value is 0.383 km. T2's (0.1905) is already close to correct (0.1979),
  so this fix bites almost entirely at T1.

## Figures

- `figures/03_assoc_velocity_model_diff.png` — the two associations, map / section /
  depth distribution / per-event change. **This is the association-only comparison.**
- `figures/03_assoc_vmtest_models_traveltimes.png` — the models and the Δ(S−P) they imply.
- `figures/03_assoc_vmtest_catalog_shifts.png` — shifts against the noise floor.
- `figures/04_reloc_vmtest_downstream.png` — downstream response and redo cost.


## 4. All four rebuild arms (complete 2026-09-18)

Air-quake rows with non-finite coordinates excluded (<=0.2% everywhere).

### T1
| catalogue | events | rcc med | P cc-obs | S cc-obs | depth p5/med/p95 (km) |
|---|---|---|---|---|---|
| v5 (production) | 2164 | 0.0100 | 400 | 109,512 | 2.09 / 2.89 / 3.39 |
| v6/vmonly | 2148 | 0.0100 | 318 | 112,404 | 2.05 / 2.87 / 3.39 |
| v6/allfix | 2153 | **0.0090** | **568,528** | 272,910 | 2.34 / 2.81 / 3.28 |

### T2
| catalogue | events | rcc med | P cc-obs | S cc-obs | depth p5/med/p95 (km) |
|---|---|---|---|---|---|
| v5 (production) | 3293 | 0.0100 | 11,558 | 555,632 | 1.06 / 1.66 / 2.49 |
| v6/vmonly | 3512 | 0.0100 | 14,304 | 464,252 | 0.75 / 1.67 / 2.02 |
| v6/allfix | **3650** | **0.0090** | **2,462,068** | 848,198 | 0.75 / 1.47 / 1.90 |

### Displacement from v5
| | epicentre med | depth \|Δ\|med | within-cluster (centroid removed) |
|---|---|---|---|
| T1 → vmonly | 239 m | 80 m | 29 / 20 m |
| T1 → allfix | 207 m | 159 m | 108 / 82 m |
| T2 → vmonly | 441 m | 88 m | 35 / 71 m |
| T2 → allfix | **1305 m** | 293 m | 43 / 92 m |

T2/allfix's 1.3 km epicentre move is large in absolute terms, though within-cluster geometry
holds (43/92 m), so it is largely a rigid translation per cluster. It deserves scrutiny before
promotion.

Runtimes (this machine, 8 CC workers): T1 ~28 min/arm, T2/vmonly 53 min, T2/allfix 105 min
(dt.cc 31 min + hypoDD 74 min — the last is slow because dt.cc.filtered holds 2.56M
observations versus vmonly's 478k).

## What is NOT done — do not skip before promoting anything

* **Neither v6 arm has had the validation v5 got.** No held-out CV, no per-cluster depth-shift
  stability across two seeds. `hypodd_tune.py` exists for exactly this, and the recorded lesson
  ([[hypodd-tuning-metric-lessons]]) is that held-out RMS and median-only stability BOTH gave
  confident wrong answers here — rank on median|residual| plus per-cluster depth shift across
  two seeds.
* ~~Nothing has been promoted.~~ **PROMOTED 2026-09-18**: `catalog_paths.AUTHORITATIVE` now
  defaults to `hypodd_v6_allfix` for both arrays, with a `PROVENANCE.md` in each run dir. v5
  stays on disk and is reachable via `ICEQUAKE_RELOC=hypodd_vels1d`.
* **v5-derived artifacts do NOT carry over.** `work_dir()` follows the authoritative
  relocation, and cluster IDs/membership differ between v5 and v6, so cluster event-id lists,
  polarity CSVs, stacks and the focal-mech / cluster-structure results must be regenerated
  against v6 before being quoted alongside it.
* **T2/allfix's 1.3 km epicentre shift is unexplained** beyond "rigid per-cluster translation".
* **The upsampling fix is confounded with the MAXSEP change** inside the promoted `allfix`
  catalogue. The T1 `vmonly` arm shows the association velocity model contributes ~nothing,
  so the gain is essentially all upsampling, but MAXSEP and upsampling are not separated.
  An `upsonly` arm was started for this and deliberately abandoned as not worth the compute.
