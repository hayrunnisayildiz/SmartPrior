# SmartPrior — Geochemistry + Petrophysics Pipeline

**One line:** a Julia/Lux.jl neural field that predicts ore grade, density,
magnetic susceptibility, and rock-specimen conductivity on every cell of a 3D
block model, trained only on sparse geochemistry, lithology, and
petrophysical measurements — no gravity, no magnetics, no MT as input.

The active line of work is this **non-geophysical prior**. It is trained on
the **Cloncurry–Ernest Henry (METAL, Queensland)** collection as the current
case study, not as a claim that the same weights transfer to other deposits.
Keivitsa (GTK, Finland) was the previous case study; its readers remain in
`src/KeivitsaIO.jl` but are not the training target.

An earlier gravity + MT pipeline (warm-start prior for VFSA) is still in
the repo (`examples/compare_prior_2d.jl`, `examples/musgrave_*.jl`) and its
run directories live under `archive/mt_gravity_tmp_outputs/`. The two
pipelines share Grid / PriorNet / Train utilities, but not inputs, outputs,
or the training loop. This pipeline does **not** feed VFSA.

---

## Status

| Question | Answer |
|---|---|
| Active dataset | **Cloncurry–Ernest Henry** (METAL, 1,590 samples) |
| Fourth head | **`conductivity_100kHz`** — KT-20, 100 kHz, specimen-scale. Not MT bulk conductivity; do not call it resistivity |
| Connected to VFSA? | **No.** Deferred — see [Relationship to VFSA](#relationship-to-vfsa) |
| Full-data D (Ernest Henry) | log10-RMSE **0.005** on **104** grade cells. **Memorization**, not a hold-out. Do not report as better than Keivitsa (0.249 on 988 cells) |
| Leak-free hold-out | Drillhole groups on Ernest Henry, 191/45/19 **Cu samples** (7/2/2 holes). Grade test log10-RMSE **0.697** (beats train-mean 0.908). Density / susceptibility / `conductivity_100kHz` all **lose to naive**. Single seed |
| District hold-out | Driver exists (`examples/holdout_cloncurry_prior.jl`, sample AABB, ~70/15/15 of named collars). **No completed run yet** |

Keivitsa experiment numbers (25 m grid, 300-epoch checkpoint, etc.) are in
[Previous case study (Keivitsa)](#previous-case-study-keivitsa). They are not
Cloncurry results.

---

## Data (not in this repository)

Training does not download anything. `examples/train_cloncurry_prior.jl`
reads a local METAL-derived package:

```
CLONCURRY_ROOT  (or, if unset, ~/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17)
└── derived/
    ├── petrophysics_samples.csv      # coordinates, lithology, density / κ / σ_100kHz
    └── metal_all_fields.csv.gz       # pXRF *_Concentration (ppm)
```

Set `CLONCURRY_ROOT` if the package lives somewhere else. `mt/`, `gravity/`
and `magnetics/` are in that tree and are **not** opened. Tiny copies for
unit tests only: `test/fixtures/cloncurry_*.csv`.

`tmp_cloncurry_prior*` directories are **output** (checkpoints, reports),
not the source data.

---

## Input

Four data families. No gravity, no magnetics, no MT.

| # | Channel group | Source | Count (Cloncurry) | Role |
|---|---|---|---:|---|
| 1 | Position | grid centres | — | x, y, z (MGA54, m ASL) → Fourier encoding (`n_bands=4`) |
| 2 | Geochemistry | `derived/metal_all_fields.csv.gz` `*_Concentration` | 1,590 | pXRF ppm, Mg→U except **Cu** |
| 3 | Lithology | `petrophysics_samples.csv` `lithology_code` | 1,590 | one-hot (empty / rare → OTHER) |
| 4 | Coverage | derived | — | Distance to nearest sample |

`Cu_Concentration` is the grade *target* (anchor), not a feature — same
leakage rule as Keivitsa drill Cu. `<LOD` on input channels is treated as
missing so each element interpolates independently.

---

## Architecture

```
[position, geochemistry, lithology, coverage]
                    │
          Fourier positional encoding (coordinates only)
                    │
              PriorNet (Lux.jl MLP, gelu)
         width = 256, depth = 4
                    │
        ┌───────────┼───────────┬──────────────────────┐
   mu_grade    mu_density  mu_susceptibility  mu_conductivity_100kHz
   sigma_grade sigma_density sigma_susceptibility sigma_conductivity_100kHz
```

Eight outputs (`Dense(width => 8)`): four properties, each with a mean (`μ`)
and a heteroscedastic uncertainty (`σ`). Cloncurry starts at **width = 256**,
depth = 4, and a per-property σ floor of `1 ×` group std.

**Per cell:**
- `μ` — point estimate (`μ_grade` is Cu_Log; exported `grade` is `10^μ_grade`
  in ppm). Density is g/cm³. Susceptibility and conductivity_100kHz are log10.
- `σ` — learned cell-wise uncertainty via heteroscedastic NLL in
  `src/Losses.jl`. After the data-driven floor (`σ ≥ 1 ×` group std of the
  anchors), most anchors sit on that floor, so `σ` currently carries little
  cell-to-cell information.

The fourth head is **`conductivity_100kHz`**, not resistivity. METAL measured
small core specimens with a KT-20 at 100 kHz. That is not bulk low-frequency
MT conductivity; do not treat `μ_conductivity_100kHz` as an MT starting model.
See Austin et al. (2024), §5.2.4.

---

## Output — anchors and confidence

Supervision is only at anchor cells. The rest of the grid is filled by the
learned geochemistry/lithology → property map. There is no kriging-style
spatial continuity prior: distant cells can get similar predictions if their
features match.

A sample is an anchor for a property only when that property is populated
(same missing-data rule as Keivitsa PTR_D/PTR_J). Counts below are METAL
rows. On grid D, those rows collapse to ~104 occupied cells.

| Property | Anchor source | Rows (of 1,590) | Units in the NLL | Confidence |
|---|---|---:|---|---|
| `grade` (Cu_Log) | `Cu_Concentration` (pXRF ppm, log10; `<LOD` → 1 ppm) | 1,185 | log10 ppm | Direct pXRF |
| `density` | `density_mean_g_cm3` | 1,288 | g/cm³ | Direct (`density_mean_kg_m3` is a ×1,000 check only) |
| `susceptibility` | `susceptibility_mean_SI` | 1,288 | log10 SI (>0) | Direct |
| `conductivity_100kHz` | `conductivity_mean_S_m_100kHz` | 1,250 | log10 S/m (zeros lifted to 0.01) | Specimen measurement; **not MT-equivalent** |

Of the 1,590 samples, 457 lie in the requested work area
(140.5–140.9°E, 20.68–19.99°S). Ernest Henry itself has 269 finite-xyz
samples, of which 255 carry Cu and all 255 map into grid D (104 grade
cells). The remaining district Cu (1,185 − 255) sits outside this box.
The sample AABB covering every finite-xyz row is ~118 × 219 km; the
42 × 76 km work box drops ~1,100 of them.

---

## Relationship to VFSA

Not connected. VFSA2D in this package accepts a log-resistivity starting
grid only. Cloncurry's fourth head is specimen conductivity at 100 kHz, not
that quantity. Explicitly deferred.

---

## How to run

Julia 1.10+, from the repo root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# District sample AABB (current default), spacing from SMARTPRIOR_TARGET_CELLS
julia --project=. examples/train_cloncurry_prior.jl

# Grid D — Ernest Henry, 100 m × 50 m, 250 epoch (full-data; not a hold-out)
SMARTPRIOR_WORK=tmp_cloncurry_prior_eh SMARTPRIOR_BOX=ernest_henry \
  SMARTPRIOR_CELL_M=100 SMARTPRIOR_CELL_Z=50 SMARTPRIOR_EPOCHS=250 \
  julia --project=. examples/train_cloncurry_prior.jl

# VTK + Cu isosurface from a checkpoint
SMARTPRIOR_WORK=tmp_cloncurry_prior_eh SMARTPRIOR_BOX=ernest_henry \
  SMARTPRIOR_CELL_M=100 SMARTPRIOR_CELL_Z=50 \
  julia --project=. examples/export_cloncurry_blockmodel.jl

# Drillhole-group hold-out (district AABB, ~70/15/15 of named collars)
SMARTPRIOR_WORK=tmp_cloncurry_prior_district_holdout_w256_d4 \
  julia --project=. examples/holdout_cloncurry_prior.jl
```

Useful environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CLONCURRY_ROOT` | `~/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17` | METAL package root |
| `SMARTPRIOR_WORK` | `tmp_cloncurry_prior/` | checkpoint + report directory |
| `SMARTPRIOR_BOX` | `district` | `district` (sample AABB), `work`, or `ernest_henry` (grid D) |
| `SMARTPRIOR_CELL_M` | from target (district) / 1000 (work) | XY cell size (m) |
| `SMARTPRIOR_CELL_Z` | from target (district) / 100 (work) | vertical cell size (m) |
| `SMARTPRIOR_TARGET_CELLS` | 50000 | district spacing target |
| `SMARTPRIOR_EPOCHS` | 50 (train) / 250 (hold-out) | training budget |
| `SMARTPRIOR_WIDTH` | 256 | MLP width |
| `SMARTPRIOR_DEPTH` | 4 | MLP depth |

Train outputs in `SMARTPRIOR_WORK`: `cloncurry_prior.jld2`,
`cloncurry_prior_report.txt`, `cloncurry_prior_shares.tsv`.
Export also writes `cloncurry_smartprior_blockmodel.vts` and copies the
PNG to `docs/assets/cloncurry_eh_cu_iso.png`.

Tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.

---

## Repository layout

| Path | Role |
|---|---|
| `src/CloncurryIO.jl` | METAL readers (active pipeline) |
| `src/KeivitsaIO.jl` | GTK readers (previous case study) |
| `src/{Grid,Features,PriorNet,Losses,Train}.jl` | shared neural-field stack |
| `examples/train_cloncurry_prior.jl` | train (full-data) |
| `examples/export_cloncurry_blockmodel.jl` | VTK + 3D Cu isosurface from a checkpoint |
| `examples/holdout_cloncurry_prior.jl` | drillhole-group hold-out (district default) |
| `examples/train_keivitsa_prior.jl` | previous case study |
| `tmp_cloncurry_prior/` | work-box health check (gitignored) |
| `tmp_cloncurry_prior_eh/` | grid D, 250 epoch full-data (gitignored) |
| `tmp_cloncurry_prior_eh_group_holdout/` | EH drillhole hold-out, width 256 / depth 4 |
| `tmp_cloncurry_prior_eh_group_holdout_w64_d2/` | same split, width 64 / depth 2 |
| `tmp_keivitsa_prior*/` | previous diagnostic runs (gitignored) |
| `archive/mt_gravity_tmp_outputs/` | archived gravity+MT / Musgrave runs |
| `examples/compare_prior_2d.jl`, `examples/musgrave_*.jl` | legacy gravity+MT examples |

---

## Data licensing note

Cloncurry METAL is CC BY 4.0 (Austin et al., 2024). Gravity / magnetic /
MT products in the same folder have their own attributions; this pipeline
does not open them.

Keivitsa originates from GTK's open data release. A parallel look at
Ontario Geological Survey drillhole / specific-gravity / susceptibility
databases found them downloadable, but Ontario's Terms of Use restrict
**"substantial reproduction"** without prior written permission. Do not
assume that data is open-for-training.

---

## Open questions (Cloncurry)

1. **District hold-out** — the driver is retargeted to the sample AABB
   (~70/15/15 of 121 named collars). Until that run exists, the leak-free
   numbers below are Ernest Henry only (11 holes).
2. **Petrophysics heads lose to naive on EH** — density, susceptibility,
   and `conductivity_100kHz` lose at both 256/4 and 64/2. That is not a
   capacity result; 7 train holes may not carry those heads. Not tested
   on the district split.
3. **`conductivity_100kHz` zeros** — 711 of 1,250 finite values are exact 0
   and are lifted to 0.01 S/m before log10.
4. **No gradient-level loss decomposition** — pay tables are value shares
   only.

---

## Ernest Henry, grid D (250 epoch, full-data)

Width 256, depth 4, σ floor = 1× group std. Wall 1,322 s.
Single seed. From `tmp_cloncurry_prior_eh/` (gitignored).

| | Health check H (50 ep) | **D, 250 epoch** | Keivitsa 250 ep (for scale, not a ranking) |
|---|---:|---:|---:|
| Box | work ~42×76 km | Ernest Henry ~2.4×7.3×1.65 km | Keivitsa 25 m |
| Cells | 57,456 (1000/100 m) | **57,816 (100/50 m)** | 23,800 |
| Grade-anchor cells | 48 | **104** | **988** |
| Mapped Cu samples | — | 255 / 1,185 | drill assays |
| log10-RMSE | 0.215 | **0.005** | **0.249** |
| Pred high/bg | — | **19.80×** (assay 19.98×) | **5.81×** (assay 6.72×) |
| NLL grade / density / susc / cond | 0.015 / 0.039 / 0.012 / 0.028 | 0.00025 / 0.00039 / 0.00044 / 0.00044 | — |

**Do not cite D as beating Keivitsa.** 0.005 is the fit to 104 cell-mean
assays with a 256-wide net; predicted contrast 19.80× sits on the assay
ratio 19.98×. Keivitsa's 0.249 is 988 cells and 5.81× still under 6.72×
assay. Grid occupancy on D is 104 / 57,816 = 0.18%. Only 8 cells are
≥ 5,000 ppm Cu.

A 50-epoch work-box health check (`tmp_cloncurry_prior/`) is not a result:
1000 m cells stacked 347 Cu samples into 48 cells.

Predicted Cu on D, nested 2,000 / 5,000 ppm shells on a **cell-centred**
block model (corners = grid edges, so METAL sample XYZ sits inside the
block, not half a cell off a centre-point mesh). Drill traces coloured by
assay log10 Cu; distant EH holes MMA002 and MMA003 are omitted from the
frame. **In-sample** (104 grade-anchor cells); not a hold-out. The
predicted field has 96 cells ≥ 5,000 ppm (max 16,003 ppm) even though
only 8 of those 104 anchors sit at ≥ 5,000 ppm.

![Ernest Henry D — predicted Cu isosurface](docs/assets/cloncurry_eh_cu_iso.png)

Rebuild: `examples/export_cloncurry_blockmodel.jl` (writes `.vts` + PNG).

---

## Leak-free hold-out (Ernest Henry drillholes)

`examples/holdout_cloncurry_prior.jl` used to hide labels by a 300 m
sample buffer; that left **0** training points on Ernest Henry (median
nearest-neighbour 9.5 m). Whole-collar assignment replaced it.

191 / 45 / 19 are **Cu sample counts**, not hole counts. Same collar
split for both capacity runs (`split_seed=2026`):

| | train | val | test |
|---|---|---|---|
| Cu samples | 191 | 45 | 19 |
| holes | 7 | 2 | 2 |
| collars | EH242, EH550, EH591, EH691, EH699, EHMT001, MMA003 | EH435, EH632 | EH147, MMA002 |

Train labels come from train holes only, for all four properties. pXRF /
lithology / coverage still use all samples (Cu is never a feature). Val
is reported, not trained on. Naive reference: train-mean RMSE on the test
holes for grade / density / susceptibility; conductivity floor (−2) for
`conductivity_100kHz`. Single seed.

| property | 256/4 test RMSE | 64/2 test RMSE | naive | 256/4 beats naive? |
|---|---:|---:|---:|---|
| grade (log10 Cu) | **0.697** | 1.114 | 0.908 | **yes** |
| density | 0.202 | 0.267 | 0.187 | no |
| susceptibility | 0.532 | 0.833 | 0.421 | no |
| `conductivity_100kHz` | 0.276 | 0.416 | 0.228 | no |

256/4: best epoch 200 (run stopped after epoch 225; loss had already
risen). Train grade RMSE 0.652, val 0.978. `conductivity_100kHz` moved
off the −2 floor on the grid (10,292 distinct values at 4 d.p.; 0.27% of
cells at the floor).

64/2: 250 epoch, wall 303 s, same split. All four heads lose to naive,
including grade.

D's in-sample 0.005 was memorization; the leak-free grade number on this
deposit is **0.697**. Shrinking the net did not rescue the petrophysics
heads — 7 train holes do not carry them. These numbers are Ernest Henry
only; they are not a district result.

Reproduce the 256/4 numbers from
`tmp_cloncurry_prior_eh_group_holdout/cloncurry_holdout_report.txt`
(checkpoint `cloncurry_holdout_w256.jld2`). The 64/2 report is
`tmp_cloncurry_prior_eh_group_holdout_w64_d2/cloncurry_holdout_report.txt`.

The hold-out *driver* now defaults to the district sample AABB
(~70/15/15 of named collars). That run has not been completed; do not
treat the table above as a district score.

---

## Previous case study (Keivitsa)

The numbers below are **Keivitsa**, not Cloncurry. Same 971 + 16,215-point
geochemistry input unless noted. Grid: 50×34×N, 25 m in X/Y, EPSG:2393
(KKJ Finland Zone 3). From `tmp_keivitsa_prior*` (gitignored diagnostic
outputs).

### 1) Training budget (epochs)

25 m Z, width 128, depth 4, resistivity included, **no** σ floor.

| | 30 epoch (`tmp_keivitsa_prior`) | 300 epoch, best = 250 (`tmp_keivitsa_prior_e300`) |
|---|---:|---:|
| log10-RMSE (988 grade anchors) | 0.460 | **0.249** |
| High-grade pred/background | 2.65× | **5.81×** |
| Assay high/background (reference) | 6.72× | 6.72× |
| Max predicted Cu (ppm) | 3,831 | 16,751 |
| Cells ≥ 5,000 ppm | 0 | 621 |

More training closed most of the amplitude-compression gap. Loss ticked
back up between epoch 250 and 300 (density NLL 0.191 → 0.314);
`train_prior` keeps the best checkpoint.

### 2) Z-axis grid resolution

Hypothesis: 10 m Z instead of 25 m would connect the fragmented isosurface.

| | 25 m Z, ep. 250 (`e300` checkpoint) | 10 m Z, ep. 250 (`tmp_keivitsa_prior_z10`) |
|---|---:|---:|
| Cells | 23,800 | 57,800 |
| log10-RMSE | 0.249 | 0.291 |
| High-grade pred/bg | 5.81× | 5.86× |
| Fragmented 5,000 ppm shell? | Yes | **Still yes** |

**Hypothesis rejected.** Finer Z did not connect the fragments and did not
improve RMSE.

### 3) Sigma floor

Without a floor, density `σ` collapsed below the group std and NLL went
negative. Current training code floors `σ ≥ 1 ×` group std per property
(`sigma_bounds_from_anchors`).

| | No floor (e300, best ep. 250) | With floor (`tmp_keivitsa_prior_sigmafloor`, best ep. 200) |
|---|---:|---:|
| Total NLL | −5.12 | **+0.66** |
| Anchors sitting at the floor | — | 87–100% across properties |

Fixes the negative-NLL artifact. `σ` then loses most cell-to-cell range.

> Value share is not gradient share. Tables above are loss *values*, not
> which term is driving updates.

### 4) Network capacity

25 m Z, 250 epoch, σ floor on.

| width | depth | log10-RMSE | pred/bg | wall | run dir |
|---:|---:|---:|---:|---:|---|
| 128 | 4 | 0.287 | 5.55× | — | `tmp_keivitsa_prior_sigmafloor` |
| **256** | 4 | **0.232** | 5.78× | 395 s | `tmp_keivitsa_prior_wide` |
| 128 | 6 | 0.281 | 5.72× | 312 s | `tmp_keivitsa_prior_deep` |

Width helped RMSE; it did **not** move high-grade contrast (~5.5–5.8× vs
assay 6.72×).

Keivitsa figures remain under `docs/assets/` (plan views and isosurfaces
from `examples/export_keivitsa_blockmodel.jl`).

---

## License

MIT. See `LICENSE`.
