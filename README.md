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
| Grid in use | **D — Ernest Henry only**, 100 m XY / 50 m Z, 24×73×33 = 57,816 cells (`tmp_cloncurry_prior_eh/`) |
| Fourth head | **`conductivity_100kHz`** — KT-20, 100 kHz, specimen-scale. Not MT bulk conductivity; do not call it resistivity |
| Connected to VFSA? | **No.** Deferred — see [Relationship to VFSA](#relationship-to-vfsa) |
| Full-data D, 250 epoch | log10-RMSE **0.005** on **104** grade cells. **Not a hold-out.** Do not report as better than Keivitsa (0.249 on 988 cells) |
| Spatial hold-out | **Not run.** A 300 m sample-level buffer left **0** training samples (see below) |

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

`tmp_cloncurry_prior/` and `tmp_cloncurry_prior_eh/` are **output**
directories (checkpoints, reports), not the source data.

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

# Health check (work box, 1000 m × 100 m, 50 epoch)
julia --project=. examples/train_cloncurry_prior.jl

# Grid D — Ernest Henry, 100 m × 50 m, 250 epoch
SMARTPRIOR_WORK=tmp_cloncurry_prior_eh SMARTPRIOR_BOX=ernest_henry \
  SMARTPRIOR_CELL_M=100 SMARTPRIOR_CELL_Z=50 SMARTPRIOR_EPOCHS=250 \
  julia --project=. examples/train_cloncurry_prior.jl
```

Useful environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CLONCURRY_ROOT` | `~/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17` | METAL package root |
| `SMARTPRIOR_WORK` | `tmp_cloncurry_prior/` | checkpoint + report directory |
| `SMARTPRIOR_BOX` | `work` | `work` (district box) or `ernest_henry` (grid D) |
| `SMARTPRIOR_CELL_M` | 1000 | XY cell size (m) |
| `SMARTPRIOR_CELL_Z` | 100 | vertical cell size (m) |
| `SMARTPRIOR_EPOCHS` | 50 | training budget |
| `SMARTPRIOR_WIDTH` | 256 | MLP width |
| `SMARTPRIOR_DEPTH` | 4 | MLP depth |

Outputs in `SMARTPRIOR_WORK`: `cloncurry_prior.jld2`,
`cloncurry_prior_report.txt`, `cloncurry_prior_shares.tsv`.

A spatial Cu hold-out driver exists at
`examples/holdout_cloncurry_prior.jl` but has **not** completed a 250-epoch
run (see [Hold-out attempt](#hold-out-attempt-not-a-result)).

Tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.

---

## Repository layout

| Path | Role |
|---|---|
| `src/CloncurryIO.jl` | METAL readers (active pipeline) |
| `src/KeivitsaIO.jl` | GTK readers (previous case study) |
| `src/{Grid,Features,PriorNet,Losses,Train}.jl` | shared neural-field stack |
| `examples/train_cloncurry_prior.jl` | train (full-data) |
| `examples/holdout_cloncurry_prior.jl` | spatial Cu hold-out (not yet a completed run) |
| `examples/train_keivitsa_prior.jl` | previous case study |
| `tmp_cloncurry_prior/` | work-box health check (gitignored) |
| `tmp_cloncurry_prior_eh/` | grid D, 250 epoch (gitignored) |
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

1. **Hold-out of D's 0.005** — sample-level 300 m buffer is infeasible on
   this deposit (see below). A cluster-level split is the next test; until
   it exists, D is in-sample fit only.
2. **District vs deposit box** — 1,590 loaded; D uses 269 EH samples
   (255 Cu). Option C (sample hull, 400/100 m) would mix other deposits,
   not add EH samples.
3. **conductivity_100kHz zeros** — 711 of 1,250 finite values are exact 0
   and are lifted to 0.01 S/m before log10.
4. **No gradient-level loss decomposition** — pay tables are value shares
   only.

---

## Ernest Henry, grid D (250 epoch)

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

---

## Hold-out attempt (not a result)

`examples/holdout_cloncurry_prior.jl` hides Cu labels on a spatial split
(`spatial_holdout` in `src/CloncurryIO.jl`). Grade NLL sees the train
split only; pXRF / lithology / coverage still use all samples (Cu is
never a feature).

A **300 m sample-level** buffer on the 255 in-grid Cu samples left
**0 training points** (18 test, 237 inside the buffer). Median
nearest-neighbour distance on Ernest Henry is **9.5 m** (XY 2.7 m) —
specimens sit along holes, not on a 300 m lattice. The same 300 m rule
applied to **50 m-linked clusters** would leave roughly 191 train / 45
test / 19 dropped; that run has not been started.

Until a hold-out RMSE exists, treat 0.005 as memorisation risk, not
generalisation.

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
