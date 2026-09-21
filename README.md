# SmartPriorMT — Cloncurry geochemistry + petrophysics prior

**One line:** a Julia/Lux.jl neural field that predicts ore grade, density,
magnetic susceptibility, and rock-specimen conductivity on every cell of a 3D
block model from sparse pXRF geochemistry, drillhole lithology, sample
coverage, and (in code) structural geology — **no gravity, no magnetics, no MT
as input**.

Active case study: **Cloncurry district** (Queensland; METAL package
`Cloncurry_integrated_2026-09-17`). Not a claim that the same weights transfer
to other deposits.

Keivitsa (GTK) was an earlier case study; see
[README_KEIVITSA_LEGACY.md](README_KEIVITSA_LEGACY.md). An older gravity + MT
example line remains under `examples/compare_prior_2d.jl` / `musgrave_*.jl` for
archive only — **out of scope** for this pipeline (not deferred, not connected).

---

## Status

| Question | Answer |
|---|---|
| Active dataset | Cloncurry district sample AABB (~118 × 219 km), 120 named holes |
| Fourth head | **`conductivity_100kHz`** — KT-20, 100 kHz, specimen-scale. **Not** MT bulk conductivity |
| Leak-free result | District drillhole hold-out 84/18/18, `split_seed=2026`, width=256 / depth=4 |
| Density vs naive | Still loses (see [Results](#results--district-drillhole-hold-out)) |
| Structural geology | In `src/CloncurryIO.jl` (+16 channels → 69 total); district retrain with geology **not finished** (no report yet) |

---

## Why district (not Ernest Henry alone)

Ernest Henry has only **11** named holes. A leak-free 7/2/2 hole split showed
grade with a weak signal and density / susceptibility / `conductivity_100kHz`
all losing to naive
(`tmp_cloncurry_prior_eh_group_holdout/cloncurry_holdout_report.txt`). Shrinking
the net (64/2) did not rescue those heads — capacity was not the bottleneck.

The active grid is therefore the **district sample AABB** covering every
finite-xyz METAL row (~1,586 samples / **120** holes), with whole-collar
assignment ~70/15/15 → **84 / 18 / 18** holes (`split_seed=2026`).

---

## Data (not in this repository)

```
CLONCURRY_ROOT  (default ~/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17)
├── derived/
│   ├── petrophysics_samples.csv
│   └── metal_all_fields.csv.gz
└── geology/
    ├── structures.geojson      # LineStrings (fault / contact / …)
    └── surface_geology.geojson # Polygons (dom_rock, rock_type)
```

`mt/`, `gravity/`, and `magnetics/` sit in the same package and are **not
opened**. Magnetics as an input channel is deliberately withheld pending a
separate go-ahead. Tiny fixtures: `test/fixtures/cloncurry_*`.

---

## Inputs (feature channels)

Built by `examples/holdout_cloncurry_prior.jl` /
`examples/train_cloncurry_prior.jl` via `src/CloncurryIO.jl` + `src/Features.jl`.
`Cu_Concentration` is the grade **target**, never a feature.

### Live district stack without geology — **53 channels**

From `tmp_cloncurry_prior_eh/cloncurry_prior_report.txt` `feature_names` and
the same builder (district AABB run used the same feature recipe before
geology was appended):

| Group | Count | Names / notes |
|---|---:|---|
| Coordinates | 3 | `x_norm`, `y_norm`, `z_norm` (→ Fourier `n_bands=4`) |
| Depth | 2 | `log_depth`, `depth_over_skin` |
| Geochemistry | **32** | `geochem_{Mg…U}` except Cu. Code lists 33 elements in `CLONCURRY_GEOCHEM_ELEMENTS`; **Co** is dropped when no finite positive ppm remain (`nearest_sample_channels`) |
| Drillhole lithology | 15 | `lith_AMP` … `lith_PSM`, `lith_OTHER` (one-hot; rare → OTHER) |
| Coverage | 1 | `sample_distance` |

### Structural geology (in code; +16 → **69 channels**)

From a live channel dump (`structure_distance_channels` /
`surface_geology_channels` on the district grid):

| Group | Count | Names |
|---|---:|---|
| Structure distance | 2 | `struct_fault_distance`, `struct_line_distance` (log1p distance / median cell, GDA94/MGA54 m) |
| Surface `dom_rock` | 11 | `surf_dom_ARENITE_RUDITE` … `surf_dom_QUARTZITE`, `surf_dom_OTHER` |
| Surface `rock_type` | 3 | `surf_rock_INTRUSIVE_UNIT`, `surf_rock_STRATIFIED_UNIT_INCLUDING_VOLCANIC_AND_METAMORPHIC`, `surf_rock_OTHER` |

Surface one-hots sit **beside** drillhole `lith_*`, not instead of them.
Fold / Layering are not separate channels (few lines; they only feed
`struct_line_distance`).

The completed district hold-out numbers below are still the **53-channel**
run. Geology-augmented retrain:
`tmp_cloncurry_prior_district_holdout_w256_d4_geology/` (report pending).

---

## Outputs

Eight heads (`Dense(width => 8)`): four properties × (`μ`, `σ`).
Heteroscedastic NLL with per-property σ floor = `1 ×` group std of train
anchors.

| Property | NLL units | Source column |
|---|---|---|
| `grade` | log10 Cu ppm | `Cu_Concentration` (`<LOD` → 1 ppm) |
| `density` | g/cm³ | `density_mean_g_cm3` |
| `susceptibility` | log10 SI | `susceptibility_mean_SI` (>0) |
| `conductivity_100kHz` | log10 S/m | `conductivity_mean_S_m_100kHz` (zeros → 0.01) |

**`conductivity_100kHz` is intentional naming.** KT-20 at 100 kHz on small
core specimens (Austin et al. 2024, §5.2.4). It is **not** MT-equivalent bulk
conductivity and must not be called resistivity or treated as an MT start
model. 711 / 1,250 finite values are exact 0 and are lifted to 0.01 S/m before
log10 (`CLONCURRY_COND_FLOOR_S_M`).

Architecture default for district hold-out: **width = 256**, **depth = 4**,
250 epochs.

---

## Methodology — memorization vs hold-out

| Run | What it is | Cite as success? |
|---|---|---|
| `tmp_cloncurry_prior_eh/` | Ernest Henry box, **all** labels, 104 grade cells | **No** — in-sample fit (log10-RMSE **0.005**) |
| `tmp_cloncurry_prior_eh_group_holdout/` | EH 7/2/2 holes | Diagnostic only — showed EH is too thin |
| `tmp_cloncurry_prior_district_holdout_w256_d4/` | District **84/18/18** holes, `split_seed=2026` | **Yes** — leak-free district score |
| `…_w256_d4_geology/` | Same split + geology channels | Not ready |

Naive reference on district: train-mean RMSE on test holes for grade /
density / susceptibility; conductivity **floor** (−2 = log10 0.01 S/m) for
`conductivity_100kHz` (see report `naive` line).

> Value share is not gradient share. Loss-share tables (if printed) are
> term *values*, not which term drives parameter updates.

---

## Results — district drillhole hold-out

Source:
`tmp_cloncurry_prior_district_holdout_w256_d4/cloncurry_holdout_report.txt`
(53 channels, no structural geology; best epoch 200; width=256, depth=4).

Grid: sample AABB, (52, 96, 11) = 54,912 cells @ 2300 m × 200 m.
Holes 84 / 18 / 18; samples 1,105 / 251 / 230.

| property | test RMSE | naive | beats naive? |
|---|---:|---:|---|
| grade (log10 Cu) | **1.309** | 1.339 | **yes** |
| density | 0.606 | 0.469 | **no** |
| susceptibility | **1.364** | 1.612 | **yes** |
| `conductivity_100kHz` | **1.006** | 1.393 (floor) | **yes** |

Notes from the same report: conductivity train-mean baseline on test is
**0.953** — the net beats the floor but not train-mean. Grid μ is off the
floor (`cond_moved_off_floor=true`). **Density still loses to naive.**

### Earlier EH-only hold-out (not the district score)

`tmp_cloncurry_prior_eh_group_holdout/cloncurry_holdout_report.txt` —
7/2/2 holes, test RMSE grade **0.697** (naive 0.908); density / susc /
cond all lose. Explains the move to district.

### In-sample EH figure (memorization — do not cite as hold-out)

From `tmp_cloncurry_prior_eh/cloncurry_prior_report.txt`: log10-RMSE
**0.005** on 104 grade cells; predicted high/bg **19.80×** vs assay
**19.98×**. Export: `examples/export_cloncurry_blockmodel.jl` →
`docs/assets/cloncurry_eh_cu_iso.png`.

![Ernest Henry D — predicted Cu isosurface (in-sample)](docs/assets/cloncurry_eh_cu_iso.png)

Nested 2,000 / 5,000 ppm shells on a cell-centred block model; drill traces
coloured by assay log10 Cu. **In-sample only.**

---

## How to run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Full-data train (default = district sample AABB)
julia --project=. examples/train_cloncurry_prior.jl

# District drillhole hold-out (84/18/18 of 120 named holes)
SMARTPRIOR_WORK=tmp_cloncurry_prior_district_holdout_w256_d4 \
  julia --project=. examples/holdout_cloncurry_prior.jl

# VTK + Cu isosurface from an EH checkpoint
SMARTPRIOR_WORK=tmp_cloncurry_prior_eh SMARTPRIOR_BOX=ernest_henry \
  SMARTPRIOR_CELL_M=100 SMARTPRIOR_CELL_Z=50 \
  julia --project=. examples/export_cloncurry_blockmodel.jl
```

| Variable | Default | Meaning |
|---|---|---|
| `CLONCURRY_ROOT` | `~/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17` | METAL package |
| `SMARTPRIOR_WORK` | `tmp_cloncurry_prior*` | checkpoints + reports |
| `SMARTPRIOR_BOX` | `district` | `district` / `work` / `ernest_henry` |
| `SMARTPRIOR_WIDTH` / `DEPTH` | 256 / 4 | MLP size |
| `SMARTPRIOR_EPOCHS` | 50 (train) / 250 (hold-out) | budget |
| `SMARTPRIOR_SPLIT_SEED` | 2026 | collar split |

Tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.

---

## Repository layout

| Path | Role |
|---|---|
| `src/CloncurryIO.jl` | METAL + geology readers (active) |
| `src/{Grid,Features,PriorNet,Losses,Train}.jl` | neural-field stack |
| `examples/train_cloncurry_prior.jl` | full-data train |
| `examples/holdout_cloncurry_prior.jl` | drillhole-group hold-out |
| `examples/export_cloncurry_blockmodel.jl` | VTK + Cu isosurface |
| `src/KeivitsaIO.jl` | legacy GTK readers |
| `README_KEIVITSA_LEGACY.md` | Keivitsa experiment tables |
| `tmp_cloncurry_prior*/` | gitignored run outputs |

---

## Open questions

1. **Density** — on the completed district hold-out still loses to naive
   (0.606 vs 0.469). Whether structural geology closes that gap is **not
   known yet** (geology retrain unfinished).
2. **Magnetics** — present in the METAL package; not wired. Waiting on
   explicit approval before adding channels.
3. **`conductivity_100kHz`** — beats the −2 floor on district test, but
   loses to train-mean (0.953). Borderline; zeros→floor and specimen vs bulk
   remain caveats.
4. **Geology hold-out report** — code path live (69 channels); score file
   not written yet under `tmp_cloncurry_prior_district_holdout_w256_d4_geology/`.

---

## Data licensing

Cloncurry METAL is CC BY 4.0 (Austin et al., 2024). Gravity / magnetic / MT
products in the same folder have their own attributions; this pipeline does
not open them.

---

## License

MIT. See `LICENSE`.
