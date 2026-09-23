# SmartPrior — Cloncurry geochemistry + petrophysics prior

**One line:** a Julia/Lux.jl neural field that predicts ore grade, density,
magnetic susceptibility, and rock-specimen conductivity on every cell of a 3D
block model from sparse pXRF geochemistry, drillhole lithology, sample
coverage, and structural geology — **no gravity, no magnetics, no MT
as input**.

Active case study: **Cloncurry district** (Queensland; METAL package
`Cloncurry_integrated_2026-09-17`). Not a claim that the same weights transfer
to other deposits.

The MT-era package (SmartPriorMT) is archived as git tag `v0-mt-archive`.

---

## Status

| Question | Answer |
|---|---|
| Active dataset | Cloncurry district sample AABB (~118 × 219 km), 120 named holes |
| Fourth head | **`conductivity_100kHz`** — KT-20, 100 kHz, specimen-scale. **Not** MT bulk conductivity |
| Leak-free result | District drillhole hold-out 84/18/18, `split_seed=2026`, width=256 / depth=4 |
| Active mesh | **2300 m XY × 100 m Z** (default; refined from 200 m Z after variogram) |
| Density vs naive | Still loses after five independent interventions (see [Results](#results--district-drillhole-hold-out)) |
| Structural geology | Live: +16 channels → **69** total; scored under `…_w256_d4_geology/` |

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

| Group | Count | Names / notes |
|---|---:|---|
| Coordinates | 3 | `x_norm`, `y_norm`, `z_norm` (→ Fourier `n_bands=4`) |
| Depth | 2 | `log_depth`, `depth_over_skin` (geometric scale, not an MT input) |
| Geochemistry | **32** | `geochem_{Mg…U}` except Cu. **Co** is dropped when no finite positive ppm remain |
| Drillhole lithology | 15 | `lith_AMP` … `lith_PSM`, `lith_OTHER` (one-hot; rare → OTHER) |
| Coverage | 1 | `sample_distance` |

### Structural geology (+16 → **69 channels**)

| Group | Count | Names |
|---|---:|---|
| Structure distance | 2 | `struct_fault_distance`, `struct_line_distance` |
| Surface `dom_rock` | 11 | `surf_dom_ARENITE_RUDITE` … `surf_dom_QUARTZITE`, `surf_dom_OTHER` |
| Surface `rock_type` | 3 | `surf_rock_INTRUSIVE_UNIT`, `surf_rock_STRATIFIED_UNIT_INCLUDING_VOLCANIC_AND_METAMORPHIC`, `surf_rock_OTHER` |

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
conductivity. 711 / 1,250 finite values are exact 0 and are lifted to 0.01 S/m
before log10 (`CLONCURRY_COND_FLOOR_S_M`).

Architecture default for district hold-out: **width = 256**, **depth = 4**.

---

## Methodology — memorization vs hold-out

| Run | What it is | Cite as success? |
|---|---|---|
| `tmp_cloncurry_prior_eh/` | Ernest Henry box, **all** labels, 104 grade cells | **No** — in-sample fit |
| `tmp_cloncurry_prior_eh_group_holdout/` | EH 7/2/2 holes | Diagnostic only — showed EH is too thin |
| `tmp_cloncurry_prior_district_holdout_w256_d4/` | District **84/18/18**, 53 ch, Z=200 m | Leak-free; **before geology** |
| `…_w256_d4_geology/` | Same split + geology (69 ch), Z=200 m | Leak-free geology score |
| `…_w256_d4_z100/` | Same split + geology, **Z=100 m**, 100 epochs | **Current** district score |

Naive reference on district: train-mean RMSE on test holes for grade /
density / susceptibility; conductivity **floor** (−2 = log10 0.01 S/m) for
`conductivity_100kHz`.

---

## Results — district drillhole hold-out

All rows below use the same collar split: **84 / 18 / 18** holes,
`split_seed=2026`, samples 1,105 / 251 / 230, width=256 / depth=4.

### With structural geology (Z = 200 m)

Source:
`tmp_cloncurry_prior_district_holdout_w256_d4_geology/cloncurry_holdout_report.txt`

| property | test RMSE | naive | beats naive? |
|---|---:|---:|---|
| grade (log10 Cu) | **1.337** | 1.339 | **yes** (barely) |
| density | 0.590 | 0.469 | **no** |
| susceptibility | **1.341** | 1.612 | **yes** |
| `conductivity_100kHz` | **1.001** | 1.393 (floor) | **yes** |

### Before geology (same split, 53 channels, Z = 200 m)

| property | test RMSE | naive | beats naive? |
|---|---:|---:|---|
| grade (log10 Cu) | **1.309** | 1.339 | **yes** |
| density | 0.606 | 0.469 | **no** |
| susceptibility | **1.364** | 1.612 | **yes** |
| `conductivity_100kHz` | **1.006** | 1.393 (floor) | **yes** |

### Current score — Z = 100 m (100 epochs)

Source:
`tmp_cloncurry_prior_district_holdout_w256_d4_z100/cloncurry_holdout_report.txt`

| property | test RMSE | naive | beats naive? | vs geology Z=200 |
|---|---:|---:|---|---:|
| grade (log10 Cu) | **1.263** | 1.339 | **yes** | 1.337 → better |
| density | 0.600 | 0.469 | **no** | 0.590 → slightly worse |
| susceptibility | **1.340** | 1.612 | **yes** | 1.341 ≈ same |
| `conductivity_100kHz` | **0.914** | 1.393 (floor) | **yes** | 1.001 → better |

**Density — five interventions, none beat naive (0.469):**

| # | Intervention | density test RMSE |
|---|---|---:|
| 1 | Capacity ↑ (256/4) | 0.606 |
| 2 | Capacity ↓ (64/2, EH) | 0.267 (still > naive 0.187) |
| 3 | More holes (84 vs 7) | 0.606 |
| 4 | Structural geology | 0.590 |
| 5 | Z refine 200 → 100 m | 0.600 |

### Figures — district predicted Cu

![Cloncurry district Z=100 m — predicted Cu blocks + drill traces](docs/assets/cloncurry_district_z100_cu_iso.png)

![Cloncurry district geology (Z=200 m) — predicted Cu + drill traces](docs/assets/cloncurry_district_geology_cu_iso.png)

---

## How to run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Full-data train (default = district sample AABB)
julia --project=. examples/train_cloncurry_prior.jl

# District drillhole hold-out (84/18/18 of 120 named holes; default Z=100 m)
SMARTPRIOR_WORK=tmp_cloncurry_prior_district_holdout_w256_d4_z100 \
  julia --project=. examples/holdout_cloncurry_prior.jl

# Directional variogram / anisotropy (writes tmp_cloncurry_variogram/)
julia --project=. examples/variogram_cloncurry.jl

# VTK + Cu figure — current district Z=100 checkpoint
SMARTPRIOR_WORK=tmp_cloncurry_prior_district_holdout_w256_d4_z100 \
  SMARTPRIOR_BOX=district SMARTPRIOR_WIDTH=256 SMARTPRIOR_DEPTH=4 \
  SMARTPRIOR_CELL_M=2300 SMARTPRIOR_CELL_Z=100 \
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
| `src/CloncurryIO.jl` | METAL + geology readers |
| `src/{Grid,Features,PriorNet,Losses,Train,Metrics}.jl` | neural-field stack |
| `examples/train_cloncurry_prior.jl` | full-data train |
| `examples/holdout_cloncurry_prior.jl` | drillhole-group hold-out |
| `examples/export_cloncurry_blockmodel.jl` | VTK (WriteVTK) + Cu figure (GLMakie) |
| `examples/variogram_cloncurry.jl` | directional variogram + anisotropy |
| `examples/kriging_cloncurry_petro.jl` | GeoStats ordinary-kriging baseline |
| `tmp_cloncurry_prior*/` | gitignored run outputs |

---

## Open questions

1. **Density** — still loses to naive after capacity ↑/↓, more holes,
   structural geology, and Z=100 m (current **0.600** vs **0.469**).
2. **Magnetics** — present in the METAL package; not wired. Waiting on
   explicit approval before adding channels.
3. **`conductivity_100kHz`** — beats the −2 floor on district test (Z=100:
   **0.914** vs floor 1.393). Zeros→floor and specimen vs bulk remain caveats.

---

## Data licensing

Cloncurry METAL is CC BY 4.0 (Austin et al., 2024). Gravity / magnetic / MT
products in the same folder have their own attributions; this pipeline does
not open them.

---

## License

MIT. See `LICENSE`.
