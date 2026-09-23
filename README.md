# SmartPrior

A Julia / Lux.jl neural-field method that predicts rock properties — Cu grade, density, magnetic susceptibility — at any 3D location from sparse drillhole samples, together with a per-location uncertainty. A block model is obtained by evaluating the field at block locations.

The method is site-agnostic: a site enters only through a data adapter and a TOML config. **Primary site:** Keivitsa (GTK). **Secondary:** Cloncurry district (METAL; sparse-regime reference). Gravity, magnetics, and MT are not inputs.

The MT-era package (SmartPriorMT) is archived as git tag `v0-mt-archive`.

## Status

| | |
|---|---|
| Primary site | **Keivitsa** (GTK; ~289 Cu holes, ~48 m median collar spacing) |
| Phase 1 data layer | **Done** — `SampleTable`, minimum-curvature desurvey, `load_site` for Cloncurry and Keivitsa |
| Next milestone | **Feasibility experiment on Keivitsa** (same LOHO protocol as Cloncurry) |
| Real-data feasibility (Cloncurry, 4 deposits) | Done. No method — kriging, IDW or the neural field — beats a constant mean on held-out drillholes. Holes are 100–370 m apart, 4–11 per deposit. See [`docs/2026-09_cloncurry_feasibility_report.md`](docs/2026-09_cloncurry_feasibility_report.md). |
| Semi-synthetic benchmark | In progress. Known 3D fields sampled at Cloncurry's real sample locations, to measure when prediction becomes possible. |
| Multi-output heads, censored likelihood, block averaging | Planned (Phase 2). |

### Legacy path (unchanged on purpose)

`Grid.jl` / `Features.jl` / `PriorNet.jl` and `examples/train_cloncurry_prior.jl`,
`examples/holdout_cloncurry_prior.jl` still build **geochemistry, lithology, and
sample-distance channels from every specimen**, including held-out holes (only
labels were hidden). Do not cite those RMSE tables as leak-free hold-out. Phase 1
evaluation uses `Covariates.jl`, site tables, and `examples/feasibility_loho.jl`.

The same scripts are documented under **Legacy path** in [Repository layout](#repository-layout) below.

## Architecture

```
 INPUTS                         NETWORK                               OUTPUTS
 ──────                         ───────                               ───────
 query point (x, y, z)  ──►  Fourier encoding of xyz  ──┐
                                                         ├──►  MLP  ──►  μ(x)  predicted value
 covariates at (x, y, z) ─────────────────────────────── ┘              σ(x)  uncertainty
 (depth, structure distance,                                  × 5-member ensemble
  surface geology)
```

### Inputs

The network only sees quantities that are **known at every location**, so the same function can be evaluated at a drillhole sample or at an arbitrary block.

| Input | What it is |
|---|---|
| Coordinates | x, y, z normalised to the site box |
| Depth | log depth below the surface |
| Structure distance | distance to mapped faults and contacts |
| Surface geology | one-hot surface rock class at (x, y) |

Deliberately **not** inputs:

- **pXRF geochemistry and drillhole lithology** exist only at samples. At a held-out hole they would leak that hole's own measurements, and at an undrilled block they do not exist. They are candidates for *outputs*.
- **Sample distance** is used only as a confidence mask for display.

Every covariate is a pure, point-wise function. Its normalisation statistics are fixed when the covariate is built, so `evaluate(c, xyz)` does not depend on which other points are queried.

### Network

| Stage | Setting |
|---|---|
| Encoding | xyz plus sin/cos(π·s·x) for 16 log-spaced scales s ∈ [1, 16]; other covariates appended unencoded |
| Body | MLP, 3 hidden layers × 64 units, GELU |
| Head | linear → (μ, σ), σ = softplus + 10⁻³ |
| Ensemble | 5 members with different seeds; predictive variance = mean σ² + variance of member means |

Currently one network is trained per property. A shared network with one head per property is planned.

### Training

- **Loss:** heteroscedastic Gaussian negative log-likelihood on sample points. There is no grid during training.
- **Optimiser:** AdamW (lr 10⁻³, weight decay 10⁻⁴), full batch.
- **Early stopping:** on held-out *training holes* (20 %); a test hole never influences training, standardisation or stopping.
- **Target scaling:** standardised with the training fold's mean and SD.
- **Censored values** (below detection limit) are carried in the data with a flag. They are currently set to the limit; a censored likelihood is planned.

### Outputs

| Property | Unit | Per location |
|---|---|---|
| Cu grade | log10 ppm | μ, σ |
| Density | g/cm³ | μ, σ |
| Magnetic susceptibility | log10 SI | μ, σ |

Derived quantities (planned): exceedance probability P(Cu > cutoff), block averages over sub-block points, tonnage.

## Evaluation

Every result is compared under the same folds with:

- **mean** — training mean;
- **IDW** — inverse-distance weighting;
- **ordinary kriging** — GeoStats.jl, variogram refitted per fold on training holes only.

Cross-validation is grouped by drillhole (leave-one-hole-out). Metrics are RMSE, R², skill relative to the mean, and coverage of the 90 % interval.

A method is considered useful only if it clearly beats the mean *and* is comparable to or better than kriging.

## Data

Data are not in this repository.

**Keivitsa (GTK)** — set `KEIVITSA_ROOT` to the unpacked `database/keivitsa/source/gtk`
tree (or `root` in `sites/keivitsa.toml`). Raw caret tables (`kalte.txt`, `511P.txt`,
`petro.txt`, …) and shapefiles stay outside git. **GTK basic licence:** internal use
and figures in scientific publications only; **do not commit or redistribute** GTK
data or derived exports. Synthetic layout-only fixtures:
`test/fixtures/keivitsa_tiny/`. Inspection output: `tmp_keivitsa_inspect/` (gitignored).

**Cloncurry (METAL)** — `CLONCURRY_ROOT` (default
`~/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17`, CC BY 4.0, Austin et al. 2024):

```
├── derived/          # petrophysics_samples.csv, metal_all_fields.csv.gz
└── geology/          # structures.geojson, surface_geology.geojson
```

`mt/`, `gravity/`, and `magnetics/` are not opened. Fixtures: `test/fixtures/cloncurry_*`.

Site files in `sites/` (`keivitsa.toml`, `ernest_henry.toml`, `cannington.toml`, `starra.toml`, `osborne.toml`) give each box, CRS, properties and covariates.

Convention notes (no raw GTK rows): [`docs/keivitsa_data_notes.md`](docs/keivitsa_data_notes.md).

Cloncurry detection-limit policy (district, before deposit filter): censoring limit = 1st percentile of positive values → 5.8 ppm Cu and 0.0356 S/m for conductivity. Keivitsa Cu uses a fixed 1 ppm limit in `sites/keivitsa.toml`.

## Repository layout

| Path | Role |
|---|---|
| `src/SmartPrior.jl` | module entry |
| `src/{Schema,Sites,Covariates,Desurvey}.jl` | Phase 1 site schema, covariates, desurvey |
| `src/{CloncurryIO,KeivitsaIO}.jl` | METAL / GTK → `SampleTable` |
| `src/{Grid,Features,PriorNet,Losses,Train,Metrics}.jl` | legacy grid neural-field stack |
| `src/SyntheticFields.jl` | seedable Gaussian fields (Experiment 1) |
| `sites/{keivitsa,ernest_henry,starra,cannington,osborne}.toml` | site configuration |
| `examples/keivitsa_inspect.jl` | Keivitsa counts, azimuth check, GLMakie traces |
| `examples/feasibility_loho.jl` | leak-free Cloncurry LOHO feasibility |
| `examples/synthetic_exp1.jl` | semi-synthetic mean / kriging / nn_xyz benchmark |
| `examples/train_cloncurry_prior.jl` | legacy full-data train |
| `examples/holdout_cloncurry_prior.jl` | legacy drillhole-group hold-out |
| `examples/export_cloncurry_blockmodel.jl` | VTK + Cu figure (legacy checkpoints) |
| `examples/variogram_cloncurry.jl` | directional variogram |
| `examples/kriging_cloncurry_petro.jl` | ordinary-kriging baseline |
| `examples/kriging_env/` | isolated GeoStats.jl kriging env |
| `docs/keivitsa_data_notes.md` | GTK conventions (statistics only) |
| `docs/2026-09_cloncurry_feasibility_report.md` | Cloncurry LOHO report |
| `tmp_*/` | gitignored run outputs |

**Legacy path.** `src/{Grid,Features,PriorNet,Losses,Train,Metrics}.jl` and `examples/{train,holdout}_cloncurry_prior.jl` implement the earlier grid-based pipeline. It used test-hole pXRF as input, so its published numbers are not leak-free. It is kept only until Phase 2 replaces it, and should not be used for new results.

## How to run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'

# Keivitsa adapter checks (requires KEIVITSA_ROOT or sites/keivitsa.toml root)
julia --project=. examples/keivitsa_inspect.jl

# Real-data feasibility (writes tmp_feasibility/)
julia --project=. examples/feasibility_loho.jl
```

| Variable | Meaning |
|---|---|
| `KEIVITSA_ROOT` | GTK `source/gtk` tree (required for Keivitsa) |
| `CLONCURRY_ROOT` | METAL package root |

Legacy district train / hold-out / VTK export: `examples/train_cloncurry_prior.jl`, `examples/holdout_cloncurry_prior.jl`, `examples/export_cloncurry_blockmodel.jl` (see legacy-path note above).

## Roadmap

1. Semi-synthetic benchmark: hole geometry × correlation length (experiment 1), then multiple properties, lithology and censoring.
2. Keivitsa LOHO feasibility, then Phase 2: shared multi-output network, censored likelihood, block averaging.
3. Visualisation: GLMakie block viewer showing μ, σ and exceedance probability, faded by distance to data; VTK export.

## License

MIT. See `LICENSE`. Keivitsa data: GTK basic licence (no redistribution). Cloncurry METAL: CC BY 4.0. Other METAL geophysical products keep their own attributions.
