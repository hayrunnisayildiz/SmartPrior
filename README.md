# SmartPrior

A Julia / Lux.jl neural-field method that predicts rock properties — Cu grade, density, magnetic susceptibility — at any 3D location from sparse drillhole samples, together with a per-location uncertainty. A block model is obtained by evaluating the field at block locations.

The method is site-agnostic: a site enters only through a data adapter and a TOML config. The current case study is the Cloncurry district (Queensland).

## Status

| | |
|---|---|
| Real-data feasibility (Cloncurry, 4 deposits) | Done. No method — kriging, IDW or the neural field — beats a constant mean on held-out drillholes. Holes are 100–370 m apart, 4–11 per deposit. See [`docs/2026-09_cloncurry_feasibility_report.md`](docs/2026-09_cloncurry_feasibility_report.md). |
| Semi-synthetic benchmark | In progress. Known 3D fields sampled at Cloncurry's real sample locations, to measure when prediction becomes possible. |
| Multi-output heads, censored likelihood, block averaging | Planned (Phase 2). |

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

| Variable | Default |
|---|---|
| `CLONCURRY_ROOT` | `~/Desktop/datasets4HY/Cloncurry_integrated_2026-09-17` (METAL package, CC BY 4.0, Austin et al. 2024) |

Each deposit has a site file in `sites/` (`ernest_henry.toml`, `cannington.toml`, `starra.toml`, `osborne.toml`) giving its box, CRS (EPSG:28354), properties and covariates.

Detection-limit policy, applied to the district before any deposit filter: the censoring limit is the 1st percentile of positive values. This gives 5.8 ppm for Cu and 0.0356 S/m for conductivity.

## Repository layout

| Path | Role |
|---|---|
| `src/Schema.jl` | `SampleTable`, `PropertySpec`, `training_mask` |
| `src/Covariates.jl` | point-wise covariates |
| `src/Sites.jl` | `load_site` and TOML site configs |
| `src/CloncurryIO.jl` | Cloncurry adapter → `SampleTable` |
| `src/SyntheticFields.jl` | continuous Gaussian random fields (in progress) |
| `examples/feasibility_loho.jl` | real-data leave-one-hole-out comparison |
| `examples/synthetic_exp1.jl` | semi-synthetic benchmark, experiment 1 (in progress) |
| `docs/` | reports |

**Legacy path.** `src/{Grid,Features,PriorNet,Losses,Train,Metrics}.jl` and `examples/{train,holdout}_cloncurry_prior.jl` implement the earlier grid-based pipeline. It used test-hole pXRF as input, so its published numbers are not leak-free. It is kept only until Phase 2 replaces it, and should not be used for new results.

The MT-era package is archived as tag `v0-mt-archive`.

## How to run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'

# Real-data feasibility (writes tmp_feasibility/)
julia --project=. examples/feasibility_loho.jl
```

## Roadmap

1. Semi-synthetic benchmark: hole geometry × correlation length (experiment 1), then multiple properties, lithology and censoring.
2. Phase 2: shared multi-output network, censored likelihood, block averaging.
3. Visualisation: GLMakie block viewer showing μ, σ and exceedance probability, faded by distance to data; VTK export.

## License

MIT. See `LICENSE`. Data retain their own licences.
