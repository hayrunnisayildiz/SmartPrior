# SmartPrior

A Julia / Lux.jl neural field that predicts a rock property, and its uncertainty, at any 3D point from sparse drillholes. It is tested against ordinary kriging on GTK drillhole data from the Keivitsa Cu–Ni deposit, Finland.

**Status:** research prototype. Copper and density runs are complete; susceptibility is supported by the scripts but has not been run yet.

![E–W cross-section of predicted log10 Cu (top) and its uncertainty σ (bottom), kriging on the left and the neural field on the right](docs/figures/section_mu_sigma.png)

*A vertical east–west slice through the deposit. **Top row:** predicted copper grade μ (log10 ppm; 2 = 100 ppm, 3 = 1,000 ppm). Circles are drill samples within 20 m of the slice, coloured by their measured grade. **Bottom row:** predicted uncertainty σ (log10 units; higher = less certain). Black marks show where those samples are. **How to read it:** kriging (left) produces a detailed μ but an almost uniform σ, so it reports the same confidence everywhere. The neural field (right) gives a smoother μ, and its σ is highest near the surface and around isolated holes, which is where predictions should be least certain.*

## Key results

Scores come from 10-fold cross-validation in which whole drillholes are held out. **Skill** is `1 − RMSE / RMSE_mean`: 0 means no better than predicting the training average, and higher is better. **Coverage** is the share of held-out samples that fall inside the method's 90 % interval; a well-calibrated method scores close to 0.90.

| Site / target | Holes | Samples | Kriging (v1) skill | Neural field skill | Kriging (v1) coverage | Neural field coverage |
|---|---:|---:|---:|---:|---:|---:|
| Keivitsa Cu (log10 ppm) | 261 | 15,859 | 0.106 | 0.151 (`nn_xyz`) | 0.78 | 0.87 |
| Keivitsa density (kg/m³) | 263 | 30,908 | 0.058 | 0.059 (`nn_xyz`), 0.065 (`nn_cov`) | 0.77 | 0.89 (`nn_xyz`), 0.90 (`nn_cov`) |

`nn_xyz` sees only the coordinates; `nn_cov` also sees depth below the ground surface.

**Copper.** The neural field is non-inferior to kriging (v1): all three pre-registered conditions pass. The difference in pooled skill is not statistically significant. Both methods under-cover, but kriging (v1) is more overconfident (0.78 vs 0.87).

**Density.** Coordinates carry little information about density: RMSE falls only from 146.1 to 137.5 kg/m³ compared with the training mean. The pre-registered rule is not met (`useful = 0`): skill is below 0.10, although its 95 % interval (0.003 … 0.114 for `nn_xyz`) is above 0. Hole by hole, the network beats kriging (v1) on 153 of 263 holes (mean paired skill difference +0.055, 95 % CI 0.016 … 0.093). The network's intervals are well calibrated (0.89–0.90); kriging (v1) is overconfident (0.77).

![Pooled Cu skill with 95 % confidence intervals for the mean, kriging and the two neural-field variants](docs/figures/skill_ci.png)

*Copper skill for each method. Dots are the pooled skill; bars are 95 % confidence intervals from resampling whole holes. The dotted line at 0.10 is the pre-registered minimum. **How to read it:** both neural-field dots sit above the line, but their intervals overlap kriging's, so the network is "at least as good", not proven better.*

![Calibration curve: empirical versus nominal interval coverage for kriging and the neural field](docs/figures/calibration_curve.png)

*Are the uncertainty intervals honest? For each nominal interval width (x-axis: 10 %, 20 %, … 95 %), the y-axis shows how many held-out Cu samples actually fell inside it. **How to read it:** a perfectly calibrated method lies on the dashed 1:1 line. Curves below the line are overconfident, meaning their intervals are too narrow. The neural field (red) stays closer to the line than kriging (blue), most clearly at the wide intervals used in practice.*

More figures (per-fold coverage, predicted vs observed, 3D block models, training curves) are in the [Keivitsa technical report](docs/2026-09_keivitsa_technical_report.md).

## How it works

![Method overview: GTK tables are desurveyed into a sample table with covariates; a Fourier-feature MLP ensemble predicts μ and σ; evaluation uses 10-fold hole-grouped cross-validation against the mean and kriging](docs/figures/method_overview.svg)

The network code is `build_mlp`, `train_member` and `ensemble_predict` in [`examples/keivitsa_common.jl`](examples/keivitsa_common.jl). Training uses AdamW (learning rate 10⁻³, weight decay 10⁻⁴) for up to 2,000 full-batch steps. The reasoning behind the stopping rule is in [`docs/keivitsa_data_notes.md`](docs/keivitsa_data_notes.md).

## Requirements

- **Julia 1.10 or newer.** The committed `Manifest.toml` was resolved with Julia 1.12.
- **GDAL** is installed automatically through ArchGDAL; no system GDAL is needed.
- **A display (OpenGL)** for the `figures_*.jl` scripts, which use GLMakie. On a headless server, run them under `xvfb-run`.
- **Time:** the density cross-validation took 3.2 h on a MacBook Air; Cu is shorter.

## Data

The Keivitsa data are not in this repository. They were downloaded from [Hakku](https://hakku.gtk.fi/en), the search and download service of GTK (Geological Survey of Finland).

Point `KEIVITSA_ROOT` at the unpacked GTK `source/gtk` folder. The adapter reads these files, relative to that folder (paths can be overridden in `sites/keivitsa.toml`):

| Table | Path |
|---|---|
| Collars | `report/3_DRILLINGS/Logs/Shape_files/collar.shp` (with `.dbf`, `.prj`) |
| Survey stations | `report/3_DRILLINGS/Logs/Shape_files/kalte.txt` |
| Cu assays, method 511P | `report/3_DRILLINGS/Assays/Shape_files/511P.txt` |
| Density and susceptibility | `report/3_DRILLINGS/Downhole_soundings_and_core_measurements/petro.txt` |

### Licence and attribution

The data are used under the [GTK basic licence](https://www.gtk.fi/en/basic-licence/), which allows use in academic publications but does not allow redistributing the data. For that reason no raw or derived data tables are committed, and the test fixture in `test/fixtures/keivitsa_tiny/` is synthetic. The figures in `docs/figures/` are derived from the data and carry this notice:

> **Keivitsa drillhole data (Hakku), edited © Geological Survey of Finland [2026].** The figures show model results derived from this material; they are not the original data.

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'

# sanity checks, no GTK data needed (seconds)
julia --project=. examples/check_kriging_sigma.jl
julia --project=. examples/check_binned_variogram.jl

# Keivitsa cross-validation; target = cu (default) | density | susceptibility
export KEIVITSA_ROOT=/path/to/gtk
julia --project=. examples/feasibility_keivitsa.jl
SMARTPRIOR_TARGET=density julia --project=. examples/feasibility_keivitsa.jl

# diagnostics, final block model, and report figures
julia --project=. examples/diagnose_nn_keivitsa.jl
julia --project=. examples/keivitsa_run2_checks.jl
julia --project=. examples/keivitsa_blockmodel.jl
julia --project=. examples/figures_data.jl
julia --project=. examples/figures_training.jl
julia --project=. examples/figures_results.jl
julia --project=. examples/figures_3d.jl
```

Outputs go to `tmp_feasibility_keivitsa/` for Cu and `tmp_feasibility_keivitsa_<target>/` for other targets (all gitignored). The script refuses a non-empty work directory and a leftover `.run.lock`, so two runs cannot share one output folder. It logs a runtime estimate after the first fold.

## Repository layout

| Path | Role |
|---|---|
| `src/SmartPrior.jl` | module entry |
| `src/Schema.jl` | `SampleTable` and `PropertySpec` |
| `src/Sites.jl` | `load_site`: site file → table + covariates |
| `src/Covariates.jl` | `CoordinateCovariate`, `DepthCovariate`, `DepthBelowSurface` |
| `src/Desurvey.jl` | minimum-curvature desurvey |
| `src/KeivitsaIO.jl` | GTK adapter to `SampleTable` |
| `src/SyntheticFields.jl` | seedable Gaussian fields |
| `sites/keivitsa.toml` | Keivitsa box, CRS (EPSG:2393), properties, covariates |
| `examples/keivitsa_common.jl` | shared variogram, kriging, network and bookkeeping code (included by the scripts below, not run on its own) |
| `examples/feasibility_keivitsa.jl` | 10-fold Keivitsa CV, target set by `SMARTPRIOR_TARGET` |
| `examples/feasibility_keivitsa_pilot.jl` | 30-hole pilot |
| `examples/keivitsa_inspect.jl` | data checks |
| `examples/diagnose_nn_keivitsa.jl` | early-stopping diagnosis on a synthetic target |
| `examples/keivitsa_run2_checks.jl` | byte compare of reruns, coverage, variograms |
| `examples/keivitsa_blockmodel.jl` | final Cu block model |
| `examples/figures_{data,training,results,3d}.jl` | report figures |
| `examples/check_kriging_sigma.jl` | kriging σ is a standard deviation (known case) |
| `examples/check_binned_variogram.jl` | streaming variogram bins equal stored-pair bins |
| `docs/2026-09_keivitsa_technical_report.md` | full Keivitsa write-up |
| `docs/keivitsa_data_notes.md` | GTK file conventions, statistics only |
| `docs/figures/` | figures cited in the README and the report |
| `test/` | package tests; `test/fixtures/keivitsa_tiny/` is a synthetic GTK-format fixture |
| `legacy/grid_stack/` | archived grid-based prior (`PriorGrid`, `PriorNet`, training loop); not loaded by the package |

## Notes

- The kriging variogram step counts point pairs without storing them (memory O(bins) instead of O(n²)). The stored-pair version ran out of memory on Keivitsa density, about 250 million pairs per fold; `check_binned_variogram.jl` shows the bins are bit-identical.
- Kriging (v1) pins the vertical range at the optimiser bound in most folds for both Cu and density. Kriging (v2) is the next baseline.
- `legacy/grid_stack/` keeps the earlier grid-based prior as its own module, `GridStack`. The Keivitsa pipeline does not use it. Its tests run with `julia --project=. legacy/grid_stack/test/runtests.jl` after `Pkg.add("JLD2")`; see [`legacy/grid_stack/README.md`](legacy/grid_stack/README.md) for known issues.

## Changelog

- **2026-09-28.** Removed an earlier, unrelated study site and its code; archived the unused grid stack under `legacy/`.
- **2026-09-25.** Kriging σ corrected. GeoStats.jl, in the version pinned here, returns ordinary-kriging predictions as `Normal(μ, σ²)`: the second parameter is the kriging *variance*. Earlier runs read it as the standard deviation, so every kriging σ and kriging coverage reported before this date was wrong (for example, a previously reported kriging coverage of 0.48 for Cu is really 0.78). Kriging means, RMSE and skill were unaffected, as were all neural-field results. `kriging_predict` now checks the behaviour on a known case (`examples/check_kriging_sigma.jl`), and all figures were regenerated.

## Licence and citation

The code is released under the MIT licence; see [`LICENSE`](LICENSE). The GTK data keep their own licence (see [Data](#data)).

If you use this code, please cite:

> Yıldız, H. (2026). *SmartPrior: neural-field priors with uncertainty from sparse drillholes.* https://github.com/hayrunnisayildiz/SmartPrior
