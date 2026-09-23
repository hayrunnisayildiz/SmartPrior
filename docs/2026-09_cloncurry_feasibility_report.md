# SmartPrior — Project status and Cloncurry feasibility report

*Date: 2026-09-23. Scope: work from the legacy audit through the Cloncurry leave-one-hole-out (LOHO) feasibility experiment.*

## 1. Summary

SmartPrior aims to be a site-independent Lux.jl neural-field method that predicts several rock properties (Cu grade, density, magnetic susceptibility, lithology probabilities) for every block of a 3D model, together with calibrated uncertainty, from sparse drillhole data and maps that are known everywhere.

After removing the MT-era code (Phase 0) and building a leakage-free, site-agnostic data layer (Phase 1), we ran a feasibility experiment on four Cloncurry deposits. **No method — ordinary kriging, IDW or the neural field — predicted held-out drillholes better than a constant training mean.** Pooled R² was negative for every method in every deposit–target combination except two (both ≤ 0.009). The neural field matched or slightly beat kriging only because it collapsed to the mean: early stopping selected near-initial weights in most folds (median best epoch 9 of up to 2000).

Diagnostics indicate a data limitation rather than a method failure: 16–59 % of the variance lies between holes, but holes are 100–370 m apart, 4–11 holes per deposit, and the available covariates carry no information about the subsurface at that scale. With the current Cloncurry data, any block model would reduce to "deposit mean plus wide uncertainty".

**Decision:** move the primary site to Keivitsa (GTK; 290 Cu-assayed holes at ~48 m median collar spacing, dense petrophysics, airborne and ground geophysics covering the drilled area). Cloncurry is retained as the sparse-regime reference.

## 2. Goal of the project

> Develop a site-independent neural-field method, written entirely in Julia with Lux.jl, that predicts several properties for each 3D block (grade, density, magnetic susceptibility, lithology probabilities) together with their uncertainty, from sparse drillhole data and maps known everywhere. The method is applied to a new site through configuration only and is compared with established methods such as kriging under the same spatial cross-validation.

Scope decisions made along the way:

- **Product:** a method (paper + Julia package), not an end-user tool.
- **Language:** Julia only; Lux.jl for the neural field; GeoStats.jl for the kriging baseline; GLMakie / WriteVTK for visualisation.
- **Kriging:** a baseline, not a component of the method (no kriging predictions as network inputs).
- **Generalisation** means that the same code and hyperparameters, trained separately at each site, behave sensibly everywhere. It does *not* mean transferring predictions between deposits: with coordinate inputs, a leave-one-deposit-out test would be meaningless.

## 3. Audit of the legacy pipeline

The project had drifted from its original purpose (warm-start priors for MT inversion) to a Cloncurry geochemistry block model. The audit of the district drillhole hold-out found:

- **Leakage.** The held-out holes' own pXRF, lithology and coverage channels were used as inputs; only the Cu labels were hidden.
- **Model selection on training loss.** The best epoch was chosen by `argmin(history.total)`; the validation split was computed but unused.
- **Misleading baselines.** "Beats naive" was measured against the training mean, which differed from the test mean by ~0.67 log10 units.

Test RMSE for log10 Cu (196 samples, 18 holes):

| Predictor | Test RMSE |
|---|---:|
| Training mean (legacy "naive") | 1.339 |
| Legacy neural field (Z = 100) | 1.263 |
| Deposit mean from training holes | 1.207 |
| Test set's own mean (constant) | 1.160 |

On the validation split the legacy network lost to the training mean for grade (1.377 vs 1.356) and susceptibility (1.288 vs 1.264). The driver script that produced the district results was not in the repository.

## 4. Work completed

### Phase 0 — cleanup

- MT-era state archived as tag `v0-mt-archive` (`49e387b`); package renamed to `SmartPrior`.
- Removed: `Gravity.jl`, `MT1DAD.jl`, `Profile2D.jl`, `RealDataIO.jl`, `Export.jl`, `KeivitsaIO.jl`, `Synthetic.jl`, the physics losses, Keivitsa / Musgrave / `compare_prior_2d` examples, Python renderers, and the `MTGeophysics`, `LsqFit`, `Plots` dependencies.
- Added `GLMakie`, `GeoStats` 0.90, `WriteVTK`. README and `.cursor/rules` rewritten for the new goal.

### Phase 1 — site-agnostic data layer

Added alongside the legacy training path, which was left untouched (commits `1423674` … `b7a0f8b`; 509/509 tests passing after fixes):

- `Schema.jl`: `PropertySpec`, `SampleTable` (coordinates, hole and deposit IDs, per-property values and censoring flags), `training_mask`.
- `Covariates.jl`: point-wise covariates (`CoordinateCovariate`, `DepthCovariate`, `StructureDistance`, `SurfaceGeology`). Normalisation statistics are fixed at construction, so `evaluate` is independent of the query set. Equivalence with the legacy grid channels holds to `atol = 1e-10`.
- `Sites.jl` and TOML site configs (`sites/ernest_henry.toml`, later `cannington`, `starra`, `osborne`).
- A Cloncurry adapter producing `SampleTable`.

Censoring policies (district data):

| Property | Policy | Limit | Censored (district / Ernest Henry) |
|---|---|---|---|
| Cu | 1st percentile of 962 positive values | 5.8 ppm | 223 (18.8 %) / 53 (20.8 %) |
| Conductivity (100 kHz) | 1st percentile of 539 positive values | 0.0356 S/m | 711 (56.9 %) / 220 (88.4 %) |
| Susceptibility | non-positive → below smallest positive | 7.44 × 10⁻⁷ SI | 1 / 0 |

The previous policy mapped Cu `<LOD` to 1 ppm; only 6 positive Cu values lie below 5 ppm, so 1 ppm (and the 1.5 ppm minimum) would have asserted an unrealistically tight bound.

## 5. Feasibility experiment

**Question.** Can a neural field trained at sample points on everywhere-known inputs compete with kriging at deposit scale?

**Design** (`examples/feasibility_loho.jl`, outputs in `tmp_feasibility/`):

- Deposits: Ernest Henry, Cannington, Starra, Osborne. Targets: log10 Cu, density, log10 susceptibility.
- Leave-one-hole-out: 89 folds, 9,095 prediction rows, 239 s. The test hole is excluded from training, variogram fitting, standardisation and early stopping.
- Methods:
  - `mean`: training mean.
  - `idw`: power 2, 16 neighbours, z scaled by the variogram anisotropy ratio.
  - `kriging`: GeoStats.jl ordinary kriging; variogram refitted per fold on training holes only (spherical or exponential, geometric anisotropy from downhole and horizontal pairs), 16 neighbours.
  - `nn_xyz` / `nn_cov`: Lux MLP (width 64, depth 3, GELU) on Fourier-encoded xyz (16 log-spaced scales in [1, 16]), with or without depth / structure-distance / surface-geology covariates. Heteroscedastic Gaussian NLL, AdamW (lr 1e-3, weight decay 1e-4), full batch, ≤ 2000 epochs, early stopping on 20 % of training holes (patience 200), 5-member ensemble.
- Hyperparameters were fixed before the run and not tuned.
- Censored values were set to the censoring limit for all methods (kriging cannot use a censored likelihood); metrics are reported on all and on uncensored samples.

## 6. Results

### 6.1 Accuracy

Pooled LOHO RMSE, all samples (best per row in bold):

| Deposit | Target | n | mean | IDW | kriging | nn_xyz | nn_cov |
|---|---|---:|---:|---:|---:|---:|---:|
| Ernest Henry | cu | 255 | 1.227 | 1.394 | 1.251 | **1.186** | 1.191 |
| Ernest Henry | density | 253 | 0.275 | 0.287 | 0.312 | **0.272** | 0.274 |
| Ernest Henry | susceptibility | 253 | 0.942 | 0.957 | 0.975 | **0.875** | 0.892 |
| Cannington | cu | 160 | 0.854 | 0.957 | 0.960 | 0.873 | **0.854** |
| Cannington | density | 162 | **0.456** | 0.734 | 0.598 | 0.461 | 0.464 |
| Cannington | susceptibility | 162 | **1.217** | 1.721 | 1.513 | 1.257 | 1.250 |
| Starra | cu | 106 | 0.911 | 1.106 | 0.950 | **0.909** | 0.915 |
| Starra | density | 109 | 0.741 | 0.829 | 0.778 | 0.732 | **0.720** |
| Starra | susceptibility | 110 | 1.214 | 1.219 | **1.195** | 1.212 | 1.273 |
| Osborne | cu | 83 | 1.178 | 1.201 | 1.195 | 1.168 | **1.165** |
| Osborne | density | 83 | **0.786** | 0.809 | 0.876 | 0.815 | 0.807 |
| Osborne | susceptibility | 83 | **1.598** | 2.201 | 2.001 | 1.762 | 1.743 |

Median pooled R² by method:

| Method | Median R² | Range |
|---|---:|---|
| mean | −0.133 | −0.255 … −0.013 |
| IDW | −0.369 | −1.999 … −0.052 |
| kriging | −0.338 | −0.991 … −0.042 |
| nn_xyz | −0.127 | −0.420 … 0.004 |
| nn_cov | −0.128 | −0.390 … 0.009 |

Where the neural field has the lowest RMSE, its margin over `mean` is small (≤ 0.07). Pooled LOHO R² is biased negative even for the mean predictor, because removing a hole shifts the training mean away from that hole's values; RMSE relative to `mean` is therefore the more informative comparison.

### 6.2 Paired comparison with kriging

Per-hole RMSE difference (NN − kriging), hole bootstrap (2000 resamples). Of 24 comparisons (4 deposits × 3 targets × 2 NN variants, all samples), 22 have a 95 % interval that includes zero. The two exceptions favour the network:

| Deposit | Target | Variant | Mean diff. | 95 % CI | Holes won |
|---|---|---|---:|---|---:|
| Ernest Henry | density | nn_cov | −0.044 | [−0.091, −0.003] | 7 / 10 |
| Osborne | susceptibility | nn_xyz | −0.210 | [−0.426, −0.006] | 4 / 5 |

### 6.3 Uncertainty calibration

Coverage of the nominal 90 % interval:

| Deposit | Target | kriging | nn_xyz | nn_cov |
|---|---|---:|---:|---:|
| Ernest Henry | cu | 0.84 | 0.79 | 0.82 |
| Ernest Henry | density | 0.29 | 0.84 | 0.84 |
| Ernest Henry | susceptibility | 0.70 | 0.86 | 0.81 |
| Cannington | cu | 0.96 | 0.87 | 0.87 |
| Cannington | density | 0.60 | 0.85 | 0.85 |
| Cannington | susceptibility | 0.96 | 0.81 | 0.84 |
| Starra | cu | 0.75 | 0.62 | 0.64 |
| Starra | density | 0.82 | 0.70 | 0.75 |
| Starra | susceptibility | 0.85 | 0.65 | 0.76 |
| Osborne | cu | 0.94 | 0.71 | 0.81 |
| Osborne | density | 0.54 | 0.71 | 0.76 |
| Osborne | susceptibility | 0.86 | 0.80 | 0.84 |

Both methods under-cover in most cases (median: kriging 0.83, nn_xyz 0.79, nn_cov 0.81). Kriging is more erratic (0.29–0.96); the network is more uniform but too narrow at Starra and Osborne.

## 7. Diagnostics

The following were computed from `predictions.tsv`, `kriging_params.tsv` and `run.log` after the run; they are not produced by the feasibility script.

### 7.1 Where the variance is, and whether any method uses it

| Deposit | Target | Holes | Median NN hole spacing (m) | Between-hole variance share | Oracle hole-mean RMSE | mean RMSE | Depth-linear RMSE | Pred. SD kriging | Pred. SD nn_xyz | Obs. SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ernest Henry | cu | 11 | 361 | 0.32 | 0.969 | 1.227 | 1.292 | 0.84 | 0.24 | 1.18 |
| Ernest Henry | density | 10 | 372 | 0.21 | 0.234 | 0.275 | 0.286 | 0.14 | 0.05 | 0.26 |
| Ernest Henry | susceptibility | 10 | 372 | 0.53 | 0.577 | 0.942 | 0.996 | 0.22 | 0.11 | 0.85 |
| Cannington | cu | 10 | 167 | 0.23 | 0.732 | 0.854 | 0.843 | 0.36 | 0.15 | 0.84 |
| Cannington | density | 10 | 164 | 0.59 | 0.270 | 0.456 | 0.459 | 0.35 | 0.07 | 0.43 |
| Cannington | susceptibility | 10 | 164 | 0.35 | 0.945 | 1.217 | 1.250 | 0.79 | 0.25 | 1.18 |
| Starra | cu | 5 | 100 | 0.16 | 0.794 | 0.911 | 0.931 | 0.49 | 0.22 | 0.87 |
| Starra | density | 4 | 197 | 0.27 | 0.575 | 0.741 | 0.707 | 0.27 | 0.13 | 0.68 |
| Starra | susceptibility | 4 | 197 | 0.34 | 0.884 | 1.214 | 1.644 | 0.50 | 0.21 | 1.09 |
| Osborne | cu | 5 | 139 | 0.02 | 1.161 | 1.178 | 1.178 | 0.41 | 0.26 | 1.18 |
| Osborne | density | 5 | 133 | 0.19 | 0.630 | 0.786 | 1.828 | 0.37 | 0.25 | 0.71 |
| Osborne | susceptibility | 5 | 133 | 0.18 | 1.340 | 1.598 | 1.669 | 0.92 | 0.56 | 1.49 |

- **Between-hole variance share** is 1 − (within-hole variance / total variance). The oracle hole-mean RMSE is the error achievable if each hole's mean were known exactly: an upper bound on what inter-hole interpolation could achieve.
- A gap exists between the oracle and `mean`, but no method closes any meaningful part of it. A per-fold linear depth trend does not beat the mean either.
- Kriging predictions vary substantially (SD 0.14–0.92) but in the wrong places. The network's predictions are nearly constant (SD 0.05–0.56), which is why its RMSE tracks the mean.

### 7.2 Early stopping

Across 445 member trainings per variant, the best validation epoch had a median of 9 (xyz) and 8 (cov). It was ≤ 5 in 34 % / 39 % of runs and exactly 1 in 24 % / 17 %. The validation loss on held-out training holes is minimised almost at initialisation, which, after target standardisation, corresponds to predicting the training mean. This is consistent with the absence of transferable inter-hole signal, but it also means the NN results are effectively a mean predictor and say little about the network's capacity.

### 7.3 Variogram fits

All 89 fits were non-degenerate (partial sill ≥ 5 % of total); 87 were spherical and 2 exponential. Several fits are nonetheless implausible. Ernest Henry Cu and density have a median horizontal range of 6,941 m, which appears to be the fitting upper bound. Cannington vertical ranges are ~1,076 m, longer than the horizontal ones. Some Osborne folds have ranges of 12–35 m. Kriging's poor performance may partly reflect unstable variogram fitting with 4–11 holes, but the same failure of IDW and the network argues against fitting being the main cause.

### 7.4 Data quality

Several "hole" IDs are sample-type placeholders rather than drillholes:

| Deposit | ID | Cu samples |
|---|---|---:|
| Osborne | `Block` | 35 |
| Osborne | `core` | 13 |
| Osborne | `Core` | 1 |
| Starra | `outcrop` | 24 |

Osborne therefore has only two named drillholes (`OSHQ0067`, `TTNQ364`); the remaining "holes" mix unrelated samples. The case-variant pair `Core`/`core` may also split one group across train and test. Osborne results, and the Starra `outcrop` fold, should be treated as unreliable.

## 8. Interpretation

1. **The limitation is data density relative to the scale of variability, not the method.** Most variance is short-range (within holes); holes are 100–370 m apart with 4–11 per deposit. Kriging, IDW and the neural field all fail in the same way.
2. **Covariates add nothing at this scale.** `nn_cov` and `nn_xyz` are indistinguishable. Ernest Henry is under ~50 m of cover, and surface geology and structure distance do not describe the subsurface properties.
3. **The neural field behaves correctly in the absence of signal**: it shrinks toward the mean. This avoids kriging's large errors but is not evidence that the method works.
4. For this data, a 3D block model of any method would be essentially "deposit mean plus wide uncertainty".

## 9. Limitations and errors

- **Flawed decision rule.** The pre-registered rule was "proceed if the NN is comparable to kriging". It was met trivially, because neither method beats the mean. The rule should also have required both methods to beat the mean. The corrected rule is in §10.
- **Censoring simplified.** Censored values were fixed at the limit for all methods; the censored likelihood (Phase 2) was not tested.
- **Single configuration.** One set of NN hyperparameters; one variogram-fitting procedure. The results are conclusive about *this* data but do not rule out gains from much stronger covariates.
- **Placeholder hole IDs** (§7.4) affect Osborne and part of Starra.
- **No block-level validation.** Evaluation is at sample support.

## 10. Decision and next steps

**Primary site: Keivitsa (GTK).** Data inspection shows what Cloncurry lacks:

| | Cloncurry (Ernest Henry) | Keivitsa |
|---|---|---|
| Cu samples | 255 | 16,005 (single method, 511P) |
| Holes with Cu | 11 | 290 |
| Cu below LOD | 21 % | 0.5 % |
| Median NN collar spacing | ~360 m | 48 m |
| Density / susceptibility | ~250 | 10,412 core (122 holes) + 20,496 further (141 holes) |
| Everywhere-known covariates | surface geology, structures | airborne magnetics, EM, radiometrics; ground gravity (~29k stations), ground magnetics, IP, slingram, VLF |

Known Keivitsa issues:

- **Sample coordinates are not desurveyed.** The assay tables repeat collar coordinates; positions must be computed from the survey table (`kalte.txt`).
- **Coordinates are in KKJ (EPSG:2393) with X = north**; a few collars have invalid coordinates.
- **Angle units** (degrees vs gon) must be verified from the GTK documentation.
- **Licence.** The GTK basic licence permits internal use and images in scientific publications, but not redistribution of the data. Raw or derived data must not enter the repository or test fixtures.

Plan:

1. **Keivitsa adapter and desurvey** (minimum curvature) into the Phase 1 schema. Validate conventions against the GTK documentation.
2. **Repeat the feasibility experiment on Keivitsa** with the corrected rule: proceed only if the neural field beats the training mean clearly *and* is comparable to or better than kriging. Also record per-member best epochs and the fraction of folds stopping at epoch ≤ 5.
3. **Hole-thinning experiment**: train on subsets of Keivitsa holes to estimate the hole spacing at which location-based prediction starts to beat the mean. The Cloncurry result serves as the sparse-regime reference point.
4. Then Phase 2 (point-based training, censored likelihood, multi-output heads, ensembles) on Keivitsa.

Cloncurry is retained as a secondary site. Its placeholder hole IDs should be fixed in the adapter (map `Block`, `core`, `Core`, `outcrop` to a sample-type field and exclude them from hole-grouped CV).

## Appendix — reproducibility

- Tag `v0-mt-archive` (`49e387b`): MT-era state.
- Phase 1 commits: `1423674`, `fb9ef09`, `2a0874e`, `a452ac9`, `7a4d3f3`, `b7a0f8b`, plus the fix commits.
- Feasibility script: `examples/feasibility_loho.jl` (commit: *to be filled in*).
- Data: `Cloncurry_integrated_2026-09-17`.
- Outputs: `tmp_feasibility/` (`summary.tsv`, `paired.tsv`, `kriging_params.tsv`, `predictions.tsv`, `run.log`).
- Section 7 diagnostics were computed from these outputs outside the Julia package. They should be reimplemented in Julia (e.g. `examples/feasibility_diagnostics.jl`) before being cited in a paper.
