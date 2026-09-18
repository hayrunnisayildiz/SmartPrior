"""
    SmartPriorMT

Uncertainty-aware starting models ("smart priors") for magnetotelluric inversion.

The package turns complementary geophysical data (gravity, topography, shallow
structural constraints) into a per-cell nominal resistivity `mu` and a per-cell
confidence `sigma` on the same tensor grid that MTGeophysics.jl inverts on. The
pair defines both a starting model and a per-cell search interval, which narrows
the space a stochastic solver such as VFSA has to explore.

The mapping is learned by a Lux.jl neural field trained against differentiable
physics: a closed-form prism gravity operator and an AD-safe 1D MT recursion.

## Validation status (read before relying on a code path)

- **2D warm start (`mu` only): validated.** `to_mt2d` converts the learned prior
  to MTGeophysics' 2D mesh convention (linear ohm-m, axis order, air-layer
  offset) and the result has been run end to end through
  `MTGeophysics.run_mt2d_vfsa` / `VFSA2DMT` as the literal starting model; see
  `examples/compare_prior_2d.jl` and `examples/musgrave_vfsa_compare.jl`.
  Quantitative results (3 seeds, data RMS and correlation vs true model)
  are in README.md's "Ampirik doğrulama (2B)" section. Improvement is
  consistent for data fit and structural correlation; absolute resistivity
  amplitude (model RMSE) shows no consistent gain, and a 4-scenario sweep
  shows the prior underperforms a naive baseline in dip55/resistive-contrast
  geologies.
- **Per-cell search-interval narrowing (`sigma`, via `BoundedVFSA.jl`): not
  connected to a real solver run, and not possible in this MTGeophysics
  version.** `prior_bounds` and `BoundedVFSA.jl` compute and report what the
  per-cell bounds *would* narrow the search to (`bound_report`), but
  MTGeophysics v0.5.0's `VFSA2DMTConfig`/`VFSA3DMTConfig.log_bounds` only
  accepts a scalar `Tuple{Float64,Float64}` -- confirmed by inspecting
  `_propose_controls!`/`clamp.` and by `MethodError` on an array-valued
  `log_bounds`. Every real run so far uses one scalar interval for the whole
  grid. This is an MTGeophysics API gap, not a SmartPriorMT bug; see
  `BoundedVFSA.jl`'s docstring.
- **3D warm start (WS3D export via `write_prior`, `VFSA3DMT`): not implemented
  or validated, and currently out of scope.** No test or example in this
  package calls a 3D MTGeophysics inversion. `write_prior` writes `mu` (log10
  resistivity) directly into a WS3D file with no unit conversion, unlike the
  2D bridge; whether that matches MTGeophysics' actual WS3D convention has not
  been confirmed against MTGeophysics' own source. Do not assume `prior.rho` is
  a correct 3D starting model until this is checked and an end-to-end 3D run
  exists.
- **Known issue:** `examples/robustness_misspecified_gravity.jl` throws
  `UndefVarError: BoundedCore` partway through (sections 8-9 do not run);
  because output is piped through `tee`, the run still reports exit code
  0. The inversion comparison itself (sections through data RMS/model
  RMSE/correlation) completes successfully before the error. Not yet
  fixed.
"""
module SmartPriorMT

using LinearAlgebra
using Printf
using Random
using Statistics

using Lux
using Lux: Chain, Dense, gelu, sigmoid
using Random: AbstractRNG, Xoshiro

import JLD2
import Optimisers
import Zygote

using MTGeophysics: WS3DModel, load_ws3d_model, write_ws3d_model
using MTGeophysics: RBFMap, build_rbf_map, apply_rbf_map!

include("Grid.jl")
include("Gravity.jl")
include("MT1DAD.jl")
include("Features.jl")
include("KeivitsaIO.jl")
include("PriorNet.jl")
include("Losses.jl")
include("Train.jl")
include("Export.jl")
include("BoundedVFSA.jl")
include("Synthetic.jl")
include("Metrics.jl")
include("Profile2D.jl")
include("RealDataIO.jl")

# grid
export PriorGrid
export ncells, cell_volumes, cell_centers, normalized_centers, depth_below_top
export containing_cell

# gravity
export prism_gz, gravity_matrix, gravity_cell_sensitivity, forward_gravity

# 1-D MT physics (AD-safe)
export mt1d_impedance, mt1d_apparent, mt1d_column_response
export skin_depth, bostick_depth, bostick_resistivity, niblett_bostick

# observations and features
export GravityObs, MTSites, FeatureStack, PointSamples, LabelSamples
export nsites, nchannels, build_features, feature_matrix, nb_baseline
export nb_baseline_lateral_std, residual_span_half_band, residual_span_from_baseline
export standardize, extrude, idw_to_grid, gaussian_smooth_xy, gradient_xy
export gravity_channels, topography_channels, depth_channels, coverage_channels
export gravity_sensitivity_channel
export nearest_sample_channels, geochemistry_channels, lithology_channels

# neural field
export PriorNet, setup_prior, predict, predict_grid
export fourier_encode, encode_features

# training objective
export GravityCoupling, density_from_mu, physical_slope, DENSITY_SCALE
export heteroscedastic_nll, gravity_misfit, mt_column_misfit, smoothness, reference_penalty, sigma_penalty
export SigmaDriveConfig, mt_column_residuals, compute_sigma_targets
export LossWeights, PriorTargets, prior_loss, loss_report
export sigma_bounds_from_anchors

# training
export TrainConfig, TrainResult, PriorEnsemble
export init_params, coupling_of, train_prior, train_ensemble
export predict_ensemble, predict_ensemble_grid
export save_prior, load_prior

# export to the inversion's format
export PriorBundle, prior_bounds, apply_air_mask, write_prior, prior_from_ensemble

# synthetic ground truth
export SyntheticModel, truth_halfspace, truth_vector
export add_layer, add_block, add_dipping_slab, add_topography
export synth_gravity, synth_mt_sites

# metrics
export coverage, volume_reduction, rmse, mae, anomaly_correlation, calibration
export prior_report, print_report, compare_starts

# 2-D profile bridge
export grid_from_mt2dmesh, to_mt2d, from_mt2d, profile_sites

# field EDI
export read_musgrave_edi, build_musgrave_datafile2d, musgrave_phase_tensor_skew

# Keivitsa (GTK) non-geophysical prior line
export cu_log10, merge_petro_pair, read_petro_txt, petro_coverage
export read_keivitsa_grid_bounds, keivitsa_grid, keivitsa_report_root
export load_keivitsa_collars, load_keivitsa_surveys
export load_keivitsa_geochemistry, load_keivitsa_lithology
export load_keivitsa_drill_geochemistry, combine_geochemistry
export keivitsa_cleaned_intervals_path
export load_keivitsa_grade_anchors, load_keivitsa_petrophysics_anchors
export map_points_to_cells, aggregate_to_cells, desurvey_depths
export KEIVITSA_PETRO_STATUS, KEIVITSA_GEOCHEM_PRIORITY, KEIVITSA_DRILL_GEOCHEM

end # module
