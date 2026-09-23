"""
    SmartPrior

Julia/Lux.jl neural field for 3D block models: predict grade, density,
magnetic susceptibility, and (where configured) other rock properties with
heteroscedastic uncertainty from **covariates that are known everywhere**
in the model box — normalised coordinates, depth, distance to mapped
structures, and surface geology — plus optional semi-synthetic fields for
method tests.

**Not inputs (leakage):** pXRF geochemistry, drillhole lithology, and
distance to the nearest sample. Those are training labels or confidence
masks only; [`load_site`](@ref) rejects them as covariates.

**Not a geophysical inversion:** gravity, magnetics, and MT are neither
inputs nor outputs here.

**Sites:** [`load_site`](@ref) loads a [`SampleTable`](@ref) from TOML under
`sites/` — **Keivitsa** (GTK; primary) and **Cloncurry** (METAL; sparse-regime
reference). Desurvey is in [`Desurvey.jl`](@ref); site readers are
[`KeivitsaIO.jl`](@ref) and [`CloncurryIO.jl`](@ref). The older grid stack
(`Grid.jl`, `Features.jl`, `examples/*_cloncurry_prior.jl`) is unchanged and
still uses geochemistry / lithology / coverage as features; see the README
legacy-path note.
"""
module SmartPrior

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

using ArchGDAL
using CodecZlib
using TOML

include("Grid.jl")
include("Features.jl")
include("Schema.jl")
include("Desurvey.jl")
include("Covariates.jl")
include("SyntheticFields.jl")
include("CloncurryIO.jl")
include("KeivitsaIO.jl")
include("Sites.jl")
include("PriorNet.jl")
include("Losses.jl")
include("Train.jl")
include("Metrics.jl")

# grid
export PriorGrid
export ncells, cell_volumes, cell_centers, normalized_centers, depth_below_top
export containing_cell
export cu_log10, aggregate_to_cells, map_points_to_cells

# observations and features
export PropertySpec, SampleTable, nsamples, subset, observed_mask, training_mask
export real_hole_mask
export Covariate, evaluate, channel_names, evaluate_all
export CoordinateCovariate, DepthCovariate, DepthBelowSurface, StructureDistance, SurfaceGeology
export FeatureStack, PointSamples, LabelSamples
export nchannels, build_features, feature_matrix
export standardize, extrude, idw_to_grid, gaussian_smooth_xy, gradient_xy
export topography_channels, depth_channels, coverage_channels
export nearest_sample_channels, geochemistry_channels, lithology_channels
export append_channels

# neural field
export PriorNet, setup_prior, predict, predict_grid
export fourier_encode, encode_features

# training objective
export heteroscedastic_nll, smoothness, reference_penalty, sigma_penalty
export LossWeights, PriorTargets, prior_loss, loss_report
export sigma_bounds_from_anchors

# training
export TrainConfig, TrainResult, PriorEnsemble
export init_params, train_prior, train_ensemble
export predict_ensemble, predict_ensemble_grid
export save_prior, load_prior

# metrics
export coverage, volume_reduction, rmse, mae, anomaly_correlation, calibration

# Cloncurry–Ernest Henry
export CloncurrySamples, load_cloncurry_samples
export cloncurry_derived_dir, cloncurry_petrophysics_path, cloncurry_metals_path
export cloncurry_work_bounds, cloncurry_deposit_bounds, cloncurry_sample_bounds
export cloncurry_district_spacing, cloncurry_grid
export cloncurry_geochemistry, cloncurry_lithology, cloncurry_coverage_points
export cloncurry_property_coverage, load_cloncurry_anchors
export cloncurry_grade_eligible, cloncurry_inside_mask, spatial_holdout, group_holdout
export cloncurry_group_keys, group_mask
export cloncurry_geology_dir, cloncurry_structures_path, cloncurry_surface_geology_path
export load_cloncurry_structures, load_cloncurry_surface_geology
export structure_distance_channels, surface_geology_channels
export cloncurry_sample_table
export cloncurry_sample_type, CLONCURRY_PLACEHOLDER_HOLE_IDS
export CLONCURRY_PROPERTY_NAMES, CLONCURRY_GEOCHEM_ELEMENTS
export CLONCURRY_SULFIDE_INDEX_ELEMENTS, CLONCURRY_GEOCHEM_RATIOS
export CLONCURRY_CONDUCTIVITY_STATUS, CLONCURRY_WORK_BOUNDS
export CLONCURRY_CU_DL_PPM, CLONCURRY_COND_FLOOR_S_M

# desurvey
export DesurveyPath, desurvey, positions

# Keivitsa
export keivitsa_sample_table, KeivitsaReport, KeivitsaExclusion, exclusion_summary

# site files
export load_site

# synthetic fields
export GaussianField, gaussian_field, latent_field, smooth_field, SYNTHETIC_RFF_M

end # module
